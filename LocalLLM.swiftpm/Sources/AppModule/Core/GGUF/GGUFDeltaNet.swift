import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Gated DeltaNet（Qwen3.5 の 4 層のうち 3 層）。注意の代わりに、ヘッドごとに「キー × バリュー」の大きさの状態を持ち、
/// トークンごとに次のように更新する（llama.cpp の ggml_gated_delta_net と同じ）:
///
///     S ← S × exp(g)                      … 古い記憶を減らす
///     δ = (v − S·k) × β                   … 今の記憶で k から思い出せる値と v のずれ
///     S ← S + δ ⊗ k                       … ずれの分だけ書き込む
///     出力 = S·q / √(ヘッドの大きさ)
///
/// 入力は、q・k・v をまとめた射影に短い畳み込み（直前の数トークンを混ぜる）と SiLU を掛けたもの。q・k は L2 正規化する。
/// バリューのヘッド j は、キーのヘッド j % (キーのヘッドの数) を使う。出力はヘッドごとに RMSNorm して silu(z) を掛ける。
/// 状態 S は「バリューの次元 × キーの次元」で置く（行 = バリューの 1 つ、行の中がキーの次元）。
final class GGUFDeltaNetLayer {
    typealias In = UnsafePointer<Float>
    typealias Out = UnsafeMutablePointer<Float>

    let sizes: GGUFModelConfig.DeltaNet
    /// [バリューのヘッド][バリューの次元][キーの次元]
    let state: Out
    /// 畳み込みの直前の入力 [長さ − 1][チャンネル]
    let history: Out

    init(sizes: GGUFModelConfig.DeltaNet) {
        self.sizes = sizes
        state = .allocate(capacity: sizes.stateFloats)
        history = .allocate(capacity: max(1, (sizes.convolutionLength - 1) * sizes.channels))
        reset()
    }

    deinit {
        state.deallocate()
        history.deallocate()
    }

    var historyFloats: Int { (sizes.convolutionLength - 1) * sizes.channels }

    func reset() {
        state.update(repeating: 0, count: sizes.stateFloats)
        history.update(repeating: 0, count: max(1, historyFloats))
    }

    /// 今の状態を写す（巻き戻し用）
    func save(into target: Out) {
        target.update(from: state, count: sizes.stateFloats)
        (target + sizes.stateFloats).update(from: history, count: historyFloats)
    }

    func restore(from source: In) {
        state.update(from: source, count: sizes.stateFloats)
        history.update(from: source + sizes.stateFloats, count: historyFloats)
    }

    var savedFloats: Int { sizes.stateFloats + historyFloats }

    /// 作業用のメモリ
    final class Work {
        let mixed = GGUFBuffer(), z = GGUFBuffer(), beta = GGUFBuffer(), decay = GGUFBuffer(), core = GGUFBuffer()
        let projected = GGUFBuffer(), scratch = GGUFBuffer(), vector = GGUFBuffer()
    }

    /// n トークンぶん。projected = q・k・v の射影 [n][チャンネル]、z = 出力のゲートの射影 [n][バリュー全体]、
    /// betaRaw・alphaRaw = [n][バリューのヘッド]。結果（出力の射影の前）を core に書く [n][バリュー全体]
    func run(count n: Int, projected: In, z: Out, betaRaw: Out, alphaRaw: Out, weights: GGUFDeltaNetWeights,
             kernel: In, epsilon: Float, into core: Out, work: Work) {
        let channels = sizes.channels
        let mixed = work.mixed.reserve(n * channels)
        convolve(count: n, projected: projected, kernel: kernel, into: mixed)
        GGUFKernels.activate(.silu, mixed, count: n * channels, scratch: work.scratch.reserve(n * channels))
        normalizeKeys(mixed, count: n, epsilon: epsilon)
        gates(count: n, betaRaw: betaRaw, alphaRaw: alphaRaw, weights: weights)
        let vector = work.vector.reserve(2 * sizes.valueHeadSize)
        for token in 0..<n {
            step(row: mixed + token * channels, beta: betaRaw + token * sizes.valueHeads, decay: alphaRaw + token * sizes.valueHeads,
                 into: core + token * sizes.valueSize, scratch: vector)
        }
        // ヘッドごとの RMSNorm × silu(z)
        GGUFKernels.rmsNorm(core, weight: weights.norm, rows: n * sizes.valueHeads, columns: sizes.valueHeadSize,
                            epsilon: epsilon, into: core)
        GGUFKernels.activate(.silu, z, count: n * sizes.valueSize, scratch: work.scratch.reserve(n * sizes.valueSize))
        MiniVector.multiply(core, z, core, n * sizes.valueSize)
    }

    /// 短い畳み込み: 出力[t][c] = Σ_i 入力[t + i][c] × 重み[c][i]（入力 = 直前の (長さ − 1) 行 + 新しい n 行）。直前の行を更新する
    private func convolve(count n: Int, projected: In, kernel: In, into output: Out) {
        let channels = sizes.channels, length = sizes.convolutionLength
        let keep = length - 1
        for token in 0..<n {
            let target = output + token * channels
            MiniVector.fillZero(target, channels)
            for tap in 0..<length {
                let source = token + tap   // 並べた入力の中の位置（keep より前は直前の行）
                let row: In = source < keep ? In(history + source * channels) : projected + (source - keep) * channels
                Self.multiplyStrided(row, kernel + tap, stride: length, accumulate: target, count: channels)
            }
        }
        guard keep > 0 else { return }
        // 直前の (長さ − 1) 行を、並べた入力の最後の行にする
        if n >= keep {
            history.update(from: projected + (n - keep) * channels, count: keep * channels)
        } else {
            let shift = keep - n
            history.update(from: history + n * channels, count: shift * channels)
            (history + shift * channels).update(from: projected, count: n * channels)
        }
    }

    /// accumulate[c] += row[c] × kernel[c × stride]
    private static func multiplyStrided(_ row: In, _ kernel: In, stride: Int, accumulate: Out, count: Int) {
        #if canImport(Accelerate)
        vDSP_vma(row, 1, kernel, vDSP_Stride(stride), accumulate, 1, accumulate, 1, vDSP_Length(count))
        #else
        for index in 0..<count { accumulate[index] += row[index] * kernel[index * stride] }
        #endif
    }

    /// q・k をヘッドごとに L2 正規化する（llama.cpp: rms_norm(x, eps / n) / √n = x / √(Σx² + eps)）
    private func normalizeKeys(_ mixed: Out, count n: Int, epsilon: Float) {
        let size = sizes.keyHeadSize, heads = sizes.keyHeads
        let scale = 1 / Float(size).squareRoot()
        for token in 0..<n {
            let row = mixed + token * sizes.channels
            for part in 0..<2 {
                let base = row + part * sizes.keySize
                GGUFKernels.rmsNorm(base, weight: nil, rows: heads, columns: size, epsilon: epsilon / Float(size), into: base)
                MiniVector.scale(base, scale, base, sizes.keySize)
            }
        }
    }

    /// β = sigmoid(betaRaw)、減衰 = exp(softplus(alphaRaw + bias) × a)（a は負なので 0〜1）
    private func gates(count n: Int, betaRaw: Out, alphaRaw: Out, weights: GGUFDeltaNetWeights) {
        let heads = sizes.valueHeads
        for token in 0..<n {
            for head in 0..<heads {
                let index = token * heads + head
                betaRaw[index] = 1 / (1 + Foundation.exp(-betaRaw[index]))
                let x = alphaRaw[index] + weights.timeStepBias[head]
                // ggml の softplus: 大きいときはそのまま
                let softplus = x > 20 ? x : Foundation.log(1 + Foundation.exp(x))
                alphaRaw[index] = Foundation.exp(softplus * weights.decay[head])
            }
        }
    }

    /// 1 トークンぶんの状態の更新と出力。row = [q | k | v]、scratch は 2 × バリューの大きさ
    private func step(row: In, beta: In, decay: In, into output: Out, scratch: Out) {
        let size = sizes.keyHeadSize, valueSize = sizes.valueHeadSize
        let keyHeads = sizes.keyHeads
        let outputScale = 1 / Float(valueSize).squareRoot()
        let recalled = scratch, delta = scratch + valueSize
        for head in 0..<sizes.valueHeads {
            let keyHead = head % keyHeads
            let q = row + keyHead * size
            let k = row + sizes.keySize + keyHead * size
            let v = row + 2 * sizes.keySize + head * valueSize
            let matrix = state + head * valueSize * size
            // S ← S × exp(g)
            MiniVector.scale(matrix, decay[head], matrix, valueSize * size)
            // δ = (v − S·k) × β
            MiniMath.gemm(transA: false, transB: false, m: valueSize, n: 1, k: size, alpha: 1,
                          a: matrix, lda: size, b: k, ldb: 1, beta: 0, c: recalled, ldc: 1)
            MiniVector.scale(recalled, -1, delta, valueSize)
            MiniVector.add(delta, v, delta, valueSize)
            MiniVector.scale(delta, beta[head], delta, valueSize)
            // S ← S + δ ⊗ k
            MiniMath.gemm(transA: false, transB: false, m: valueSize, n: size, k: 1, alpha: 1,
                          a: delta, lda: 1, b: k, ldb: size, beta: 1, c: matrix, ldc: size)
            // 出力 = S·q / √size
            MiniMath.gemm(transA: false, transB: false, m: valueSize, n: 1, k: size, alpha: outputScale,
                          a: matrix, lda: size, b: q, ldb: 1, beta: 0, c: output + head * valueSize, ldc: 1)
        }
    }
}
