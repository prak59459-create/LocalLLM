import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// 重みの行列（保存ファイルの中身をそのまま指す）。行 = 出力 1 つ分。
struct GGUFMatrix {
    enum Storage: String, Codable, Sendable {
        /// Float32 のまま
        case float32
        /// 1 行ごとに倍率を持つ 8 ビット整数（値 = 整数 × その行の倍率）
        case int8Rows
    }

    var rows: Int
    var columns: Int
    var storage: Storage
    var data: UnsafeRawPointer
    /// int8Rows のときの、行ごとの倍率
    var scales: UnsafePointer<Float>?

    /// 一部の行だけの行列（q・k・v がまとまった行列を分けるときなど）
    func slice(_ range: Range<Int>) -> GGUFMatrix {
        let elementSize = storage == .float32 ? MemoryLayout<Float>.size : 1
        return GGUFMatrix(rows: range.count, columns: columns, storage: storage,
                          data: data + range.lowerBound * columns * elementSize,
                          scales: scales.map { $0 + range.lowerBound })
    }
}

/// 並列に計算するときの作業用メモリ。
///
/// 行列の行を「まとまり」に分け、空いているコアが次のまとまりを取っていく（DispatchQueue.concurrentPerform）。
/// iPad は速いコアと遅いコアが混ざっているので、コアの数で均等に割るより、細かく分けたほうが速いコアを生かせる。
final class GGUFWorkers {
    /// 1 つの行列を分けるまとまりの数の上限
    let chunkCount: Int
    /// 整数の重みを Float に直して置く場所（まとまりごと）
    private var scratch: [UnsafeMutablePointer<Float>]
    private let scratchCapacity: Int

    /// 1 回に Float に直す量（CPU のキャッシュに収まる大きさ）
    static let blockElements = 96 * 1024

    /// - Parameter chunkCount: まとまりの数（nil なら、コアの数の 3 倍。1 なら並列にしない）
    init(chunkCount: Int? = nil, widestRow: Int) {
        let cores = ProcessInfo.processInfo.activeProcessorCount
        let chunks = max(1, chunkCount ?? min(24, cores * 3))
        let capacity = max(Self.blockElements, widestRow * 4)
        self.chunkCount = chunks
        scratchCapacity = capacity
        scratch = (0..<chunks).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: capacity) }
    }

    deinit {
        for pointer in scratch { pointer.deallocate() }
    }

    func blockRows(columns: Int) -> Int {
        max(1, min(512, scratchCapacity / max(columns, 1)))
    }

    func scratch(_ chunk: Int) -> UnsafeMutablePointer<Float> { scratch[chunk] }
}

/// GGUF の推論で使う計算（iPad では Accelerate、Linux のテストでは素朴なループ）。
enum GGUFKernels {
    typealias In = UnsafePointer<Float>
    typealias Out = UnsafeMutablePointer<Float>

    // MARK: - 整数と Float の変換

    static func int8ToFloat(_ source: UnsafePointer<Int8>, _ output: Out, count: Int) {
        #if canImport(Accelerate)
        vDSP_vflt8(source, 1, output, 1, vDSP_Length(count))
        #else
        for index in 0..<count { output[index] = Float(source[index]) }
        #endif
    }

    /// 1 行を 8 ビットにする。戻り値はその行の倍率（値 ≒ 整数 × 倍率）。scratch は count 個以上。
    static func quantizeRow(_ row: In, count: Int, into output: UnsafeMutablePointer<Int8>, scratch: Out) -> Float {
        var peak: Float = 0
        #if canImport(Accelerate)
        vDSP_maxmgv(row, 1, &peak, vDSP_Length(count))
        #else
        for index in 0..<count { peak = max(peak, abs(row[index])) }
        #endif
        guard peak > 0, peak.isFinite else {
            output.update(repeating: 0, count: count)
            return 0
        }
        let scale = peak / 127
        #if canImport(Accelerate)
        var inverse = 127 / peak
        vDSP_vsmul(row, 1, &inverse, scratch, 1, vDSP_Length(count))
        vDSP_vfixr8(scratch, 1, output, 1, vDSP_Length(count))
        #else
        let inverse = 127 / peak
        for index in 0..<count {
            let value = (row[index] * inverse).rounded()
            output[index] = Int8(max(-127, min(127, value)))
        }
        #endif
        return scale
    }

    // MARK: - 線形層

    /// Y[n, r] = Σ_c X[n, c] · W[r, c]（X は count 行 × columns、Y は count 行 × rows。rowStride は Y の 1 行の長さ）
    static func linear(_ matrix: GGUFMatrix, input: In, count: Int, output: Out, rowStride: Int? = nil, workers: GGUFWorkers) {
        let stride = rowStride ?? matrix.rows
        let work = matrix.rows * matrix.columns * count
        // 小さい行列は、分けるほうが高くつく
        let chunks = work < 400_000 ? 1 : min(workers.chunkCount, max(1, matrix.rows / 16))
        if chunks <= 1 {
            linearRows(matrix, 0..<matrix.rows, input: input, count: count, output: output, stride: stride,
                       scratch: workers.scratch(0), workers: workers)
            return
        }
        let perChunk = (matrix.rows + chunks - 1) / chunks
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let start = chunk * perChunk
            let end = min(matrix.rows, start + perChunk)
            guard start < end else { return }
            linearRows(matrix, start..<end, input: input, count: count, output: output, stride: stride,
                       scratch: workers.scratch(chunk), workers: workers)
        }
    }

    private static func linearRows(_ matrix: GGUFMatrix, _ range: Range<Int>, input: In, count: Int, output: Out, stride: Int,
                                   scratch: Out, workers: GGUFWorkers) {
        let columns = matrix.columns
        switch matrix.storage {
        case .float32:
            let weights = matrix.data.assumingMemoryBound(to: Float.self) + range.lowerBound * columns
            multiply(input: input, count: count, weights: weights, rows: range.count, columns: columns,
                     output: output + range.lowerBound, stride: stride)
        case .int8Rows:
            let weights = matrix.data.assumingMemoryBound(to: Int8.self)
            let blockRows = workers.blockRows(columns: columns)
            var row = range.lowerBound
            while row < range.upperBound {
                let rows = min(blockRows, range.upperBound - row)
                int8ToFloat(weights + row * columns, scratch, count: rows * columns)
                multiply(input: input, count: count, weights: scratch, rows: rows, columns: columns,
                         output: output + row, stride: stride)
                if let scales = matrix.scales {
                    for n in 0..<count {
                        let target = output + n * stride + row
                        MiniVector.multiply(target, scales + row, target, rows)
                    }
                }
                row += rows
            }
        }
    }

    /// output[n, r] = X[n, :] · weights[r, :]
    private static func multiply(input: In, count: Int, weights: In, rows: Int, columns: Int, output: Out, stride: Int) {
        #if canImport(Accelerate)
        if count == 1 {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rows), Int32(columns), 1, weights, Int32(columns),
                        input, 1, 0, output, 1)
            return
        }
        #endif
        MiniMath.gemm(transA: false, transB: true, m: count, n: rows, k: columns, alpha: 1,
                      a: input, lda: columns, b: weights, ldb: columns, beta: 0, c: output, ldc: stride)
    }

    /// 埋め込みの 1 行を取り出す（scale 倍して output に書く）
    static func embeddingRow(_ matrix: GGUFMatrix, row: Int, scale: Float, into output: Out) {
        let columns = matrix.columns
        switch matrix.storage {
        case .float32:
            MiniVector.scale(matrix.data.assumingMemoryBound(to: Float.self) + row * columns, scale, output, columns)
        case .int8Rows:
            int8ToFloat(matrix.data.assumingMemoryBound(to: Int8.self) + row * columns, output, count: columns)
            MiniVector.scale(output, scale * (matrix.scales?[row] ?? 1), output, columns)
        }
    }

    // MARK: - 正規化と活性化

    /// RMSNorm: out = x / sqrt(mean(x²) + eps) × weight（rows 行 × columns。x と out は同じ場所でもよい）
    static func rmsNorm(_ x: In, weight: In, rows: Int, columns: Int, epsilon: Float, into output: Out) {
        for row in 0..<rows {
            let source = x + row * columns
            let target = output + row * columns
            let mean = MiniMath.sumOfSquares(source, count: columns) / Float(columns)
            MiniVector.scale(source, 1 / (mean + epsilon).squareRoot(), target, columns)
            MiniVector.multiply(target, weight, target, columns)
        }
    }

    /// gate を活性化して up と掛ける: gate = act(gate) × up（scratch は count 個以上）
    static func gatedActivation(_ activation: GGUFModelConfig.Activation, gate: Out, up: In, count: Int, scratch: Out) {
        switch activation {
        case .silu:
            // x / (1 + exp(-x))
            MiniVector.scale(gate, -1, scratch, count)
            MiniVector.exp(scratch, scratch, count)
            MiniVector.shift(scratch, 1, scratch, count)
            MiniVector.divide(gate, scratch, gate, count)
        case .gelu:
            // 0.5 x (1 + tanh(√(2/π) x (1 + 0.044715 x²)))
            MiniVector.multiply(gate, gate, scratch, count)
            MiniVector.scaleShift(scratch, 0.044715, 1, scratch, count)
            MiniVector.multiply(scratch, gate, scratch, count)
            MiniVector.scale(scratch, 0.7978845608028654, scratch, count)
            MiniVector.tanh(scratch, scratch, count)
            MiniVector.scaleShift(scratch, 0.5, 0.5, scratch, count)
            MiniVector.multiply(scratch, gate, gate, count)
        }
        MiniVector.multiply(gate, up, gate, count)
    }

    /// x = cap × tanh(x / cap)
    static func softcap(_ x: Out, count: Int, cap: Float) {
        MiniVector.scale(x, 1 / cap, x, count)
        MiniVector.tanh(x, x, count)
        MiniVector.scale(x, cap, x, count)
    }

    /// その場で softmax（count 個）
    static func softmax(_ x: Out, count: Int) {
        guard count > 0 else { return }
        let peak = MiniVector.maximum(x, count)
        MiniVector.shift(x, -peak, x, count)
        MiniVector.exp(x, x, count)
        let total = MiniVector.sum(x, count)
        MiniVector.scale(x, 1 / total, x, count)
    }

    // MARK: - RoPE

    static func cosSin(_ angles: In, count: Int, cos: Out, sin: Out) {
        #if canImport(Accelerate)
        var n = Int32(count)
        vvcosf(cos, angles, &n)
        vvsinf(sin, angles, &n)
        #else
        for index in 0..<count {
            cos[index] = Foundation.cos(angles[index])
            sin[index] = Foundation.sin(angles[index])
        }
        #endif
    }

    /// 1 つのヘッド（先頭の 2 × pairs 個）を回転させる。
    /// split: 組は (i, i + pairs)。そうでなければ (2i, 2i + 1)。scratch は 2 × pairs 個以上。
    static func rotate(_ x: Out, pairs: Int, split: Bool, cos: In, sin: In, scratch: Out) {
        #if canImport(Accelerate)
        let n = vDSP_Length(pairs)
        let step = split ? 1 : 2
        let second = x + (split ? pairs : 1)
        // a = x0·cos − x1·sin、b = x0·sin + x1·cos
        vDSP_vmul(x, step, cos, 1, scratch, 1, n)
        vDSP_vmul(second, step, sin, 1, scratch + pairs, 1, n)
        vDSP_vsub(scratch + pairs, 1, scratch, 1, scratch, 1, n)
        vDSP_vmul(x, step, sin, 1, scratch + pairs, 1, n)
        vDSP_vma(second, step, cos, 1, scratch + pairs, 1, second, step, n)
        if split {
            x.update(from: scratch, count: pairs)
        } else {
            var zero: Float = 0
            vDSP_vsadd(scratch, 1, &zero, x, 2, n)
        }
        #else
        for index in 0..<pairs {
            let first = split ? index : index * 2
            let other = split ? index + pairs : index * 2 + 1
            let a = x[first], b = x[other]
            x[first] = a * cos[index] - b * sin[index]
            x[other] = a * sin[index] + b * cos[index]
        }
        #endif
    }
}
