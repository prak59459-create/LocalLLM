import Foundation

/// 必要になったら大きくする Float の作業用メモリ。
final class GGUFBuffer {
    private(set) var pointer: UnsafeMutablePointer<Float>
    private(set) var capacity: Int

    init(_ capacity: Int = 0) {
        self.capacity = max(capacity, 16)
        pointer = .allocate(capacity: self.capacity)
    }

    deinit { pointer.deallocate() }

    /// count 個入るようにする（中身は引き継がない）
    @discardableResult
    func reserve(_ count: Int) -> UnsafeMutablePointer<Float> {
        if count > capacity {
            pointer.deallocate()
            capacity = count
            pointer = .allocate(capacity: count)
        }
        return pointer
    }
}

/// GGUF から変換した重みで、トランスフォーマーの計算（順伝播）を行う。KV キャッシュを持つ。
///
/// llama.cpp の llama / qwen2 / qwen3 / gemma / gemma2 / gemma3 / phi3 と同じ計算の流れ。
/// スレッドセーフではないので、1 つの場所から順番に使う。
final class GGUFModel {
    private struct Layer {
        var attentionNorm: UnsafePointer<Float>
        var query: GGUFMatrix
        var key: GGUFMatrix
        var value: GGUFMatrix
        var queryBias: UnsafePointer<Float>?
        var keyBias: UnsafePointer<Float>?
        var valueBias: UnsafePointer<Float>?
        var output: GGUFMatrix
        var outputBias: UnsafePointer<Float>?
        var queryNorm: UnsafePointer<Float>?
        var keyNorm: UnsafePointer<Float>?
        var postAttentionNorm: UnsafePointer<Float>?
        var feedForwardNorm: UnsafePointer<Float>
        var gate: GGUFMatrix
        var up: GGUFMatrix
        var down: GGUFMatrix
        var gateBias: UnsafePointer<Float>?
        var upBias: UnsafePointer<Float>?
        var downBias: UnsafePointer<Float>?
        var postFeedForwardNorm: UnsafePointer<Float>?
    }

    let config: GGUFModelConfig
    /// 扱える最大のトークン数（KV キャッシュの大きさ）
    let maxContext: Int
    /// KV キャッシュに入っているトークン数
    private(set) var length = 0

    private let store: GGUFWeightStore
    private let workers: GGUFWorkers
    private let layers: [Layer]
    private let embedding: GGUFMatrix
    private let outputMatrix: GGUFMatrix
    private let outputNorm: UnsafePointer<Float>
    private let outputBias: UnsafePointer<Float>?

    /// 層ごとの KV キャッシュ: [kvHead][位置][headSize]（使った位置のページだけが実メモリになる）
    private var keys: [UnsafeMutablePointer<Float>] = []
    private var values: [UnsafeMutablePointer<Float>] = []

    /// RoPE の周波数（位置に掛ける値）。全体を見る層用と、直近だけを見る層用
    private let frequencies: [Float]
    private let slidingFrequencies: [Float]

    // 作業用
    private let hidden = GGUFBuffer(), normed = GGUFBuffer()
    private let queries = GGUFBuffer(), newKeys = GGUFBuffer(), newValues = GGUFBuffer()
    private let attended = GGUFBuffer(), projected = GGUFBuffer()
    private let gateBuffer = GGUFBuffer(), upBuffer = GGUFBuffer(), scratch = GGUFBuffer()
    private let scores = GGUFBuffer()
    private let angles = GGUFBuffer(), cosines = GGUFBuffer(), sines = GGUFBuffer(), slidingCosines = GGUFBuffer(), slidingSines = GGUFBuffer()

    /// 1 回にまとめて計算するトークン数の上限
    static let maxBatch = 64

    init(store: GGUFWeightStore, maxContext: Int? = nil, threadCount: Int? = nil) throws {
        self.store = store
        let config = store.config
        self.config = config
        self.maxContext = max(8, min(config.contextLength, maxContext ?? config.contextLength))
        workers = GGUFWorkers(threadCount: threadCount, widestRow: max(config.feedForwardSize, config.embeddingSize))
        embedding = try store.matrix("token_embd.weight")
        outputMatrix = try store.matrix(config.tiedOutput ? "token_embd.weight" : "output.weight")
        outputNorm = try store.requiredVector("output_norm.weight")
        outputBias = store.vector("output.bias")
        layers = try (0..<config.layerCount).map { try Self.makeLayer($0, store: store, config: config) }

        let factors = store.vector("rope_freqs.weight")
        frequencies = Self.ropeFrequencies(config: config, base: config.ropeBase, factors: factors)
        slidingFrequencies = Self.ropeFrequencies(config: config, base: config.slidingRopeBase, factors: factors)

        let cacheCount = config.kvSize * self.maxContext
        for _ in 0..<config.layerCount {
            keys.append(.allocate(capacity: cacheCount))
            values.append(.allocate(capacity: cacheCount))
        }
    }

    deinit {
        for pointer in keys { pointer.deallocate() }
        for pointer in values { pointer.deallocate() }
    }

    private static func makeLayer(_ index: Int, store: GGUFWeightStore, config: GGUFModelConfig) throws -> Layer {
        let prefix = "blk.\(index)."
        let q = config.querySize, kv = config.kvSize, ff = config.feedForwardSize
        var query: GGUFMatrix, key: GGUFMatrix, value: GGUFMatrix
        var queryBias: UnsafePointer<Float>?, keyBias: UnsafePointer<Float>?, valueBias: UnsafePointer<Float>?
        if config.hasFusedQKV {
            let fused = try store.matrix(prefix + "attn_qkv.weight")
            query = fused.slice(0..<q)
            key = fused.slice(q..<(q + kv))
            value = fused.slice((q + kv)..<(q + 2 * kv))
            if let bias = store.vector(prefix + "attn_qkv.bias") {
                queryBias = bias
                keyBias = bias + q
                valueBias = bias + q + kv
            }
        } else {
            query = try store.matrix(prefix + "attn_q.weight")
            key = try store.matrix(prefix + "attn_k.weight")
            value = try store.matrix(prefix + "attn_v.weight")
        }
        queryBias = queryBias ?? store.vector(prefix + "attn_q.bias")
        keyBias = keyBias ?? store.vector(prefix + "attn_k.bias")
        valueBias = valueBias ?? store.vector(prefix + "attn_v.bias")

        var gate: GGUFMatrix, up: GGUFMatrix
        if config.hasFusedGateUp {
            let fused = try store.matrix(prefix + "ffn_up.weight")
            gate = fused.slice(0..<ff)
            up = fused.slice(ff..<(2 * ff))
        } else {
            gate = try store.matrix(prefix + "ffn_gate.weight")
            up = try store.matrix(prefix + "ffn_up.weight")
        }
        return Layer(
            attentionNorm: try store.requiredVector(prefix + "attn_norm.weight"),
            query: query, key: key, value: value,
            queryBias: queryBias, keyBias: keyBias, valueBias: valueBias,
            output: try store.matrix(prefix + "attn_output.weight"),
            outputBias: store.vector(prefix + "attn_output.bias"),
            queryNorm: config.hasQKNorm ? try store.requiredVector(prefix + "attn_q_norm.weight") : nil,
            keyNorm: config.hasQKNorm ? try store.requiredVector(prefix + "attn_k_norm.weight") : nil,
            postAttentionNorm: config.hasPostNorms ? try store.requiredVector(prefix + "post_attention_norm.weight") : nil,
            feedForwardNorm: try store.requiredVector(prefix + "ffn_norm.weight"),
            gate: gate, up: up,
            down: try store.matrix(prefix + "ffn_down.weight"),
            gateBias: config.hasFusedGateUp ? nil : store.vector(prefix + "ffn_gate.bias"),
            upBias: config.hasFusedGateUp ? nil : store.vector(prefix + "ffn_up.bias"),
            downBias: store.vector(prefix + "ffn_down.bias"),
            postFeedForwardNorm: config.hasPostNorms ? try store.requiredVector(prefix + "post_ffw_norm.weight") : nil
        )
    }

    /// 位置 p の i 番目の組の角度 = p × frequencies[i]
    private static func ropeFrequencies(config: GGUFModelConfig, base: Float, factors: UnsafePointer<Float>?) -> [Float] {
        let pairs = config.ropeDimensions / 2
        return (0..<pairs).map { index in
            let theta = Foundation.pow(Double(base), -2 * Double(index) / Double(config.ropeDimensions))
            let factor = factors.map { Double($0[index]) } ?? 1
            return Float(theta / factor) * config.ropeScale
        }
    }

    // MARK: - KV キャッシュ

    /// KV キャッシュを length トークンまで巻き戻す（それより後ろは次の計算で上書きされる）
    func truncate(to newLength: Int) {
        length = max(0, min(length, newLength))
    }

    /// KV キャッシュが今使っているメモリの目安（バイト）
    var cacheBytes: Int { length * config.kvBytesPerToken }

    // MARK: - 計算

    /// tokens を今の系列の後ろに足して計算し、最後のトークンの次の logits を返す。
    /// - Parameters:
    ///   - batch: 1 回にまとめて計算するトークン数
    ///   - progress: 何トークン読めたか（ここでエラーを投げると、そこで止まる。読めた分は KV キャッシュに残る）
    @discardableResult
    func evaluate(_ tokens: [Int], batch: Int = GGUFModel.maxBatch, progress: ((Int) throws -> Void)? = nil) throws -> [Float] {
        guard !tokens.isEmpty else { throw GGUFError.invalid("no tokens") }
        guard length + tokens.count <= maxContext else {
            throw GGUFError.invalid(L("コンテキストの上限（\(maxContext) トークン）を超えました", "context limit of \(maxContext) tokens exceeded"))
        }
        if let bad = tokens.first(where: { $0 < 0 || $0 >= config.vocabSize }) { throw GGUFError.invalid("token id \(bad)") }
        var offset = 0
        var logits: [Float] = []
        while offset < tokens.count {
            let count = min(max(1, batch), tokens.count - offset)
            let isLast = offset + count == tokens.count
            let batch = Array(tokens[offset..<(offset + count)])
            let result = forward(batch, wantsLogits: isLast)
            if isLast { logits = result }
            offset += count
            try progress?(offset)
        }
        return logits
    }

    private func forward(_ tokens: [Int], wantsLogits: Bool) -> [Float] {
        let n = tokens.count
        let d = config.embeddingSize, ff = config.feedForwardSize
        let q = config.querySize, kv = config.kvSize
        let start = length

        let x = hidden.reserve(n * d)
        let h = normed.reserve(n * d)
        let work = scratch.reserve(n * max(ff, d, q) + 4 * config.headSize)
        for (index, token) in tokens.enumerated() {
            GGUFKernels.embeddingRow(embedding, row: token, scale: config.embeddingScale, into: x + index * d)
        }
        prepareRotations(start: start, count: n)

        for (index, layer) in layers.enumerated() {
            // 注意（self-attention）
            GGUFKernels.rmsNorm(x, weight: layer.attentionNorm, rows: n, columns: d, epsilon: config.normEpsilon, into: h)
            let qs = queries.reserve(n * q), ks = newKeys.reserve(n * kv), vs = newValues.reserve(n * kv)
            linear(layer.query, bias: layer.queryBias, input: h, count: n, output: qs)
            linear(layer.key, bias: layer.keyBias, input: h, count: n, output: ks)
            linear(layer.value, bias: layer.valueBias, input: h, count: n, output: vs)
            if let queryNorm = layer.queryNorm, let keyNorm = layer.keyNorm {
                GGUFKernels.rmsNorm(qs, weight: queryNorm, rows: n * config.headCount, columns: config.headSize,
                                    epsilon: config.normEpsilon, into: qs)
                GGUFKernels.rmsNorm(ks, weight: keyNorm, rows: n * config.kvHeadCount, columns: config.headSize,
                                    epsilon: config.normEpsilon, into: ks)
            }
            rotate(qs, heads: config.headCount, count: n, sliding: config.isSliding(index), scratch: work)
            rotate(ks, heads: config.kvHeadCount, count: n, sliding: config.isSliding(index), scratch: work)
            MiniVector.scale(qs, config.attentionScale, qs, n * q)
            store(ks, vs, layer: index, start: start, count: n)

            let attention = attended.reserve(n * q)
            attend(layer: index, queries: qs, start: start, count: n, into: attention)
            let out = projected.reserve(n * max(d, ff))
            linear(layer.output, bias: layer.outputBias, input: attention, count: n, output: out)
            if let norm = layer.postAttentionNorm {
                GGUFKernels.rmsNorm(out, weight: norm, rows: n, columns: d, epsilon: config.normEpsilon, into: out)
            }
            MiniVector.add(x, out, x, n * d)

            // FFN
            GGUFKernels.rmsNorm(x, weight: layer.feedForwardNorm, rows: n, columns: d, epsilon: config.normEpsilon, into: h)
            let gate = gateBuffer.reserve(n * ff), up = upBuffer.reserve(n * ff)
            linear(layer.gate, bias: layer.gateBias, input: h, count: n, output: gate)
            linear(layer.up, bias: layer.upBias, input: h, count: n, output: up)
            GGUFKernels.gatedActivation(config.activation, gate: gate, up: up, count: n * ff, scratch: work)
            linear(layer.down, bias: layer.downBias, input: gate, count: n, output: out)
            if let norm = layer.postFeedForwardNorm {
                GGUFKernels.rmsNorm(out, weight: norm, rows: n, columns: d, epsilon: config.normEpsilon, into: out)
            }
            MiniVector.add(x, out, x, n * d)
        }
        length = start + n

        guard wantsLogits else { return [] }
        // 最後のトークンだけ、出力層を計算する
        GGUFKernels.rmsNorm(x + (n - 1) * d, weight: outputNorm, rows: 1, columns: d, epsilon: config.normEpsilon, into: h)
        var result = [Float](repeating: 0, count: config.vocabSize)
        result.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            GGUFKernels.linear(outputMatrix, input: h, count: 1, output: base, workers: workers)
            if let outputBias { MiniMath.add(outputBias, into: base, count: config.vocabSize) }
            if config.finalSoftcap > 0 { GGUFKernels.softcap(base, count: config.vocabSize, cap: config.finalSoftcap) }
        }
        return result
    }

    private func linear(_ matrix: GGUFMatrix, bias: UnsafePointer<Float>?, input: UnsafePointer<Float>, count: Int,
                        output: UnsafeMutablePointer<Float>) {
        GGUFKernels.linear(matrix, input: input, count: count, output: output, workers: workers)
        if let bias { MiniMath.addBias(bias, to: output, rows: count, columns: matrix.rows) }
    }

    // MARK: - RoPE

    /// 位置 start ..< start + count の cos / sin を用意する
    private func prepareRotations(start: Int, count: Int) {
        let pairs = frequencies.count
        guard pairs > 0 else { return }
        let angle = angles.reserve(pairs)
        func fill(_ source: [Float], _ cosBuffer: GGUFBuffer, _ sinBuffer: GGUFBuffer) {
            let cos = cosBuffer.reserve(count * pairs), sin = sinBuffer.reserve(count * pairs)
            source.withUnsafeBufferPointer { frequency in
                guard let base = frequency.baseAddress else { return }
                for index in 0..<count {
                    MiniVector.scale(base, Float(start + index), angle, pairs)
                    GGUFKernels.cosSin(angle, count: pairs, cos: cos + index * pairs, sin: sin + index * pairs)
                }
            }
        }
        fill(frequencies, cosines, sines)
        if config.slidingWindow > 0 { fill(slidingFrequencies, slidingCosines, slidingSines) }
    }

    /// [count][heads][headSize] の各ヘッドを、その位置の角度で回転させる
    private func rotate(_ vectors: UnsafeMutablePointer<Float>, heads: Int, count: Int, sliding: Bool,
                        scratch: UnsafeMutablePointer<Float>) {
        let pairs = frequencies.count
        guard pairs > 0 else { return }
        let cos = (sliding ? slidingCosines : cosines).pointer
        let sin = (sliding ? slidingSines : sines).pointer
        let size = config.headSize
        for index in 0..<count {
            for head in 0..<heads {
                GGUFKernels.rotate(vectors + (index * heads + head) * size, pairs: pairs, split: config.ropePairsAreSplit,
                                   cos: cos + index * pairs, sin: sin + index * pairs, scratch: scratch)
            }
        }
    }

    // MARK: - 注意

    private func store(_ newKeys: UnsafePointer<Float>, _ newValues: UnsafePointer<Float>, layer: Int, start: Int, count: Int) {
        let size = config.headSize, groups = config.kvHeadCount
        for group in 0..<groups {
            let keyBase = keys[layer] + (group * maxContext + start) * size
            let valueBase = values[layer] + (group * maxContext + start) * size
            for index in 0..<count {
                (keyBase + index * size).update(from: newKeys + (index * groups + group) * size, count: size)
                (valueBase + index * size).update(from: newValues + (index * groups + group) * size, count: size)
            }
        }
    }

    private func attend(layer: Int, queries: UnsafePointer<Float>, start: Int, count: Int, into output: UnsafeMutablePointer<Float>) {
        let size = config.headSize, q = config.querySize
        let total = start + count
        let groupSize = config.headCount / config.kvHeadCount
        let window = config.isSliding(layer) ? config.slidingWindow : 0
        let score = scores.reserve(count * total)
        for head in 0..<config.headCount {
            let group = head / groupSize
            let keyBase = keys[layer] + group * maxContext * size
            let valueBase = values[layer] + group * maxContext * size
            // スコア [count × total] = Q · Kᵀ
            MiniMath.gemm(transA: false, transB: true, m: count, n: total, k: size, alpha: 1,
                          a: queries + head * size, lda: q, b: keyBase, ldb: size, beta: 0, c: score, ldc: total)
            for index in 0..<count {
                let position = start + index
                let first = window > 0 ? max(0, position - window + 1) : 0
                let row = score + index * total
                let visible = position - first + 1
                if config.attentionSoftcap > 0 { GGUFKernels.softcap(row + first, count: visible, cap: config.attentionSoftcap) }
                GGUFKernels.softmax(row + first, count: visible)
                // 見えない位置（窓より前・自分より後ろ）は 0
                if first > 0 { MiniVector.fillZero(row, first) }
                if position + 1 < total { MiniVector.fillZero(row + position + 1, total - position - 1) }
            }
            // 出力 [count × size] = 重み · V
            MiniMath.gemm(transA: false, transB: false, m: count, n: size, k: total, alpha: 1,
                          a: score, lda: total, b: valueBase, ldb: size, beta: 0, c: output + head * size, ldc: q)
        }
    }
}
