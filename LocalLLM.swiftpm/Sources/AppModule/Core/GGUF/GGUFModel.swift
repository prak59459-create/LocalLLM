import Foundation

/// GGUF から変換した重みで、トランスフォーマーの計算（順伝播）を CPU で行う。KV キャッシュを持つ。
///
/// 計算の流れは llama.cpp と同じ。系列ごとの違いは GGUFModelConfig の値と印（正規化の種類・並列の残差・
/// 層ごとの形・KV の共有・層ごとの入力など）で表す。スレッドセーフではないので、1 つの場所から順番に使う。
final class GGUFModel: GGUFEvaluator {
    typealias In = UnsafePointer<Float>
    typealias Out = UnsafeMutablePointer<Float>

    let config: GGUFModelConfig
    /// 扱える最大のトークン数（KV キャッシュの大きさ）
    let maxContext: Int
    /// KV キャッシュに入っているトークン数
    private(set) var length = 0

    private let store: GGUFWeightStore
    private let workers: GGUFWorkers
    private let shapes: [GGUFModelConfig.Layer]
    private let layers: [GGUFLayerWeights]
    private let globals: GGUFGlobalWeights

    /// 層ごとの KV キャッシュ: [kvHead][位置][headSize]（使った位置のページだけが実メモリになる）。借りる層は nil
    private var keys: [Out?] = []
    private var values: [Out?] = []

    /// RoPE の表。全体を見る層用と、直近だけを見る層用
    private let fullRope: GGUFRope.Table
    private let slidingRope: GGUFRope.Table

    // 作業用
    private let hidden = GGUFBuffer(), normed = GGUFBuffer(), normed2 = GGUFBuffer()
    private let queries = GGUFBuffer(), newKeys = GGUFBuffer(), newValues = GGUFBuffer()
    private let attended = GGUFBuffer(), projected = GGUFBuffer(), feedForwardOut = GGUFBuffer()
    private let gateBuffer = GGUFBuffer(), upBuffer = GGUFBuffer(), scratch = GGUFBuffer()
    private let scores = GGUFBuffer(), perLayerInputs = GGUFBuffer(), perLayerTokens = GGUFBuffer(), perLayerWork = GGUFBuffer()
    private let angles = GGUFBuffer()
    private let fullCos = GGUFBuffer(), fullSin = GGUFBuffer(), slidingCos = GGUFBuffer(), slidingSin = GGUFBuffer()

    /// 1 回にまとめて計算するトークン数の上限
    static let maxBatch = 64

    /// - Parameter chunkCount: 並列に計算するときのまとまりの数（nil ならコアの数から決める）
    init(store: GGUFWeightStore, maxContext: Int? = nil, chunkCount: Int? = nil) throws {
        self.store = store
        let config = store.config
        self.config = config
        self.maxContext = max(8, min(config.contextLength, maxContext ?? config.contextLength))
        shapes = config.layers
        let widest = max(config.maxFeedForwardSize, config.embeddingSize, config.maxQuerySize, config.perLayerInputSize)
        // 畳み込みの層の入力の射影は 3 × 埋め込みの大きさ
        workers = GGUFWorkers(chunkCount: chunkCount, widestRow: widest)
        let globals = try GGUFGlobalWeights(store: store, config: config)
        self.globals = globals
        layers = try (0..<config.layerCount).map { try GGUFLayerWeights.load($0, store: store, config: config) }
        fullRope = GGUFRope.table(config: config, sliding: false, factors: globals.ropeFactors)
        slidingRope = GGUFRope.table(config: config, sliding: true, factors: globals.ropeFactors)
        allocateCache()
    }

    deinit { freeCache() }

    var computeDescription: String { "CPU (Accelerate)" }

    // MARK: - KV キャッシュ

    private func allocateCache() {
        keys = []
        values = []
        for (index, shape) in shapes.enumerated() {
            if shape.convolution {
                // 畳み込みの層は、位置ごとの b × x を覚えておく（巻き戻しても、その位置から続けられる）
                keys.append(Out.allocate(capacity: config.embeddingSize * maxContext))
                values.append(nil)
                continue
            }
            let owns = shape.kvLayer == index
            keys.append(owns ? Out.allocate(capacity: shape.kvSize * maxContext) : nil)
            values.append(owns ? Out.allocate(capacity: shape.kvSize * maxContext) : nil)
        }
    }

    private func freeCache() {
        for pointer in keys { pointer?.deallocate() }
        for pointer in values { pointer?.deallocate() }
        keys = []
        values = []
    }

    /// KV キャッシュを length トークンまで巻き戻す（それより後ろは次の計算で上書きされる）
    func truncate(to newLength: Int) {
        length = max(0, min(length, newLength))
    }

    /// KV キャッシュを空にして、使っていたメモリを OS に返す（確保し直す。使うまで実メモリにはならない）
    func releaseCache() {
        length = 0
        freeCache()
        allocateCache()
    }

    /// KV キャッシュが今使っているメモリの目安（バイト）
    var cacheBytes: Int { length * config.kvBytesPerToken }

    // MARK: - 計算

    /// tokens を今の系列の後ろに足して計算し、最後のトークンの次の logits を返す。
    /// - Parameters:
    ///   - batch: 1 回にまとめて計算するトークン数
    ///   - progress: 何トークン読めたか（ここでエラーを投げると、そこで止まる。読めた分は KV キャッシュに残る）
    @discardableResult
    func evaluate(_ tokens: [Int], batch: Int, progress: ((Int) throws -> Void)?) throws -> [Float] {
        try validate(tokens)
        var offset = 0
        var logits: [Float] = []
        while offset < tokens.count {
            let count = min(max(1, batch), tokens.count - offset)
            let isLast = offset + count == tokens.count
            let result = forward(Array(tokens[offset..<(offset + count)]), wantsLogits: isLast)
            if isLast { logits = result }
            offset += count
            try progress?(offset)
        }
        return logits
    }

    private func forward(_ tokens: [Int], wantsLogits: Bool) -> [Float] {
        let n = tokens.count, d = config.embeddingSize
        let start = length
        let x = hidden.reserve(n * d)
        embed(tokens, start: start, into: x)
        let inputs = preparePerLayerInputs(tokens, embedded: x)
        prepareRotations(start: start, count: n)
        for index in layers.indices {
            runLayer(index, x: x, start: start, count: n, perLayerInputs: inputs)
        }
        length = start + n
        guard wantsLogits else { return [] }
        return logits(from: x + (n - 1) * d)
    }

    private func embed(_ tokens: [Int], start: Int, into x: Out) {
        let d = config.embeddingSize
        for (index, token) in tokens.enumerated() {
            GGUFKernels.embeddingRow(globals.embedding, row: token, scale: config.embeddingScale, into: x + index * d)
        }
        guard let positions = globals.positions else { return }
        let row = scratch.reserve(d)
        for index in tokens.indices {
            GGUFKernels.embeddingRow(positions, row: start + index, scale: 1, into: row)
            MiniVector.add(x + index * d, row, x + index * d, d)
        }
    }

    /// 層ごとの入力（Gemma 4）: [トークン][層][perLayerInputSize]
    private func preparePerLayerInputs(_ tokens: [Int], embedded x: Out) -> In? {
        let size = config.perLayerInputSize
        guard size > 0, let table = globals.perLayerEmbedding, let projection = globals.perLayerProjection else { return nil }
        let n = tokens.count, d = config.embeddingSize, total = size * config.layerCount
        let inputs = perLayerInputs.reserve(n * total)
        // 埋め込みからの射影 × 1/√d を、層ごとに正規化する
        GGUFKernels.linear(projection, input: x, count: n, output: inputs, workers: workers)
        MiniVector.scale(inputs, 1 / Float(d).squareRoot(), inputs, n * total)
        GGUFKernels.rmsNorm(inputs, weight: globals.perLayerProjectionNorm, rows: n * config.layerCount, columns: size,
                            epsilon: config.normEpsilon, into: inputs)
        // トークンごとの表 × √size を足して、1/√2 を掛ける
        let rows = perLayerTokens.reserve(n * total)
        for (index, token) in tokens.enumerated() {
            GGUFKernels.embeddingRow(table, row: token, scale: Float(size).squareRoot(), into: rows + index * total)
        }
        MiniVector.add(inputs, rows, inputs, n * total)
        MiniVector.scale(inputs, 1 / Float(2).squareRoot(), inputs, n * total)
        return In(inputs)
    }

    private func runLayer(_ index: Int, x: Out, start: Int, count n: Int, perLayerInputs: In?) {
        let layer = layers[index]
        let d = config.embeddingSize
        // 注意
        var attentionInput = In(x)
        if config.hasPreNorms {
            let h = normed.reserve(n * d)
            normalize(x, weight: layer.attentionNorm, bias: layer.attentionNormBias, rows: n, columns: d, into: h)
            attentionInput = In(h)
        }
        let out = projected.reserve(n * d)
        attention(index, input: attentionInput, start: start, count: n, into: out)
        finishBranch(out, postNorm: layer.postAttentionNorm, count: n)

        // FFN
        let ffn = feedForwardOut.reserve(n * d)
        if config.parallelResidual {
            var input = attentionInput
            if let norm = layer.feedForwardNorm {
                let h = normed2.reserve(n * d)
                normalize(x, weight: norm, bias: layer.feedForwardNormBias, rows: n, columns: d, into: h)
                input = In(h)
            }
            feedForward(index, input: input, count: n, into: ffn)
            finishBranch(ffn, postNorm: layer.postFeedForwardNorm, count: n)
            MiniVector.add(x, out, x, n * d)
            MiniVector.add(x, ffn, x, n * d)
        } else {
            MiniVector.add(x, out, x, n * d)
            var input = In(x)
            if config.hasPreNorms {
                let h = normed.reserve(n * d)
                normalize(x, weight: layer.feedForwardNorm, bias: layer.feedForwardNormBias, rows: n, columns: d, into: h)
                input = In(h)
            }
            feedForward(index, input: input, count: n, into: ffn)
            finishBranch(ffn, postNorm: layer.postFeedForwardNorm, count: n)
            MiniVector.add(x, ffn, x, n * d)
        }
        if let perLayerInputs { addPerLayerInput(index, x: x, inputs: perLayerInputs, count: n) }
        if layer.outputScale != 1 { MiniVector.scale(x, layer.outputScale, x, n * d) }
    }

    /// 注意・FFN の出力の仕上げ（後ろの正規化と倍率）
    private func finishBranch(_ out: Out, postNorm: In?, count n: Int) {
        let d = config.embeddingSize
        if let postNorm { normalize(out, weight: postNorm, bias: nil, rows: n, columns: d, into: out) }
        if config.residualScale != 1 { MiniVector.scale(out, config.residualScale, out, n * d) }
    }

    private func normalize(_ x: In, weight: In?, bias: In?, rows: Int, columns: Int, into output: Out, weightStride: Int = 0) {
        switch config.norm {
        case .rms:
            GGUFKernels.rmsNorm(x, weight: weight, rows: rows, columns: columns, epsilon: config.normEpsilon, into: output,
                                weightStride: weightStride)
        case .layer:
            GGUFKernels.layerNorm(x, weight: weight, bias: bias, rows: rows, columns: columns, epsilon: config.normEpsilon,
                                  into: output, weightStride: weightStride)
        }
    }

    private func linear(_ matrix: GGUFMatrix, bias: In?, input: In, count: Int, output: Out) {
        GGUFKernels.linear(matrix, input: input, count: count, output: output, workers: workers)
        if let bias { MiniMath.addBias(bias, to: output, rows: count, columns: matrix.rows) }
    }

    // MARK: - 注意

    private func attention(_ index: Int, input: In, start: Int, count n: Int, into out: Out) {
        let layer = layers[index], shape = shapes[index]
        if shape.convolution {
            convolve(index, input: input, start: start, count: n, into: out)
            return
        }
        let q = shape.querySize, kv = shape.kvSize
        let qs = queries.reserve(n * q)
        linear(layer.query, bias: layer.queryBias, input: input, count: n, output: qs)
        if config.qkNormAfterRope { rotate(qs, heads: shape.headCount, layer: index, count: n) }
        if let norm = layer.queryNorm {
            normalizeHeads(qs, weight: norm, perHead: layer.queryNormPerHead, heads: shape.headCount, size: shape.headSize, count: n)
        }
        if !config.qkNormAfterRope { rotate(qs, heads: shape.headCount, layer: index, count: n) }
        MiniVector.scale(qs, config.attentionScale, qs, n * q)

        if shape.kvLayer == index, let keyMatrix = layer.key {
            let ks = newKeys.reserve(n * kv), vs = newValues.reserve(n * kv)
            linear(keyMatrix, bias: layer.keyBias, input: input, count: n, output: ks)
            if let value = layer.value {
                linear(value, bias: layer.valueBias, input: input, count: n, output: vs)
            } else {
                // Gemma 4: v が無ければ、正規化する前の k を使う
                vs.update(from: ks, count: n * kv)
            }
            if config.qkNormAfterRope { rotate(ks, heads: shape.kvHeadCount, layer: index, count: n) }
            if let norm = layer.keyNorm {
                normalizeHeads(ks, weight: norm, perHead: layer.keyNormPerHead, heads: shape.kvHeadCount, size: shape.headSize, count: n)
            }
            if config.hasValueNorm {
                GGUFKernels.rmsNorm(vs, weight: nil, rows: n * shape.kvHeadCount, columns: shape.headSize,
                                    epsilon: config.normEpsilon, into: vs)
            }
            if !config.qkNormAfterRope { rotate(ks, heads: shape.kvHeadCount, layer: index, count: n) }
            store(ks, vs, layer: index, start: start, count: n)
        }
        let attention = attended.reserve(n * q)
        attend(index, queries: qs, start: start, count: n, into: attention)
        linear(layer.output, bias: layer.outputBias, input: attention, count: n, output: out)
    }

    /// 短い畳み込み（LFM2）: [b, c, x] = 射影、bx = b × x を位置ごとに覚え、
    /// 出力 = c × Σ_j 重み[j] × bx[位置 − (長さ − 1) + j] → 出力の射影
    private func convolve(_ index: Int, input: In, start: Int, count n: Int, into out: Out) {
        let layer = layers[index]
        guard let kernel = layer.convolutionKernel, let history = keys[index] else { return }
        let d = config.embeddingSize, length = config.convolutionLength
        let projected = queries.reserve(n * 3 * d)
        linear(layer.query, bias: nil, input: input, count: n, output: projected)
        for token in 0..<n {
            let row = projected + token * 3 * d
            MiniVector.multiply(row, row + 2 * d, history + (start + token) * d, d)
        }
        let mixed = attended.reserve(n * d)
        for token in 0..<n {
            let position = start + token
            let target = mixed + token * d
            for channel in 0..<d {
                var sum: Float = 0
                for offset in 0..<length {
                    let source = position - (length - 1) + offset
                    if source >= 0 { sum += kernel[channel * length + offset] * history[source * d + channel] }
                }
                target[channel] = sum
            }
            MiniVector.multiply(target, projected + token * 3 * d + d, target, d)
        }
        linear(layer.output, bias: nil, input: mixed, count: n, output: out)
    }

    /// q / k の正規化: 全体（OLMo 2）、ヘッドごとに同じ重み、ヘッドごとに違う重み（StableLM・Command R）
    private func normalizeHeads(_ x: Out, weight: In, perHead: Bool, heads: Int, size: Int, count n: Int) {
        if config.qkNormIsFull {
            normalize(x, weight: weight, bias: nil, rows: n, columns: heads * size, into: x)
        } else if perHead {
            for token in 0..<n {
                let row = x + token * heads * size
                normalize(row, weight: weight, bias: nil, rows: heads, columns: size, into: row, weightStride: size)
            }
        } else {
            normalize(x, weight: weight, bias: nil, rows: n * heads, columns: size, into: x)
        }
    }

    private func store(_ newKeys: In, _ newValues: In, layer: Int, start: Int, count: Int) {
        guard let keyCache = keys[layer], let valueCache = values[layer] else { return }
        let shape = shapes[layer]
        let size = shape.headSize, groups = shape.kvHeadCount
        for group in 0..<groups {
            let keyBase = keyCache + (group * maxContext + start) * size
            let valueBase = valueCache + (group * maxContext + start) * size
            for index in 0..<count {
                (keyBase + index * size).update(from: newKeys + (index * groups + group) * size, count: size)
                (valueBase + index * size).update(from: newValues + (index * groups + group) * size, count: size)
            }
        }
    }

    private func attend(_ layer: Int, queries: In, start: Int, count: Int, into output: Out) {
        let shape = shapes[layer]
        guard let keyCache = keys[shape.kvLayer], let valueCache = values[shape.kvLayer] else {
            output.update(repeating: 0, count: count * shape.querySize)
            return
        }
        let size = shape.headSize, q = shape.querySize
        let total = start + count
        let groupSize = shape.headCount / shape.kvHeadCount
        let window = shape.sliding ? config.slidingWindow : 0
        let score = scores.reserve(count * total)
        for head in 0..<shape.headCount {
            let group = head / groupSize
            let keyBase = keyCache + group * maxContext * size
            let valueBase = valueCache + group * maxContext * size
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

    // MARK: - FFN

    private func feedForward(_ index: Int, input: In, count n: Int, into out: Out) {
        let layer = layers[index]
        let ff = shapes[index].feedForwardSize
        let work = scratch.reserve(n * ff)
        let up = upBuffer.reserve(n * ff)
        linear(layer.up, bias: layer.upBias, input: input, count: n, output: up)
        if let gateMatrix = layer.gate {
            let gate = gateBuffer.reserve(n * ff)
            linear(gateMatrix, bias: layer.gateBias, input: input, count: n, output: gate)
            GGUFKernels.gatedActivation(config.activation, gate: gate, up: up, count: n * ff, scratch: work)
            linear(layer.down, bias: layer.downBias, input: gate, count: n, output: out)
        } else {
            GGUFKernels.activate(config.activation, up, count: n * ff, scratch: work)
            linear(layer.down, bias: layer.downBias, input: up, count: n, output: out)
        }
    }

    /// 層ごとの入力を足す（Gemma 4）
    private func addPerLayerInput(_ index: Int, x: Out, inputs: In, count n: Int) {
        let layer = layers[index]
        guard let inputGate = layer.inputGate, let projection = layer.projection else { return }
        let size = config.perLayerInputSize, total = size * config.layerCount, d = config.embeddingSize
        let gate = perLayerWork.reserve(n * size)
        linear(inputGate, bias: nil, input: x, count: n, output: gate)
        GGUFKernels.activate(.gelu, gate, count: n * size, scratch: scratch.reserve(n * size))
        for token in 0..<n {
            MiniVector.multiply(gate + token * size, inputs + token * total + index * size, gate + token * size, size)
        }
        let out = feedForwardOut.reserve(n * d)
        linear(projection, bias: nil, input: gate, count: n, output: out)
        GGUFKernels.rmsNorm(out, weight: layer.perLayerPostNorm, rows: n, columns: d, epsilon: config.normEpsilon, into: out)
        MiniVector.add(x, out, x, n * d)
    }

    // MARK: - 出力

    private func logits(from last: Out) -> [Float] {
        let d = config.embeddingSize
        let h = normed.reserve(d)
        normalize(last, weight: globals.outputNorm, bias: globals.outputNormBias, rows: 1, columns: d, into: h)
        var result = [Float](repeating: 0, count: max(config.vocabSize, globals.output.rows))
        result.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            GGUFKernels.linear(globals.output, input: h, count: 1, output: base, workers: workers)
            GGUFLogits.finish(base, computedRows: globals.output.rows, config: config, bias: globals.outputBias)
        }
        if result.count > config.vocabSize { result.removeLast(result.count - config.vocabSize) }
        return result
    }

    // MARK: - RoPE

    /// 位置 start ..< start + count の cos / sin を用意する
    private func prepareRotations(start: Int, count: Int) {
        fill(fullRope, cos: fullCos, sin: fullSin, start: start, count: count)
        fill(slidingRope, cos: slidingCos, sin: slidingSin, start: start, count: count)
    }

    private func fill(_ table: GGUFRope.Table, cos cosBuffer: GGUFBuffer, sin sinBuffer: GGUFBuffer, start: Int, count: Int) {
        let pairs = table.pairs
        guard pairs > 0 else { return }
        let angle = angles.reserve(pairs)
        let cos = cosBuffer.reserve(count * pairs), sin = sinBuffer.reserve(count * pairs)
        table.frequencies.withUnsafeBufferPointer { frequency in
            guard let base = frequency.baseAddress else { return }
            for index in 0..<count {
                MiniVector.scale(base, Float(start + index), angle, pairs)
                GGUFKernels.cosSin(angle, count: pairs, cos: cos + index * pairs, sin: sin + index * pairs)
            }
        }
        if table.magnitude != 1 {
            MiniVector.scale(cos, table.magnitude, cos, count * pairs)
            MiniVector.scale(sin, table.magnitude, sin, count * pairs)
        }
    }

    /// [count][heads][headSize] の各ヘッドを、その位置の角度で回転させる（その層で RoPE を使わなければ何もしない）
    private func rotate(_ vectors: Out, heads: Int, layer: Int, count: Int) {
        let shape = shapes[layer]
        guard shape.usesRope else { return }
        let table = shape.sliding ? slidingRope : fullRope
        let pairs = table.pairs
        guard pairs > 0 else { return }
        let cos = (shape.sliding ? slidingCos : fullCos).pointer
        let sin = (shape.sliding ? slidingSin : fullSin).pointer
        let size = shape.headSize
        let work = scratch.reserve(2 * pairs)
        for index in 0..<count {
            for head in 0..<heads {
                GGUFKernels.rotate(vectors + (index * heads + head) * size, pairs: pairs, split: config.ropePairsAreSplit,
                                   cos: cos + index * pairs, sin: sin + index * pairs, scratch: work)
            }
        }
    }
}
