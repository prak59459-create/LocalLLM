import Foundation

/// Gemma 4 の MTP（先読み・推測デコード）用の小さなモデル（GGUF の gemma4-assistant。E2B なら 4 層・約 77M）。
///
/// 本体が次のトークンを 1 つ決めるたびに、その後ろの数トークンを安く予想し、本体はそれをまとめて 1 回で確かめる
/// （予想が当たった分だけ、1 回の重みの読み込みで何トークンも進める。外れたら、そこまでで止めて本体の答えを使う）。
///
/// 1 回の予想の計算（llama.cpp の gemma4-assistant.cpp と同じ）:
/// - 入力 = [本体の埋め込み(トークン) × √(本体の大きさ), 本体の隠れ状態（最後の正規化のあと）] → 射影（pre_projection）
/// - 層: q だけを持ち、k・v は本体の KV キャッシュを使う（直近だけを見る層は本体の最後から 2 番目の層、全体を見る層は
///   最後の層が使うもの）。q の位置はどの予想でも同じ（本体がまだ読んでいない位置）
/// - 出力: 正規化 → logits（自分の埋め込みの表）と、次の予想に渡す隠れ状態（post_projection）
struct GGUFDrafterWeights {
    struct Layer {
        var attentionNorm: UnsafePointer<Float>
        var query: GGUFMatrix
        var queryNorm: UnsafePointer<Float>
        var output: GGUFMatrix
        var postAttentionNorm: UnsafePointer<Float>
        var feedForwardNorm: UnsafePointer<Float>
        var gate: GGUFMatrix
        var up: GGUFMatrix
        var down: GGUFMatrix
        var postFeedForwardNorm: UnsafePointer<Float>
        var outputScale: Float
    }

    var layers: [Layer]
    var preProjection: GGUFMatrix
    var postProjection: GGUFMatrix
    /// logits の表（自分の埋め込みの表と同じ）
    var output: GGUFMatrix
    var outputNorm: UnsafePointer<Float>
    var ropeFactors: UnsafePointer<Float>?

    init(store: GGUFWeightStore) throws {
        let config = store.config
        layers = try (0..<config.layerCount).map { index in
            let prefix = "blk.\(index)."
            return Layer(attentionNorm: try store.requiredVector(prefix + "attn_norm.weight"),
                         query: try store.matrix(prefix + "attn_q.weight"),
                         queryNorm: try store.requiredVector(prefix + "attn_q_norm.weight"),
                         output: try store.matrix(prefix + "attn_output.weight"),
                         postAttentionNorm: try store.requiredVector(prefix + "post_attention_norm.weight"),
                         feedForwardNorm: try store.requiredVector(prefix + "ffn_norm.weight"),
                         gate: try store.matrix(prefix + "ffn_gate.weight"),
                         up: try store.matrix(prefix + "ffn_up.weight"),
                         down: try store.matrix(prefix + "ffn_down.weight"),
                         postFeedForwardNorm: try store.requiredVector(prefix + "post_ffw_norm.weight"),
                         outputScale: store.vector(prefix + "layer_output_scale.weight")?.pointee ?? 1)
        }
        preProjection = try store.matrix("nextn.pre_projection.weight")
        postProjection = try store.matrix("nextn.post_projection.weight")
        output = try store.matrix("token_embd.weight")
        outputNorm = try store.requiredVector("output_norm.weight")
        ropeFactors = store.vector("rope_freqs.weight")
    }
}

extension GGUFDrafterWeights {
    /// 本体と組にできるか確かめ、使う本体の層（KV キャッシュ）を返す: (直近だけを見る層, 全体を見る層)
    static func cacheLayers(drafter: GGUFModelConfig, target: GGUFModelConfig) throws -> (sliding: Int, full: Int) {
        guard drafter.family == .gemma4Assistant, target.family == .gemma4,
              drafter.drafterOutputSize == target.embeddingSize, drafter.vocabSize <= target.vocabSize else {
            throw GGUFError.invalid(L("先読み用のモデルが、このモデルと組になりません", "The draft model does not match this model"))
        }
        let shapes = target.layers
        guard shapes.count >= 2 else { throw GGUFError.invalid("draft target layers") }
        let sliding = shapes[shapes.count - 2], full = shapes[shapes.count - 1]
        for shape in drafter.layers {
            let source = shape.sliding ? sliding : full
            guard source.sliding == shape.sliding, source.headSize == shape.headSize, source.kvHeadCount == 1,
                  shape.kvHeadCount == 1, !shape.convolution else {
                throw GGUFError.invalid(L("先読み用のモデルの形が、このモデルと合いません", "The draft model's shape does not match this model"))
            }
        }
        return (sliding.kvLayer, full.kvLayer)
    }
}

/// 先読みが本体から借りるもの（CPU 用の本体）
protocol GGUFDraftSource: AnyObject {
    var config: GGUFModelConfig { get }
    /// 本体の埋め込みの表の 1 行（埋め込みの倍率は掛けない）
    func embeddingRow(_ token: Int, into output: UnsafeMutablePointer<Float>)
    /// 層 layer の KV キャッシュ（KV のヘッドは 1 つ: [位置][ヘッドの大きさ]）
    func draftCache(layer: Int) -> (keys: UnsafePointer<Float>, values: UnsafePointer<Float>)?
}

/// 先読みを CPU で計算する（GPU が使えないときと、テストで GPU の計算を確かめる用）
final class GGUFDrafter {
    typealias Out = UnsafeMutablePointer<Float>

    let config: GGUFModelConfig
    private let store: GGUFWeightStore
    private let weights: GGUFDrafterWeights
    private let workers: GGUFWorkers
    private let fullRope: GGUFRope.Table
    private let slidingRope: GGUFRope.Table
    private let sources: (sliding: Int, full: Int)
    private let input = GGUFBuffer(), current = GGUFBuffer(), normed = GGUFBuffer(), queries = GGUFBuffer()
    private let attended = GGUFBuffer(), projected = GGUFBuffer(), gate = GGUFBuffer(), up = GGUFBuffer()
    private let scratch = GGUFBuffer(), scores = GGUFBuffer(), angles = GGUFBuffer(), cosines = GGUFBuffer(), sines = GGUFBuffer()

    init(store: GGUFWeightStore, target: GGUFModelConfig) throws {
        let config = store.config
        sources = try GGUFDrafterWeights.cacheLayers(drafter: config, target: target)
        self.store = store
        self.config = config
        weights = try GGUFDrafterWeights(store: store)
        let widest = max(2 * config.drafterOutputSize, config.maxFeedForwardSize, config.maxQuerySize, config.embeddingSize)
        workers = GGUFWorkers(widestRow: widest)
        fullRope = GGUFRope.table(config: config, sliding: false, factors: weights.ropeFactors)
        slidingRope = GGUFRope.table(config: config, sliding: true, factors: weights.ropeFactors)
    }

    /// 1 回の予想: 位置 position のトークン token と、位置 position − 1 の本体の隠れ状態 hidden から、
    /// 次のトークンの logits と、次の予想に渡す隠れ状態を返す（本体の KV キャッシュは位置 position − 1 までを使う）
    func step(token: Int, hidden: [Float], position: Int, target: GGUFDraftSource) -> (logits: [Float], hidden: [Float]) {
        let d = config.embeddingSize, outer = config.drafterOutputSize
        let x = input.reserve(2 * outer)
        target.embeddingRow(token, into: x)
        MiniVector.scale(x, Float(outer).squareRoot(), x, outer)
        hidden.withUnsafeBufferPointer { (x + outer).update(from: $0.baseAddress!, count: min(outer, $0.count)) }
        let cur = current.reserve(d)
        GGUFKernels.linear(weights.preProjection, input: x, count: 1, output: cur, workers: workers)
        for index in weights.layers.indices {
            runLayer(index, cur: cur, position: position, target: target)
        }
        let final = normed.reserve(d)
        GGUFKernels.rmsNorm(cur, weight: weights.outputNorm, rows: 1, columns: d, epsilon: config.normEpsilon, into: final)
        var logits = [Float](repeating: 0, count: weights.output.rows)
        var next = [Float](repeating: 0, count: outer)
        logits.withUnsafeMutableBufferPointer {
            GGUFKernels.linear(weights.output, input: final, count: 1, output: $0.baseAddress!, workers: workers)
        }
        next.withUnsafeMutableBufferPointer {
            GGUFKernels.linear(weights.postProjection, input: final, count: 1, output: $0.baseAddress!, workers: workers)
        }
        return (logits, next)
    }

    /// count 個を貪欲に予想する（logits の最大のトークン）
    func draft(after token: Int, hidden: [Float], position: Int, count: Int, target: GGUFDraftSource) -> [Int] {
        var drafts: [Int] = []
        var current = token, state = hidden
        for _ in 0..<max(0, count) {
            let (logits, next) = step(token: current, hidden: state, position: position, target: target)
            guard let best = GGUFDrafter.argmax(logits) else { break }
            drafts.append(best)
            current = best
            state = next
        }
        return drafts
    }

    static func argmax(_ values: [Float]) -> Int? {
        values.withUnsafeBufferPointer { buffer -> Int? in
            guard var best = buffer.indices.first else { return nil }
            for index in buffer.indices where buffer[index] > buffer[best] { best = index }
            return best
        }
    }

    private func runLayer(_ index: Int, cur: Out, position: Int, target: GGUFDraftSource) {
        let layer = weights.layers[index], shape = config.layers[index]
        let d = config.embeddingSize, size = shape.headSize, heads = shape.headCount, q = shape.querySize
        let epsilon = config.normEpsilon
        let h = normed.reserve(d)
        GGUFKernels.rmsNorm(cur, weight: layer.attentionNorm, rows: 1, columns: d, epsilon: epsilon, into: h)
        let qs = queries.reserve(q)
        GGUFKernels.linear(layer.query, input: h, count: 1, output: qs, workers: workers)
        GGUFKernels.rmsNorm(qs, weight: layer.queryNorm, rows: heads, columns: size, epsilon: epsilon, into: qs)
        rotate(qs, heads: heads, size: size, position: position, sliding: shape.sliding)
        let out = attended.reserve(q)
        attend(qs, heads: heads, size: size, position: position, sliding: shape.sliding, target: target, into: out)
        let attention = projected.reserve(d)
        GGUFKernels.linear(layer.output, input: out, count: 1, output: attention, workers: workers)
        GGUFKernels.rmsNorm(attention, weight: layer.postAttentionNorm, rows: 1, columns: d, epsilon: epsilon, into: attention)
        MiniVector.add(cur, attention, cur, d)
        // FFN
        GGUFKernels.rmsNorm(cur, weight: layer.feedForwardNorm, rows: 1, columns: d, epsilon: epsilon, into: h)
        let ff = shape.feedForwardSize
        let g = gate.reserve(ff), u = up.reserve(ff)
        GGUFKernels.linear(layer.gate, input: h, count: 1, output: g, workers: workers)
        GGUFKernels.linear(layer.up, input: h, count: 1, output: u, workers: workers)
        GGUFKernels.gatedActivation(.gelu, gate: g, up: u, count: ff, scratch: scratch.reserve(ff))
        GGUFKernels.linear(layer.down, input: g, count: 1, output: attention, workers: workers)
        GGUFKernels.rmsNorm(attention, weight: layer.postFeedForwardNorm, rows: 1, columns: d, epsilon: epsilon, into: attention)
        MiniVector.add(cur, attention, cur, d)
        if layer.outputScale != 1 { MiniVector.scale(cur, layer.outputScale, cur, d) }
    }

    /// q を位置 position の角度で回す（NEOX: 前半と後半を組にする）
    private func rotate(_ qs: Out, heads: Int, size: Int, position: Int, sliding: Bool) {
        let table = sliding ? slidingRope : fullRope
        let pairs = table.pairs
        guard pairs > 0 else { return }
        let angle = angles.reserve(pairs), cos = cosines.reserve(pairs), sin = sines.reserve(pairs)
        table.frequencies.withUnsafeBufferPointer { MiniVector.scale($0.baseAddress!, Float(position), angle, pairs) }
        GGUFKernels.cosSin(angle, count: pairs, cos: cos, sin: sin)
        if table.magnitude != 1 {
            MiniVector.scale(cos, table.magnitude, cos, pairs)
            MiniVector.scale(sin, table.magnitude, sin, pairs)
        }
        let work = scratch.reserve(2 * pairs)
        for head in 0..<heads {
            GGUFKernels.rotate(qs + head * size, pairs: pairs, split: config.ropePairsAreSplit, cos: cos, sin: sin, scratch: work)
        }
    }

    /// 本体の KV キャッシュの位置 0 … position − 1 を見る（直近だけを見る層は、position から窓の大きさ未満の位置）
    private func attend(_ qs: Out, heads: Int, size: Int, position: Int, sliding: Bool, target: GGUFDraftSource, into out: Out) {
        MiniVector.fillZero(out, heads * size)
        guard position > 0, let cache = target.draftCache(layer: sliding ? sources.sliding : sources.full) else { return }
        let first = sliding && config.slidingWindow > 0 ? max(0, position - config.slidingWindow + 1) : 0
        let count = position - first
        let row = scores.reserve(count)
        for head in 0..<heads {
            let query = qs + head * size
            for t in 0..<count {
                row[t] = MiniMath.dot(query, cache.keys + (first + t) * size, count: size) * config.attentionScale
            }
            GGUFKernels.softmax(row, count: count)
            for t in 0..<count {
                MiniVector.scaleAdd(cache.values + (first + t) * size, row[t], out + head * size, out + head * size, size)
            }
        }
    }
}
