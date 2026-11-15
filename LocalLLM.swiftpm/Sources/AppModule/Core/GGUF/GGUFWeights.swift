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

/// 1 層ぶんの重み（保存ファイルの中身を指す）。CPU 用と GPU 用のどちらの計算でも使う。
struct GGUFLayerWeights {
    var attentionNorm: UnsafePointer<Float>?
    var attentionNormBias: UnsafePointer<Float>?
    var query: GGUFMatrix
    /// KV キャッシュをほかの層から借りる層（Gemma 4）では nil
    var key: GGUFMatrix?
    /// nil なら k をそのまま使う（Gemma 4）
    var value: GGUFMatrix?
    var queryBias: UnsafePointer<Float>?
    var keyBias: UnsafePointer<Float>?
    var valueBias: UnsafePointer<Float>?
    var output: GGUFMatrix
    var outputBias: UnsafePointer<Float>?
    var queryNorm: UnsafePointer<Float>?
    var keyNorm: UnsafePointer<Float>?
    /// q / k の正規化の重みが、ヘッドごとに違う（StableLM・Command R）
    var queryNormPerHead = false
    var keyNormPerHead = false
    var postAttentionNorm: UnsafePointer<Float>?
    var feedForwardNorm: UnsafePointer<Float>?
    var feedForwardNormBias: UnsafePointer<Float>?
    /// gated のときだけ
    var gate: GGUFMatrix?
    var up: GGUFMatrix
    var down: GGUFMatrix
    var gateBias: UnsafePointer<Float>?
    var upBias: UnsafePointer<Float>?
    var downBias: UnsafePointer<Float>?
    var postFeedForwardNorm: UnsafePointer<Float>?
    /// 層ごとの入力（Gemma 4）: gate = gelu(inputGate · x) × その層の入力 → projection → 正規化 → 足す
    var inputGate: GGUFMatrix?
    var projection: GGUFMatrix?
    var perLayerPostNorm: UnsafePointer<Float>?
    /// 層の出力に掛ける倍率（Gemma 4）
    var outputScale: Float = 1

    static func load(_ index: Int, store: GGUFWeightStore, config: GGUFModelConfig) throws -> GGUFLayerWeights {
        let prefix = "blk.\(index)."
        let shape = config.layers[index]
        let q = shape.querySize, kv = shape.kvSize, ff = shape.feedForwardSize
        let ownsCache = shape.kvLayer == index
        var query: GGUFMatrix, key: GGUFMatrix?, value: GGUFMatrix?
        var queryBias: UnsafePointer<Float>?, keyBias: UnsafePointer<Float>?, valueBias: UnsafePointer<Float>?
        if store.has(prefix + "attn_qkv.weight") {
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
            key = ownsCache ? try store.matrix(prefix + "attn_k.weight") : nil
            value = ownsCache && store.has(prefix + "attn_v.weight") ? try store.matrix(prefix + "attn_v.weight") : nil
        }
        queryBias = queryBias ?? store.vector(prefix + "attn_q.bias")
        keyBias = keyBias ?? store.vector(prefix + "attn_k.bias")
        valueBias = valueBias ?? store.vector(prefix + "attn_v.bias")

        var weights = GGUFLayerWeights(
            query: query, key: ownsCache ? key : nil, value: ownsCache ? value : nil,
            queryBias: queryBias, keyBias: keyBias, valueBias: valueBias,
            output: try store.matrix(prefix + "attn_output.weight"),
            // ERNIE 4.5 は、注意の出力のバイアスを llama.cpp も使わない
            outputBias: config.family == .ernie45 ? nil : store.vector(prefix + "attn_output.bias"),
            up: try store.matrix(prefix + "ffn_up.weight"),
            down: try store.matrix(prefix + "ffn_down.weight"))
        weights.loadNorms(prefix: prefix, store: store, config: config, shape: shape)
        try weights.loadFeedForward(prefix: prefix, store: store, config: config, size: ff)
        if config.perLayerInputSize > 0 {
            weights.inputGate = try store.matrix(prefix + "inp_gate.weight")
            weights.projection = try store.matrix(prefix + "proj.weight")
            weights.perLayerPostNorm = try store.requiredVector(prefix + "post_norm.weight")
        }
        if let scale = store.vector(prefix + "layer_output_scale.weight") { weights.outputScale = scale[0] }
        return weights
    }

    private mutating func loadNorms(prefix: String, store: GGUFWeightStore, config: GGUFModelConfig, shape: GGUFModelConfig.Layer) {
        if config.family == .falcon, let second = store.vector(prefix + "attn_norm_2.weight") {
            // Falcon 40B: 注意は 2 つ目の正規化、FFN は 1 つ目の正規化から
            attentionNorm = second
            attentionNormBias = store.vector(prefix + "attn_norm_2.bias")
            feedForwardNorm = store.vector(prefix + "attn_norm.weight")
            feedForwardNormBias = store.vector(prefix + "attn_norm.bias")
        } else {
            attentionNorm = store.vector(prefix + "attn_norm.weight")
            attentionNormBias = store.vector(prefix + "attn_norm.bias")
            // Seed-OSS は FFN の前の正規化を post_attention_norm という名前で持つ
            let name = config.family == .seedOSS ? "post_attention_norm" : "ffn_norm"
            feedForwardNorm = store.vector(prefix + name + ".weight")
            feedForwardNormBias = store.vector(prefix + name + ".bias")
        }
        if config.hasQKNorm {
            queryNorm = store.vector(prefix + "attn_q_norm.weight")
            keyNorm = store.vector(prefix + "attn_k_norm.weight")
            if !config.qkNormIsFull {
                queryNormPerHead = (store.length(prefix + "attn_q_norm.weight") ?? 0) == shape.querySize && shape.headCount > 1
                keyNormPerHead = (store.length(prefix + "attn_k_norm.weight") ?? 0) == shape.kvSize && shape.kvHeadCount > 1
            }
        }
        if config.hasPostNorms {
            postAttentionNorm = store.vector(prefix + "post_attention_norm.weight")
            postFeedForwardNorm = store.vector(prefix + "post_ffw_norm.weight")
        }
    }

    private mutating func loadFeedForward(prefix: String, store: GGUFWeightStore, config: GGUFModelConfig, size ff: Int) throws {
        downBias = store.vector(prefix + "ffn_down.bias")
        guard config.feedForward == .gated else {
            upBias = store.vector(prefix + "ffn_up.bias")
            return
        }
        if config.hasFusedGateUp {
            // gate と up が 1 つの行列（前半が gate）
            let fused = up
            gate = fused.slice(0..<ff)
            up = fused.slice(ff..<(2 * ff))
        } else {
            gate = try store.matrix(prefix + "ffn_gate.weight")
            gateBias = store.vector(prefix + "ffn_gate.bias")
            upBias = store.vector(prefix + "ffn_up.bias")
        }
    }
}

/// 層の外の重み
struct GGUFGlobalWeights {
    var embedding: GGUFMatrix
    var positions: GGUFMatrix?
    var output: GGUFMatrix
    var outputBias: UnsafePointer<Float>?
    var outputNorm: UnsafePointer<Float>?
    var outputNormBias: UnsafePointer<Float>?
    var ropeFactors: UnsafePointer<Float>?
    /// 層ごとの入力（Gemma 4）
    var perLayerEmbedding: GGUFMatrix?
    var perLayerProjection: GGUFMatrix?
    var perLayerProjectionNorm: UnsafePointer<Float>?

    init(store: GGUFWeightStore, config: GGUFModelConfig) throws {
        embedding = try store.matrix("token_embd.weight")
        positions = config.learnedPositions ? try store.matrix("position_embd.weight") : nil
        output = try store.matrix(config.tiedOutput ? "token_embd.weight" : "output.weight")
        outputBias = store.vector("output.bias")
        outputNorm = store.vector("output_norm.weight")
        outputNormBias = store.vector("output_norm.bias")
        ropeFactors = store.vector("rope_freqs.weight")
        if config.perLayerInputSize > 0 {
            perLayerEmbedding = try store.matrix("per_layer_token_embd.weight")
            perLayerProjection = try store.matrix("per_layer_model_proj.weight")
            perLayerProjectionNorm = try store.requiredVector("per_layer_proj_norm.weight")
        }
    }
}

/// GGUF のモデルを計算するもの（CPU 用の GGUFModel と、GPU 用の GGUFMetalModel）。
protocol GGUFEvaluator: AnyObject {
    var config: GGUFModelConfig { get }
    /// 扱える最大のトークン数
    var maxContext: Int { get }
    /// KV キャッシュに入っているトークン数
    var length: Int { get }
    /// どこで計算しているか（画面に出す）
    var computeDescription: String { get }
    func truncate(to newLength: Int)
    func releaseCache()
    /// tokens を今の系列の後ろに足して計算し、最後のトークンの次の logits を返す
    @discardableResult
    func evaluate(_ tokens: [Int], batch: Int, progress: ((Int) throws -> Void)?) throws -> [Float]
}

extension GGUFEvaluator {
    @discardableResult
    func evaluate(_ tokens: [Int], batch: Int = GGUFModel.maxBatch) throws -> [Float] {
        try evaluate(tokens, batch: batch, progress: nil)
    }

    /// evaluate に渡すトークンの確認（どの計算でも同じ）
    func validate(_ tokens: [Int]) throws {
        guard !tokens.isEmpty else { throw GGUFError.invalid("no tokens") }
        guard length + tokens.count <= maxContext else {
            throw GGUFError.invalid(L("コンテキストの上限（\(maxContext) トークン）を超えました", "context limit of \(maxContext) tokens exceeded"))
        }
        if let bad = tokens.first(where: { $0 < 0 || $0 >= config.vocabSize }) { throw GGUFError.invalid("token id \(bad)") }
    }
}

/// CPU・GPU の計算で共通の、最後の logits の仕上げ
enum GGUFLogits {
    /// 出力層の行数が語彙より少ないとき（Qwen3-VL）は、残りを -∞ にする。倍率・バイアス・tanh の上限も掛ける
    static func finish(_ logits: UnsafeMutablePointer<Float>, computedRows: Int, config: GGUFModelConfig,
                       bias: UnsafePointer<Float>?) {
        if let bias { MiniMath.add(bias, into: logits, count: computedRows) }
        if config.logitScale != 1 { MiniVector.scale(logits, config.logitScale, logits, computedRows) }
        if config.finalSoftcap > 0 { GGUFKernels.softcap(logits, count: computedRows, cap: config.finalSoftcap) }
        if computedRows < config.vocabSize {
            (logits + computedRows).update(repeating: -Float.infinity, count: config.vocabSize - computedRows)
        }
    }
}
