import Foundation

/// GGUF のメタデータと重みの一覧から、モデルの構造を読み取る（llama.cpp の load_hparams / build_graph と同じ規則）。
extension GGUFModelConfig {
    /// 系列ごとの、計算の流れの違い
    private struct Rules {
        var splitPairs = true
        var norm: Norm = .rms
        var feedForward: FeedForward = .gated
        var activation: Activation = .silu
        var parallel = false
        var preNorms = true
        var gemmaEmbedding = false
        var rope = true
        var learnedPositions = false
    }

    private static func rules(_ family: Family, file: GGUFFile) -> Rules {
        var rules = Rules()
        switch family {
        case .llama, .mistral3, .internlm2, .xverse, .baichuan, .ernie45, .granite, .minicpm, .smollm3, .maincoder, .chatglm:
            rules.splitPairs = false
        case .glm4:
            // 画像も扱う版（GLM-4V）は M-RoPE（文章だけなら NEOX と同じ）
            rules.splitPairs = file.metadata["glm4.rope.dimension_sections"] != nil
        case .arcee:
            rules.splitPairs = false
            rules.feedForward = .sequential
            rules.activation = .reluSquared
        case .commandR, .cohere2:
            rules.splitPairs = false
            rules.norm = .layer
            rules.parallel = true
        case .olmo:
            rules.splitPairs = false
            rules.norm = .layer
        case .qwen, .qwen2, .qwen2vl, .qwen3, .qwen3vl, .phi3, .exaone, .seedOSS, .openelm:
            break
        case .plamo:
            rules.parallel = true
        case .gemma, .gemma2, .gemma3, .gemma4:
            rules.activation = .gelu
            rules.gemmaEmbedding = true
        case .olmo2, .exaone4:
            rules.preNorms = false
        case .phi2, .falcon:
            rules.norm = .layer
            rules.feedForward = .sequential
            rules.activation = .gelu
            rules.parallel = true
        case .gptneox:
            rules.norm = .layer
            rules.feedForward = .sequential
            rules.activation = .gelu
            rules.parallel = file.bool("gptneox.use_parallel_residual") ?? true
        case .gpt2, .starcoder:
            rules.norm = .layer
            rules.feedForward = .sequential
            rules.activation = .gelu
            rules.rope = false
            rules.learnedPositions = true
        case .starcoder2, .codeshell:
            rules.norm = .layer
            rules.feedForward = .sequential
            rules.activation = .gelu
        case .stablelm:
            rules.norm = .layer
            rules.parallel = !file.has("blk.0.ffn_norm.weight")
        case .orion:
            rules.norm = .layer
        case .nemotron, .jais2:
            rules.norm = .layer
            rules.feedForward = .sequential
            rules.activation = .reluSquared
        }
        return rules
    }

    /// メタデータと重みの一覧から構造を決める。対応していない構造なら GGUFError を投げる。
    init(file: GGUFFile) throws {
        let arch = file.architecture
        guard let family = Self.supportedArchitectures[arch] else {
            throw GGUFError.unsupportedArchitecture(arch.isEmpty ? "?" : arch)
        }
        guard let embedding = file.tensor("token_embd.weight"), embedding.dimensions.count == 2 else {
            throw GGUFError.missingTensor("token_embd.weight")
        }
        // 混合エキスパート（MoE）は扱わない
        if file.has("blk.0.ffn_gate_inp.weight") || (file.archInt("expert_count") ?? 0) > 0 {
            throw GGUFError.unsupportedArchitecture("\(arch) (MoE)")
        }
        let rules = Self.rules(family, file: file)
        architecture = arch
        self.family = family
        name = file.name
        quantization = file.dominantQuantization
        layerCount = try Self.require(file, "block_count")
        embeddingSize = try Self.require(file, "embedding_length")
        guard layerCount > 0, embedding.columns == embeddingSize else { throw GGUFError.invalid("token_embd shape") }
        vocabSize = embedding.rows
        contextLength = max(8, file.archInt("context_length") ?? 2048)
        norm = rules.norm
        let epsilonKey = rules.norm == .rms ? "attention.layer_norm_rms_epsilon" : "attention.layer_norm_epsilon"
        normEpsilon = Float(file.archDouble(epsilonKey) ?? 1e-5)
        feedForward = rules.feedForward
        activation = rules.activation
        parallelResidual = rules.parallel
        hasPreNorms = rules.preNorms
        learnedPositions = rules.learnedPositions
        ropePairsAreSplit = rules.splitPairs
        tiedOutput = !file.has("output.weight")
        hasFusedQKV = file.has("blk.0.attn_qkv.weight")
        hasFusedGateUp = rules.feedForward == .gated && !file.has("blk.0.ffn_gate.weight")

        // 直近だけを見る層
        (slidingWindow, slidingLayers) = Self.sliding(family: family, file: file, layers: layerCount)

        // 層ごとの形
        let heads = try Self.perLayer(file, "attention.head_count", layers: layerCount)
        let kvHeads = Self.perLayerOptional(file, "attention.head_count_kv", layers: layerCount) ?? heads
        let fullHead = file.archInt("attention.key_length") ?? (embeddingSize / max(heads[0], 1))
        let slidingHead = file.archInt("attention.key_length_swa") ?? fullHead
        if let value = file.archInt("attention.value_length"), value != fullHead {
            throw GGUFError.unsupportedArchitecture("\(arch) (key_length != value_length)")
        }
        if let value = file.archInt("attention.value_length_swa"), value != slidingHead {
            throw GGUFError.unsupportedArchitecture("\(arch) (key_length_swa != value_length_swa)")
        }
        let shared = family == .gemma4 ? (file.archInt("attention.shared_kv_layers") ?? 0) : 0
        let ownCache = layerCount - shared
        var shapes: [Layer] = []
        for index in 0..<layerCount {
            let sliding = slidingWindow > 0 && slidingLayers[index]
            let usesRope = Self.usesRope(family: family, file: file, layer: index, sliding: sliding, hasSliding: slidingWindow > 0)
            var kvLayer = index
            if index >= ownCache { kvLayer = ownCache - (sliding ? 2 : 1) }
            let ff = try Self.feedForwardSize(file, layer: index, fused: hasFusedGateUp)
            shapes.append(Layer(headCount: heads[index], kvHeadCount: kvHeads[index], headSize: sliding ? slidingHead : fullHead,
                                feedForwardSize: ff, usesRope: rules.rope && usesRope, sliding: sliding, kvLayer: kvLayer))
        }
        for (index, shape) in shapes.enumerated() {
            guard shape.headCount > 0, shape.kvHeadCount > 0, shape.headCount % shape.kvHeadCount == 0,
                  shape.kvLayer >= 0, shape.kvLayer <= index else { throw GGUFError.invalid("head counts") }
            let source = shapes[shape.kvLayer]
            guard source.kvLayer == shape.kvLayer, source.kvSize == shape.kvSize, source.sliding == shape.sliding else {
                throw GGUFError.invalid("shared KV layer \(index)")
            }
        }
        layerShapes = shapes
        headCount = heads.max() ?? 1
        kvHeadCount = kvHeads.max() ?? 1
        headSize = max(fullHead, slidingHead)
        feedForwardSize = shapes.map(\.feedForwardSize).max() ?? 0

        // RoPE
        ropeDimensions = file.archInt("rope.dimension_count") ?? fullHead
        slidingRopeDimensions = file.archInt("rope.dimension_count_swa") ?? min(ropeDimensions, slidingHead)
        guard ropeDimensions % 2 == 0, ropeDimensions <= fullHead, slidingRopeDimensions % 2 == 0,
              slidingRopeDimensions <= slidingHead else { throw GGUFError.invalid("rope dimensions") }
        ropeBase = Float(file.archDouble("rope.freq_base") ?? 10000)
        slidingRopeBase = Float(file.archDouble("rope.freq_base_swa") ?? (family == .gemma3 ? 10000 : Double(ropeBase)))
        ropeFactorsOnSlidingLayers = family != .gemma4
        (ropeScale, yarn) = Self.ropeScaling(file)
        // llama.cpp: 直近だけを見る層は、系列によって全体と同じ倍率か、伸ばさない（1）
        switch family {
        case .gemma2, .cohere2, .exaone4: slidingRopeScale = ropeScale
        default: slidingRopeScale = 1
        }
        if yarn == nil, let original = file.archInt("rope.scaling.original_context_length"), original > 0 {
            // 学習時より長い文脈に伸ばす仕組みのうち、YaRN 以外は使わないので、学習時の長さまでにする
            contextLength = min(contextLength, original)
        }
        if family == .mistral3, (file.archDouble("attention.temperature_scale") ?? 0) != 0 {
            // 長い文脈で q を強める仕組みは、元の長さの中では効かないので、その長さまでにする
            contextLength = min(contextLength, yarn?.originalContext ?? contextLength)
        }
        if rules.learnedPositions {
            guard let positions = file.tensor("position_embd.weight") else { throw GGUFError.missingTensor("position_embd.weight") }
            contextLength = min(contextLength, positions.rows)
        }

        // 倍率
        attentionScale = 1 / Float(fullHead).squareRoot()
        if let scale = file.archDouble("attention.scale"), scale != 0 { attentionScale = Float(scale) }
        if family == .gemma4 { attentionScale = 1 }
        embeddingScale = rules.gemmaEmbedding ? Float(embeddingSize).squareRoot() : 1
        residualScale = 1
        logitScale = 1
        attentionSoftcap = 0
        finalSoftcap = 0

        // 正規化
        hasPostNorms = family != .seedOSS && file.has("blk.0.post_attention_norm.weight") && file.has("blk.0.post_ffw_norm.weight")
        let qNorm = file.tensor("blk.0.attn_q_norm.weight")
        hasQKNorm = qNorm != nil
        qkNormIsFull = (qNorm?.dimensions.count ?? 0) == 1 && (qNorm?.elementCount ?? 0) == shapes[0].querySize
            && shapes[0].headCount > 1
        hasValueNorm = family == .gemma4
        qkNormAfterRope = family == .maincoder
        perLayerInputSize = file.has("per_layer_token_embd.weight") ? (file.archInt("embedding_length_per_layer_input") ?? 0) : 0
        applyScales(family: family, file: file)

        if family == .baichuan && layerCount != 32 {
            // llama.cpp は 7B（32 層）だけ RoPE、それ以外は ALiBi
            throw GGUFError.unsupportedArchitecture("baichuan (ALiBi)")
        }
        if family == .olmo, (file.archDouble("attention.clamp_kqv") ?? 0) > 0 {
            throw GGUFError.unsupportedArchitecture("olmo (clamp_kqv)")
        }
        try validateTensors(file)
    }

    private static func require(_ file: GGUFFile, _ suffix: String) throws -> Int {
        guard let value = file.archInt(suffix) else { throw GGUFError.missingMetadata("\(file.architecture).\(suffix)") }
        return value
    }

    /// 1 つの値か、層ごとの配列
    private static func perLayerOptional(_ file: GGUFFile, _ suffix: String, layers: Int) -> [Int]? {
        let key = "\(file.architecture).\(suffix)"
        if let values = file.metadata[key]?.intArray, values.count >= layers { return Array(values.prefix(layers)) }
        if let value = file.int(key) { return [Int](repeating: value, count: layers) }
        return nil
    }

    private static func perLayer(_ file: GGUFFile, _ suffix: String, layers: Int) throws -> [Int] {
        guard let values = perLayerOptional(file, suffix, layers: layers) else {
            throw GGUFError.missingMetadata("\(file.architecture).\(suffix)")
        }
        return values
    }

    private static func feedForwardSize(_ file: GGUFFile, layer: Int, fused: Bool) throws -> Int {
        if let down = file.tensor("blk.\(layer).ffn_down.weight") { return down.columns }
        if let sizes = perLayerOptional(file, "feed_forward_length", layers: layer + 1) { return sizes[layer] }
        throw GGUFError.missingMetadata("\(file.architecture).feed_forward_length")
    }

    /// 直近だけを見る層の窓と並び
    private static func sliding(family: Family, file: GGUFFile, layers: Int) -> (Int, [Bool]) {
        let window = file.archInt("attention.sliding_window") ?? 0
        let period: Int
        switch family {
        case .gemma2: return (window > 0 ? window : 4096, slidingPattern(file: file, layers: layers, defaultPeriod: 2))
        case .gemma3: period = 6
        case .gemma4: period = 5
        case .cohere2, .olmo2: period = 4
        case .exaone4:
            // llama.cpp は 32B（64 層）だけ直近を見る層を使う
            guard layers == 64 else { return (0, []) }
            return (window > 0 ? window : 4096, slidingPattern(file: file, layers: layers, defaultPeriod: 4))
        default: return (0, [])
        }
        guard window > 0 else { return (0, []) }
        return (window, slidingPattern(file: file, layers: layers, defaultPeriod: period))
    }

    /// 直近だけを見る層の並び（メタデータに無ければ、period 層ごとに最後の 1 層だけ全体を見る）
    private static func slidingPattern(file: GGUFFile, layers: Int, defaultPeriod: Int) -> [Bool] {
        let key = "\(file.architecture).attention.sliding_window_pattern"
        if let flags = file.metadata[key]?.intArray, flags.count >= layers {
            return (0..<layers).map { flags[$0] != 0 }
        }
        let period = file.int(key) ?? defaultPeriod
        return (0..<layers).map { period == 0 || ($0 % period) < (period - 1) }
    }

    /// その層で位置の回転（RoPE）を使うか
    private static func usesRope(family: Family, file: GGUFFile, layer: Int, sliding: Bool, hasSliding: Bool) -> Bool {
        switch family {
        case .smollm3:
            // 4 層ごとに 1 層は回転しない（NoPE）
            return (layer + 1) % 4 != 0
        case .cohere2:
            return sliding
        case .exaone4:
            return sliding || !hasSliding
        case .granite:
            return file.bool("granite.rope.scaling.finetuned") ?? true
        default:
            return true
        }
    }

    /// RoPE の周波数の倍率と YaRN の値
    private static func ropeScaling(_ file: GGUFFile) -> (Float, YaRN?) {
        let factor = file.archDouble("rope.scaling.factor") ?? file.archDouble("rope.scale_linear") ?? 0
        let type = file.string("\(file.architecture).rope.scaling.type") ?? (factor > 0 ? "linear" : "none")
        guard type != "none", factor > 0 else { return (1, nil) }
        let scale = Float(1 / factor)
        guard type == "yarn", factor > 1 else { return (type == "linear" ? scale : 1, nil) }
        let original = file.archInt("rope.scaling.original_context_length") ?? (file.archInt("context_length") ?? 2048)
        // llama.cpp: cos / sin に (1 + 0.1 ln f) を掛ける（log_mul があれば、その分を割る）
        let logFactor = Float(Foundation.log(factor))
        var magnitude = 1 + 0.1 * logFactor
        if let logMul = file.archDouble("rope.scaling.yarn_log_multiplier"), logMul != 0 {
            magnitude /= 1 + 0.1 * Float(logMul) * logFactor
        }
        magnitude *= Float(file.archDouble("rope.scaling.attn_factor") ?? 1)
        let yarn = YaRN(originalContext: original,
                        betaFast: Float(file.archDouble("rope.scaling.yarn_beta_fast") ?? 32),
                        betaSlow: Float(file.archDouble("rope.scaling.yarn_beta_slow") ?? 1),
                        magnitude: magnitude)
        return (scale, yarn)
    }

    /// 系列ごとの倍率（埋め込み・FFN の出力・logits・tanh の上限）
    private mutating func applyScales(family: Family, file: GGUFFile) {
        switch family {
        case .gemma2:
            attentionSoftcap = Float(file.archDouble("attn_logit_softcapping") ?? 50)
            finalSoftcap = Float(file.archDouble("final_logit_softcapping") ?? 30)
        case .gemma3, .gemma4:
            finalSoftcap = Float(file.archDouble("final_logit_softcapping") ?? 0)
        case .commandR, .cohere2:
            let scale = Float(file.archDouble("logit_scale") ?? 0)
            logitScale = scale == 0 ? 1 : scale
        case .granite:
            let logit = Float(file.archDouble("logit_scale") ?? 1)
            logitScale = logit == 0 ? 1 : 1 / logit
            let residual = Float(file.archDouble("residual_scale") ?? 0)
            residualScale = residual == 0 ? 1 : residual
            let embedding = Float(file.archDouble("embedding_scale") ?? 0)
            embeddingScale = embedding == 0 ? 1 : embedding
        case .minicpm:
            let logit = Float(file.archDouble("logit_scale") ?? Double(256 / Float(embeddingSize)))
            logitScale = logit == 0 ? 1 : 1 / logit
            residualScale = Float(file.archDouble("residual_scale") ?? Double(1.4 / Float(layerCount).squareRoot()))
            embeddingScale = Float(file.archDouble("embedding_scale") ?? 12)
        default:
            break
        }
    }

    // MARK: - 重みの確認

    /// 必要な重みがそろっていて、形が合っているかを確かめる
    private func validateTensors(_ file: GGUFFile) throws {
        try check(file, "output_norm.weight", rows: 1, columns: embeddingSize, optional: family == .olmo)
        if let output = file.tensor("output.weight") {
            guard output.columns == embeddingSize, output.rows <= vocabSize else { throw GGUFError.invalid("output.weight shape") }
        }
        if perLayerInputSize > 0 {
            try check(file, "per_layer_token_embd.weight", rows: vocabSize, columns: perLayerInputSize * layerCount)
            try check(file, "per_layer_model_proj.weight", rows: perLayerInputSize * layerCount, columns: embeddingSize)
            try check(file, "per_layer_proj_norm.weight", rows: 1, columns: perLayerInputSize)
        }
        for (layer, shape) in layers.enumerated() {
            try validateLayer(file, layer: layer, shape: shape)
        }
    }

    private func validateLayer(_ file: GGUFFile, layer: Int, shape: Layer) throws {
        let prefix = "blk.\(layer)."
        let d = embeddingSize, q = shape.querySize, kv = shape.kvSize, ff = shape.feedForwardSize
        let ownsCache = shape.kvLayer == layer
        if hasPreNorms && family != .olmo {
            try check(file, prefix + (family == .falcon && file.has(prefix + "attn_norm_2.weight") ? "attn_norm_2.weight" : "attn_norm.weight"),
                      rows: 1, columns: d)
        }
        if hasFusedQKV {
            try check(file, prefix + "attn_qkv.weight", rows: q + 2 * kv, columns: d)
        } else {
            try check(file, prefix + "attn_q.weight", rows: q, columns: d)
            if ownsCache {
                try check(file, prefix + "attn_k.weight", rows: kv, columns: d)
                // Gemma 4 は v が無ければ k を使う
                try check(file, prefix + "attn_v.weight", rows: kv, columns: d, optional: family == .gemma4)
            }
        }
        try check(file, prefix + "attn_output.weight", rows: d, columns: q)
        switch feedForward {
        case .gated where hasFusedGateUp:
            try check(file, prefix + "ffn_up.weight", rows: 2 * ff, columns: d)
        case .gated:
            try check(file, prefix + "ffn_gate.weight", rows: ff, columns: d)
            try check(file, prefix + "ffn_up.weight", rows: ff, columns: d)
        case .sequential:
            try check(file, prefix + "ffn_up.weight", rows: ff, columns: d)
        }
        try check(file, prefix + "ffn_down.weight", rows: d, columns: ff)
        if hasQKNorm {
            let width = qkNormIsFull ? q : shape.headSize
            try checkVector(file, prefix + "attn_q_norm.weight", width: width, orPerHead: q)
            if ownsCache { try checkVector(file, prefix + "attn_k_norm.weight", width: qkNormIsFull ? kv : shape.headSize, orPerHead: kv) }
        }
        if hasPostNorms {
            try check(file, prefix + "post_attention_norm.weight", rows: 1, columns: d)
            try check(file, prefix + "post_ffw_norm.weight", rows: 1, columns: d)
        }
        if perLayerInputSize > 0 {
            try check(file, prefix + "inp_gate.weight", rows: perLayerInputSize, columns: d)
            try check(file, prefix + "proj.weight", rows: d, columns: perLayerInputSize)
            try check(file, prefix + "post_norm.weight", rows: 1, columns: d)
        }
    }

    private func check(_ file: GGUFFile, _ name: String, rows: Int, columns: Int, optional: Bool = false) throws {
        guard let tensor = file.tensor(name) else {
            if optional { return }
            throw GGUFError.missingTensor(name)
        }
        let actualRows = tensor.dimensions.count >= 2 ? tensor.rows : 1
        let actualColumns = tensor.dimensions.count >= 2 ? tensor.columns : tensor.elementCount
        guard actualRows == rows, actualColumns == columns else {
            throw GGUFError.invalid("\(name) shape \(tensor.dimensions)")
        }
        guard let type = tensor.type, type.isSupported else {
            throw GGUFError.unsupportedTensorType(tensor.type?.name ?? "type \(tensor.typeCode)")
        }
    }

    /// 正規化の重み（width 個、またはヘッドごとに違う重みで perHead 個）
    private func checkVector(_ file: GGUFFile, _ name: String, width: Int, orPerHead perHead: Int) throws {
        guard let tensor = file.tensor(name) else { throw GGUFError.missingTensor(name) }
        guard tensor.elementCount == width || tensor.elementCount == perHead else {
            throw GGUFError.invalid("\(name) shape \(tensor.dimensions)")
        }
    }

    /// 実行に使う重みの名前（この順に変換して保存する）
    func tensorNames(in file: GGUFFile) -> [String] {
        var names = ["token_embd.weight"]
        let optional = ["output_norm.weight", "output_norm.bias", "output.weight", "output.bias", "rope_freqs.weight",
                        "position_embd.weight", "per_layer_token_embd.weight", "per_layer_model_proj.weight",
                        "per_layer_proj_norm.weight"]
        names.append(contentsOf: optional.filter { file.has($0) })
        let perLayer = [
            "attn_norm.weight", "attn_norm.bias", "attn_norm_2.weight", "attn_norm_2.bias",
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight",
            "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", "attn_output.weight", "attn_output.bias",
            "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight",
            "ffn_norm.weight", "ffn_norm.bias", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias", "post_ffw_norm.weight",
            "inp_gate.weight", "proj.weight", "post_norm.weight", "layer_output_scale.weight",
        ]
        for layer in 0..<layerCount {
            for suffix in perLayer where file.has("blk.\(layer).\(suffix)") { names.append("blk.\(layer).\(suffix)") }
        }
        return names
    }

    /// 行列でも、Float32 のベクトルとして置く重み（正規化の重み・層ごとの倍率）
    static func isVectorTensor(_ name: String) -> Bool {
        name.hasSuffix("norm.weight") || name.hasSuffix("norm.bias") || name.hasSuffix("layer_output_scale.weight")
    }

    /// 量子化したまま置く重み（行ごとに取り出すだけで、掛け算には使わない大きな表）
    static func isRawTensor(_ name: String) -> Bool {
        name == "per_layer_token_embd.weight"
    }
}
