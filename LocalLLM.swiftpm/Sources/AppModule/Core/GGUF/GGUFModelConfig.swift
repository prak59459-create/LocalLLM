import Foundation

/// GGUF のメタデータから読み取った、モデルの構造（llama.cpp の hparams に当たる）。
struct GGUFModelConfig: Codable, Equatable, Sendable {
    /// 計算の流れが同じものをまとめた系列
    enum Family: String, Codable, Sendable {
        case llama      // Llama・Mistral・SmolLM・TinyLlama など
        case qwen2      // Qwen2 / 2.5・DeepSeek R1 Distill（Qwen）など
        case qwen3
        case gemma
        case gemma2
        case gemma3
        case phi3
    }

    enum Activation: String, Codable, Sendable { case silu, gelu }

    var architecture: String
    var family: Family
    var vocabSize: Int
    var layerCount: Int
    var embeddingSize: Int
    var feedForwardSize: Int
    var headCount: Int
    var kvHeadCount: Int
    var headSize: Int
    var contextLength: Int
    var normEpsilon: Float

    /// RoPE: 回転させる次元数・周波数の底・位置に掛ける倍率
    var ropeDimensions: Int
    var ropeBase: Float
    var ropeScale: Float
    /// true: 前半と後半を組にする（NEOX）。false: 隣どうしを組にする
    var ropePairsAreSplit: Bool

    /// 埋め込みに掛ける倍率（Gemma は √d）
    var embeddingScale: Float
    /// 注意のスコアに掛ける倍率（ふつうは 1/√headSize）
    var attentionScale: Float
    /// 注意のスコアを tanh で抑える上限（0 なら使わない。Gemma 2）
    var attentionSoftcap: Float
    /// 最後の logits を tanh で抑える上限（0 なら使わない）
    var finalSoftcap: Float
    var activation: Activation
    /// 注意・FFN の出力に正規化を掛ける（Gemma 2 / 3）
    var hasPostNorms: Bool
    /// ヘッドごとの q / k に正規化を掛ける（Qwen3・Gemma 3）
    var hasQKNorm: Bool
    /// q・k・v が 1 つの行列にまとまっている、FFN の gate と up が 1 つにまとまっている（Phi-3）
    var hasFusedQKV: Bool
    var hasFusedGateUp: Bool
    /// 出力層が埋め込みと同じ重み
    var tiedOutput: Bool

    /// 直近だけを見る層（スライディングウィンドウ）の窓の大きさと、その層・そのときの RoPE の底
    var slidingWindow: Int
    var slidingLayers: [Bool]
    var slidingRopeBase: Float

    var name: String?
    var quantization: String?

    var kvSize: Int { kvHeadCount * headSize }
    var querySize: Int { headCount * headSize }

    /// 1 トークンあたりの KV キャッシュのバイト数
    var kvBytesPerToken: Int { layerCount * kvSize * 2 * MemoryLayout<Float>.size }

    func isSliding(_ layer: Int) -> Bool {
        slidingWindow > 0 && layer < slidingLayers.count && slidingLayers[layer]
    }

    func ropeBase(layer: Int) -> Float {
        isSliding(layer) ? slidingRopeBase : ropeBase
    }

    static let supportedArchitectures: [String: Family] = [
        "llama": .llama, "qwen2": .qwen2, "qwen3": .qwen3,
        "gemma": .gemma, "gemma2": .gemma2, "gemma3": .gemma3, "phi3": .phi3,
    ]

    /// メタデータと重みの一覧から構造を決める。対応していない構造なら GGUFError を投げる。
    init(file: GGUFFile) throws {
        let arch = file.architecture
        guard let family = Self.supportedArchitectures[arch] else {
            throw GGUFError.unsupportedArchitecture(arch.isEmpty ? "?" : arch)
        }
        func require(_ suffix: String) throws -> Int {
            guard let value = file.archInt(suffix) else { throw GGUFError.missingMetadata("\(arch).\(suffix)") }
            return value
        }
        guard let embedding = file.tensor("token_embd.weight"), embedding.dimensions.count == 2 else {
            throw GGUFError.missingTensor("token_embd.weight")
        }
        // 混合エキスパート（MoE）は扱わない
        if file.has("blk.0.ffn_gate_inp.weight") || (file.archInt("expert_count") ?? 0) > 0 {
            throw GGUFError.unsupportedArchitecture("\(arch) (MoE)")
        }
        architecture = arch
        self.family = family
        name = file.name
        quantization = file.dominantQuantization
        layerCount = try require("block_count")
        embeddingSize = try require("embedding_length")
        guard embedding.columns == embeddingSize else { throw GGUFError.invalid("token_embd shape") }
        vocabSize = embedding.rows
        headCount = try require("attention.head_count")
        kvHeadCount = file.archInt("attention.head_count_kv") ?? headCount
        guard layerCount > 0, headCount > 0, kvHeadCount > 0, headCount % kvHeadCount == 0 else {
            throw GGUFError.invalid("head counts")
        }
        headSize = file.archInt("attention.key_length") ?? (embeddingSize / headCount)
        if let valueLength = file.archInt("attention.value_length"), valueLength != headSize {
            throw GGUFError.unsupportedArchitecture("\(arch) (key_length != value_length)")
        }
        hasFusedQKV = file.has("blk.0.attn_qkv.weight")
        hasFusedGateUp = !file.has("blk.0.ffn_gate.weight")
        if let down = file.tensor("blk.0.ffn_down.weight") {
            feedForwardSize = down.columns
        } else {
            feedForwardSize = try require("feed_forward_length")
        }
        contextLength = max(8, file.archInt("context_length") ?? 2048)
        normEpsilon = Float(file.archDouble("attention.layer_norm_rms_epsilon") ?? 1e-5)

        ropeDimensions = file.archInt("rope.dimension_count") ?? headSize
        guard ropeDimensions % 2 == 0, ropeDimensions <= headSize else { throw GGUFError.invalid("rope dimensions") }
        ropeBase = Float(file.archDouble("rope.freq_base") ?? 10000)
        let scaling = file.archDouble("rope.scaling.factor") ?? file.archDouble("rope.scale_linear") ?? 0
        ropeScale = scaling == 0 ? 1 : Float(1 / scaling)
        // 学習時より長い文脈に伸ばす仕組み（YaRN など）は使わないので、学習時の長さまでにする
        if let original = file.archInt("rope.scaling.original_context_length"), original > 0 {
            contextLength = min(contextLength, original)
        }
        ropePairsAreSplit = family != .llama

        let plainScale = 1 / Float(headSize).squareRoot()
        attentionScale = plainScale
        embeddingScale = 1
        attentionSoftcap = 0
        finalSoftcap = 0
        activation = .silu
        hasPostNorms = false
        hasQKNorm = false
        slidingWindow = 0
        slidingLayers = []
        slidingRopeBase = ropeBase
        tiedOutput = !file.has("output.weight")

        switch family {
        case .llama:
            if let scale = file.archDouble("attention.scale"), scale != 0 { attentionScale = Float(scale) }
        case .qwen2, .phi3:
            break
        case .qwen3:
            hasQKNorm = true
        case .gemma:
            embeddingScale = Float(embeddingSize).squareRoot()
            activation = .gelu
        case .gemma2:
            embeddingScale = Float(embeddingSize).squareRoot()
            activation = .gelu
            hasPostNorms = true
            attentionSoftcap = Float(file.archDouble("attn_logit_softcapping") ?? 50)
            finalSoftcap = Float(file.archDouble("final_logit_softcapping") ?? 30)
            slidingWindow = file.archInt("attention.sliding_window") ?? 4096
            slidingLayers = Self.slidingPattern(file: file, layers: layerCount, defaultPeriod: 2)
            slidingRopeBase = Float(file.archDouble("rope.freq_base_swa") ?? Double(ropeBase))
        case .gemma3:
            embeddingScale = Float(embeddingSize).squareRoot()
            activation = .gelu
            hasPostNorms = true
            hasQKNorm = true
            finalSoftcap = Float(file.archDouble("final_logit_softcapping") ?? 0)
            let window = file.archInt("attention.sliding_window") ?? 0
            if window > 0 {
                slidingWindow = window
                slidingLayers = Self.slidingPattern(file: file, layers: layerCount, defaultPeriod: 6)
                slidingRopeBase = Float(file.archDouble("rope.freq_base_swa") ?? 10000)
            }
        }
        // Gemma 2 / 3 の 27B は倍率が違うが、iPad では動かせない大きさなので考えない
        try validateTensors(file)
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

    /// 必要な重みがそろっていて、形が合っているかを確かめる
    private func validateTensors(_ file: GGUFFile) throws {
        func check(_ name: String, rows: Int, columns: Int) throws {
            guard let tensor = file.tensor(name) else { throw GGUFError.missingTensor(name) }
            guard tensor.rows == rows, tensor.columns == columns else {
                throw GGUFError.invalid("\(name) shape \(tensor.dimensions)")
            }
            guard let type = tensor.type, type.isSupported else {
                throw GGUFError.unsupportedTensorType(tensor.type?.name ?? "type \(tensor.typeCode)")
            }
        }
        try check("output_norm.weight", rows: 1, columns: embeddingSize)
        if !tiedOutput { try check("output.weight", rows: vocabSize, columns: embeddingSize) }
        for layer in 0..<layerCount {
            let prefix = "blk.\(layer)."
            try check(prefix + "attn_norm.weight", rows: 1, columns: embeddingSize)
            if hasFusedQKV {
                try check(prefix + "attn_qkv.weight", rows: querySize + 2 * kvSize, columns: embeddingSize)
            } else {
                try check(prefix + "attn_q.weight", rows: querySize, columns: embeddingSize)
                try check(prefix + "attn_k.weight", rows: kvSize, columns: embeddingSize)
                try check(prefix + "attn_v.weight", rows: kvSize, columns: embeddingSize)
            }
            try check(prefix + "attn_output.weight", rows: embeddingSize, columns: querySize)
            try check(prefix + "ffn_norm.weight", rows: 1, columns: embeddingSize)
            if hasFusedGateUp {
                try check(prefix + "ffn_up.weight", rows: 2 * feedForwardSize, columns: embeddingSize)
            } else {
                try check(prefix + "ffn_gate.weight", rows: feedForwardSize, columns: embeddingSize)
                try check(prefix + "ffn_up.weight", rows: feedForwardSize, columns: embeddingSize)
            }
            try check(prefix + "ffn_down.weight", rows: embeddingSize, columns: feedForwardSize)
            if hasQKNorm {
                try check(prefix + "attn_q_norm.weight", rows: 1, columns: headSize)
                try check(prefix + "attn_k_norm.weight", rows: 1, columns: headSize)
            }
            if hasPostNorms {
                try check(prefix + "post_attention_norm.weight", rows: 1, columns: embeddingSize)
                try check(prefix + "post_ffw_norm.weight", rows: 1, columns: embeddingSize)
            }
        }
    }

    /// 実行に使う重みの名前（この順に変換して保存する）
    func tensorNames(in file: GGUFFile) -> [String] {
        var names = ["token_embd.weight", "output_norm.weight"]
        let optional = ["output.weight", "output.bias", "rope_freqs.weight"]
        names.append(contentsOf: optional.filter { file.has($0) })
        let perLayer = [
            "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight",
            "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", "attn_output.weight", "attn_output.bias",
            "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight",
            "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias", "post_ffw_norm.weight",
        ]
        for layer in 0..<layerCount {
            for suffix in perLayer where file.has("blk.\(layer).\(suffix)") { names.append("blk.\(layer).\(suffix)") }
        }
        return names
    }
}
