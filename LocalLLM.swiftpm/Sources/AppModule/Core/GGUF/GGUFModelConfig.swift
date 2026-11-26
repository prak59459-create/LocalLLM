import Foundation

/// GGUF のメタデータから読み取った、モデルの構造（llama.cpp の hparams に当たる）。
///
/// 計算する側（GGUFModel・GGUFMetalModel）は、系列の名前ではなく、ここにある値と印だけを見て計算する。
/// 新しく足した値は、前の版で変換したファイルにも無いので、読むときに既定の値を入れる（init(from:)）。
struct GGUFModelConfig: Codable, Equatable, Sendable {
    /// 系列（llama.cpp の architecture）。読み取りの規則を決めるのに使う
    enum Family: String, Codable, Sendable, CaseIterable {
        case llama, qwen2, qwen3, gemma, gemma2, gemma3, phi3
        case gemma4, mistral3, internlm2, xverse, baichuan, ernie45 = "ernie4_5", granite, minicpm, arcee, smollm3
        case maincoder, qwen, qwen2vl, qwen3vl, exaone, exaone4, olmo, olmo2, seedOSS = "seed_oss", plamo, openelm
        case phi2, gpt2, starcoder, starcoder2, stablelm, gptneox, falcon, codeshell, orion, nemotron, jais2
        case commandR = "command-r", cohere2, chatglm, glm4, lfm2, qwen35
        case qwen2moe, qwen3moe, olmoe, granitemoe
        /// Gemma 4 の MTP（先読み）用の小さなモデル。本体の隠れ状態と KV キャッシュを使う
        case gemma4Assistant = "gemma4-assistant"
    }

    enum Activation: String, Codable, Sendable { case silu, gelu, reluSquared }
    /// 正規化の種類: RMSNorm か LayerNorm（平均も引く）
    enum Norm: String, Codable, Sendable { case rms, layer }
    /// FFN の形: gated = act(gate) × up → down、sequential = act(up) → down
    enum FeedForward: String, Codable, Sendable { case gated, sequential }

    /// 1 層ぶんの形（層ごとに違うモデルがある: Gemma 4・OpenELM など）
    struct Layer: Codable, Equatable, Sendable {
        var headCount: Int
        var kvHeadCount: Int
        var headSize: Int
        var feedForwardSize: Int
        /// 位置の回転（RoPE）をかけるか
        var usesRope: Bool
        /// 直近だけを見る層（スライディングウィンドウ）か
        var sliding: Bool
        /// KV キャッシュを持つ層の番号（自分の番号なら自分で持つ。違えば、その層のキャッシュを使う: Gemma 4）
        var kvLayer: Int
        /// 注意の代わりに、短い畳み込みを使う層（LFM2）
        var convolution = false
        /// 注意の代わりに、Gated DeltaNet（状態を持つ線形の注意）を使う層（Qwen3.5）
        var deltaNet = false
        /// FFN が専門家の混合（MoE）の層（feedForwardSize は専門家 1 つの大きさ）
        var moe = false

        var querySize: Int { headCount * headSize }
        var kvSize: Int { kvHeadCount * headSize }

        init(headCount: Int, kvHeadCount: Int, headSize: Int, feedForwardSize: Int, usesRope: Bool, sliding: Bool, kvLayer: Int,
             convolution: Bool = false, deltaNet: Bool = false, moe: Bool = false) {
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.headSize = headSize
            self.feedForwardSize = feedForwardSize
            self.usesRope = usesRope
            self.sliding = sliding
            self.kvLayer = kvLayer
            self.convolution = convolution
            self.deltaNet = deltaNet
            self.moe = moe
        }

        /// 位置ごとの KV キャッシュを使わない層（畳み込み・DeltaNet）
        var isRecurrent: Bool { convolution || deltaNet }

        enum CodingKeys: String, CodingKey {
            case headCount, kvHeadCount, headSize, feedForwardSize, usesRope, sliding, kvLayer, convolution, deltaNet, moe
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            headCount = try c.decode(Int.self, forKey: .headCount)
            kvHeadCount = try c.decode(Int.self, forKey: .kvHeadCount)
            headSize = try c.decode(Int.self, forKey: .headSize)
            feedForwardSize = try c.decode(Int.self, forKey: .feedForwardSize)
            usesRope = try c.decode(Bool.self, forKey: .usesRope)
            sliding = try c.decode(Bool.self, forKey: .sliding)
            kvLayer = try c.decode(Int.self, forKey: .kvLayer)
            convolution = try c.decodeIfPresent(Bool.self, forKey: .convolution) ?? false
            deltaNet = try c.decodeIfPresent(Bool.self, forKey: .deltaNet) ?? false
            moe = try c.decodeIfPresent(Bool.self, forKey: .moe) ?? false
        }
    }

    /// 専門家の混合（MoE）の大きさと選び方
    struct MoE: Codable, Equatable, Sendable {
        var expertCount: Int
        /// 1 トークンで使う専門家の数（モデルの既定）
        var expertsUsed: Int
        /// 専門家 1 つの FFN の大きさ
        var expertFeedForward: Int
        /// 共通の専門家（いつも使う）の FFN の大きさ（無ければ 0）
        var sharedFeedForward: Int
        /// 共通の専門家の出力に sigmoid(w · x) を掛ける（Qwen2-MoE）
        var sharedGated: Bool
        /// 選んだ専門家の重みを、合計が 1 になるように割る
        var normalizeWeights: Bool
        /// 専門家の重みに掛ける倍率（0 なら掛けない）
        var weightScale: Float
    }

    /// Gated DeltaNet の大きさ（Qwen3.5）。キー・バリューのヘッドごとに、キーの大きさ × バリューの大きさの状態を持つ
    struct DeltaNet: Codable, Equatable, Sendable {
        /// 入力の短い畳み込みの長さ
        var convolutionLength: Int
        var keyHeads: Int
        var keyHeadSize: Int
        var valueHeads: Int
        var valueHeadSize: Int

        var keySize: Int { keyHeads * keyHeadSize }
        var valueSize: Int { valueHeads * valueHeadSize }
        /// 畳み込みのチャンネル数（q・k・v）
        var channels: Int { 2 * keySize + valueSize }
        /// 1 層の状態の Float の数
        var stateFloats: Int { valueHeads * valueHeadSize * keyHeadSize }
    }

    /// YaRN（学習時より長い文脈に伸ばす仕組み）の値
    struct YaRN: Codable, Equatable, Sendable {
        var originalContext: Int
        var betaFast: Float
        var betaSlow: Float
        /// cos / sin に掛ける倍率
        var magnitude: Float
    }

    var architecture: String
    var family: Family
    var vocabSize: Int
    var layerCount: Int
    var embeddingSize: Int
    /// 以下の 3 つは、層ごとの形（layers）が無い古いファイル向けの値。新しいファイルでは最大の値
    var feedForwardSize: Int
    var headCount: Int
    var kvHeadCount: Int
    var headSize: Int
    var contextLength: Int
    var normEpsilon: Float

    /// RoPE: 回転させる次元数・周波数の底・位置に掛ける倍率（直近だけを見る層は別の値のことがある）
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
    /// 注意・FFN の出力に正規化を掛ける（Gemma 2 / 3 / 4・OLMo 2 など）
    var hasPostNorms: Bool
    /// ヘッドごとの q / k に正規化を掛ける（Qwen3・Gemma 3 など）
    var hasQKNorm: Bool
    /// q・k・v が 1 つの行列にまとまっている、FFN の gate と up が 1 つにまとまっている（Phi-3 など）
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

    // MARK: 版 3 で足したもの（無ければ既定の値）

    /// 層ごとの形（空なら上の値から作る）
    var layerShapes: [Layer]
    var norm: Norm
    var feedForward: FeedForward
    /// 注意と FFN を同じ入力から計算して、両方を足す（Phi-2・Falcon・Command R など）
    var parallelResidual: Bool
    /// 注意・FFN の前に正規化を掛ける（OLMo 2・EXAONE 4 は掛けない）
    var hasPreNorms: Bool
    /// q / k の正規化を、ヘッドごとではなく全体に掛ける（OLMo 2）
    var qkNormIsFull: Bool
    /// v に（重みの無い）RMSNorm を掛ける（Gemma 4）
    var hasValueNorm: Bool
    /// 学習した位置の埋め込みを足す（GPT-2・StarCoder）
    var learnedPositions: Bool
    /// 注意・FFN の出力に掛ける倍率（Granite・MiniCPM）
    var residualScale: Float
    /// 最後の logits に掛ける倍率
    var logitScale: Float
    /// 層ごとの入力埋め込みの大きさ（0 なら無い。Gemma 4 の E2B / E4B）
    var perLayerInputSize: Int
    /// 先読み用のモデル（gemma4-assistant）なら、本体の隠れ状態の大きさ（それ以外は 0）
    var drafterOutputSize = 0
    /// 直近だけを見る層の RoPE の次元数
    var slidingRopeDimensions: Int
    /// rope_freqs（周波数の割り算）を、直近だけを見る層にも使うか（Gemma 4 は全体を見る層だけ）
    var ropeFactorsOnSlidingLayers: Bool
    /// 直近だけを見る層の、位置に掛ける倍率（Gemma 3 / 4 は伸ばさない）
    var slidingRopeScale: Float
    /// q / k の正規化を、RoPE の後に掛ける（Maincoder）
    var qkNormAfterRope: Bool
    /// 短い畳み込みの長さ（LFM2。0 なら使わない）
    var convolutionLength: Int
    var yarn: YaRN?

    // MARK: 版 4 で足したもの（無ければ既定の値）

    /// Gated DeltaNet の層の大きさ（Qwen3.5。無ければ nil）
    var deltaNet: DeltaNet?
    /// 注意の q の射影に、ヘッドごとの出力のゲートが続いている（q とゲートが交互）。出力 × sigmoid(ゲート)（Qwen3.5）
    var attentionOutputGate: Bool
    /// 専門家の混合（MoE。無ければ nil）
    var moe: MoE?

    var kvSize: Int { kvHeadCount * headSize }
    var querySize: Int { headCount * headSize }

    /// 層ごとの形
    var layers: [Layer] {
        if layerShapes.count == layerCount { return layerShapes }
        return (0..<layerCount).map { index in
            Layer(headCount: headCount, kvHeadCount: kvHeadCount, headSize: headSize, feedForwardSize: feedForwardSize,
                  usesRope: true, sliding: isSliding(index), kvLayer: index)
        }
    }

    /// どの層よりも大きい q・kv・FFN・ヘッドの大きさ（作業用のメモリの大きさに使う）
    var maxQuerySize: Int { layers.map(\.querySize).max() ?? querySize }
    var maxKVSize: Int { layers.map(\.kvSize).max() ?? kvSize }
    var maxFeedForwardSize: Int { layers.map(\.feedForwardSize).max() ?? feedForwardSize }
    var maxHeadSize: Int { layers.map(\.headSize).max() ?? headSize }
    var maxHeadCount: Int { layers.map(\.headCount).max() ?? headCount }

    /// 1 トークンあたりの KV キャッシュのバイト数（キャッシュを持つ層だけ）
    var kvBytesPerToken: Int {
        layers.enumerated().reduce(0) { total, item in
            // 畳み込みの層は、位置ごとに 1 つ（埋め込みの大きさ）を覚えておく
            if item.element.convolution { return total + embeddingSize * MemoryLayout<Float>.size }
            // DeltaNet の層は、位置によらず決まった大きさの状態だけ
            if item.element.deltaNet { return total }
            return item.element.kvLayer == item.offset ? total + item.element.kvSize * 2 * MemoryLayout<Float>.size : total
        }
    }

    /// 注意を使う層の、いちばん大きいヘッドの数
    var maxAttentionHeads: Int { layers.filter { !$0.isRecurrent }.map(\.headCount).max() ?? headCount }

    /// 位置によらない状態（DeltaNet の状態と畳み込みの直前の入力）のバイト数。スナップショットを 1 つ持つので 2 倍
    var recurrentStateBytes: Int {
        guard let deltaNet else { return 0 }
        let perLayer = deltaNet.stateFloats + (deltaNet.convolutionLength - 1) * deltaNet.channels
        return 2 * layers.filter(\.deltaNet).count * perLayer * MemoryLayout<Float>.size
    }

    /// 巻き戻せない状態を持つ（巻き戻すときは、覚えておいた状態から読み直す）
    var hasRecurrentState: Bool { deltaNet != nil }

    func isSliding(_ layer: Int) -> Bool {
        slidingWindow > 0 && layer < slidingLayers.count && slidingLayers[layer]
    }

    func ropeBase(layer: Int) -> Float {
        isSliding(layer) ? slidingRopeBase : ropeBase
    }

    /// その層の RoPE の次元数（0 なら回転しない）
    func ropeDimensions(layer: Int) -> Int {
        let shape = layers[layer]
        guard shape.usesRope else { return 0 }
        return shape.sliding ? slidingRopeDimensions : ropeDimensions
    }

    static let supportedArchitectures: [String: Family] = {
        var map: [String: Family] = [:]
        for family in Family.allCases { map[family.rawValue] = family }
        map["seed_oss"] = .seedOSS
        return map
    }()

    /// 対応している構造の名前（説明用）
    static var supportedArchitectureNames: [String] { supportedArchitectures.keys.sorted() }

    // MARK: - 古いファイルとの互換

    enum CodingKeys: String, CodingKey {
        case architecture, family, vocabSize, layerCount, embeddingSize, feedForwardSize, headCount, kvHeadCount, headSize
        case contextLength, normEpsilon, ropeDimensions, ropeBase, ropeScale, ropePairsAreSplit, embeddingScale
        case attentionScale, attentionSoftcap, finalSoftcap, activation, hasPostNorms, hasQKNorm, hasFusedQKV
        case hasFusedGateUp, tiedOutput, slidingWindow, slidingLayers, slidingRopeBase, name, quantization
        case layerShapes, norm, feedForward, parallelResidual, hasPreNorms, qkNormIsFull, hasValueNorm
        case learnedPositions, residualScale, logitScale, perLayerInputSize, slidingRopeDimensions
        case ropeFactorsOnSlidingLayers, slidingRopeScale, qkNormAfterRope, convolutionLength, yarn
        case deltaNet, attentionOutputGate, moe, drafterOutputSize
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architecture = try c.decode(String.self, forKey: .architecture)
        family = try c.decode(Family.self, forKey: .family)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        layerCount = try c.decode(Int.self, forKey: .layerCount)
        embeddingSize = try c.decode(Int.self, forKey: .embeddingSize)
        feedForwardSize = try c.decode(Int.self, forKey: .feedForwardSize)
        headCount = try c.decode(Int.self, forKey: .headCount)
        kvHeadCount = try c.decode(Int.self, forKey: .kvHeadCount)
        headSize = try c.decode(Int.self, forKey: .headSize)
        contextLength = try c.decode(Int.self, forKey: .contextLength)
        normEpsilon = try c.decode(Float.self, forKey: .normEpsilon)
        ropeDimensions = try c.decode(Int.self, forKey: .ropeDimensions)
        ropeBase = try c.decode(Float.self, forKey: .ropeBase)
        ropeScale = try c.decode(Float.self, forKey: .ropeScale)
        ropePairsAreSplit = try c.decode(Bool.self, forKey: .ropePairsAreSplit)
        embeddingScale = try c.decode(Float.self, forKey: .embeddingScale)
        attentionScale = try c.decode(Float.self, forKey: .attentionScale)
        attentionSoftcap = try c.decode(Float.self, forKey: .attentionSoftcap)
        finalSoftcap = try c.decode(Float.self, forKey: .finalSoftcap)
        activation = try c.decode(Activation.self, forKey: .activation)
        hasPostNorms = try c.decode(Bool.self, forKey: .hasPostNorms)
        hasQKNorm = try c.decode(Bool.self, forKey: .hasQKNorm)
        hasFusedQKV = try c.decode(Bool.self, forKey: .hasFusedQKV)
        hasFusedGateUp = try c.decode(Bool.self, forKey: .hasFusedGateUp)
        tiedOutput = try c.decode(Bool.self, forKey: .tiedOutput)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        slidingLayers = try c.decode([Bool].self, forKey: .slidingLayers)
        slidingRopeBase = try c.decode(Float.self, forKey: .slidingRopeBase)
        name = try c.decodeIfPresent(String.self, forKey: .name)
        quantization = try c.decodeIfPresent(String.self, forKey: .quantization)
        layerShapes = try c.decodeIfPresent([Layer].self, forKey: .layerShapes) ?? []
        norm = try c.decodeIfPresent(Norm.self, forKey: .norm) ?? .rms
        feedForward = try c.decodeIfPresent(FeedForward.self, forKey: .feedForward) ?? .gated
        parallelResidual = try c.decodeIfPresent(Bool.self, forKey: .parallelResidual) ?? false
        hasPreNorms = try c.decodeIfPresent(Bool.self, forKey: .hasPreNorms) ?? true
        qkNormIsFull = try c.decodeIfPresent(Bool.self, forKey: .qkNormIsFull) ?? false
        hasValueNorm = try c.decodeIfPresent(Bool.self, forKey: .hasValueNorm) ?? false
        learnedPositions = try c.decodeIfPresent(Bool.self, forKey: .learnedPositions) ?? false
        residualScale = try c.decodeIfPresent(Float.self, forKey: .residualScale) ?? 1
        logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale) ?? 1
        perLayerInputSize = try c.decodeIfPresent(Int.self, forKey: .perLayerInputSize) ?? 0
        slidingRopeDimensions = try c.decodeIfPresent(Int.self, forKey: .slidingRopeDimensions) ?? ropeDimensions
        ropeFactorsOnSlidingLayers = try c.decodeIfPresent(Bool.self, forKey: .ropeFactorsOnSlidingLayers) ?? true
        slidingRopeScale = try c.decodeIfPresent(Float.self, forKey: .slidingRopeScale) ?? (family == .gemma3 ? 1 : ropeScale)
        qkNormAfterRope = try c.decodeIfPresent(Bool.self, forKey: .qkNormAfterRope) ?? false
        convolutionLength = try c.decodeIfPresent(Int.self, forKey: .convolutionLength) ?? 0
        yarn = try c.decodeIfPresent(YaRN.self, forKey: .yarn)
        deltaNet = try c.decodeIfPresent(DeltaNet.self, forKey: .deltaNet)
        attentionOutputGate = try c.decodeIfPresent(Bool.self, forKey: .attentionOutputGate) ?? false
        moe = try c.decodeIfPresent(MoE.self, forKey: .moe)
        drafterOutputSize = try c.decodeIfPresent(Int.self, forKey: .drafterOutputSize) ?? 0
    }
}
