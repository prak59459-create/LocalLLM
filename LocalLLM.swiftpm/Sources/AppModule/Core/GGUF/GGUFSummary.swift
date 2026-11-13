import Foundation

/// GGUF の先頭部分から分かる、画面に出す用の情報（ダウンロードの前に、先頭だけ取ってきて調べられる）。
struct GGUFSummary: Codable, Hashable, Sendable {
    var architecture: String
    var name: String?
    var parameterCount: Int
    /// いちばん多く使われている量子化形式（"Q4_K" など）
    var quantization: String
    /// 学習時のコンテキスト長
    var contextLength: Int
    var layerCount: Int
    var embeddingSize: Int
    var vocabSize: Int
    /// このアプリ用に変換したあとのファイルの大きさ（バイト）
    var storeBytes: Int64
    /// KV キャッシュの 1 トークンあたりのバイト数
    var kvBytesPerToken: Int
    var hasChatTemplate: Bool
    /// 動かせない理由（動かせるなら nil）
    var unsupportedReason: String?

    var isSupported: Bool { unsupportedReason == nil }

    init(file: GGUFFile) {
        architecture = file.architecture
        name = file.name
        parameterCount = file.parameterCount
        quantization = file.dominantQuantization
        hasChatTemplate = file.string("tokenizer.chat_template") != nil
        do {
            if (file.int("split.count") ?? 1) > 1 {
                throw GGUFError.invalid(L("何個かに分かれた GGUF（split）", "split GGUF"))
            }
            let config = try GGUFModelConfig(file: file)
            _ = try GGUFTokenizer.tokenizerConfigJSON(from: file)
            guard file.metadata["tokenizer.ggml.tokens"] != nil else { throw GGUFError.missingMetadata("tokenizer.ggml.tokens") }
            contextLength = config.contextLength
            layerCount = config.layerCount
            embeddingSize = config.embeddingSize
            vocabSize = config.vocabSize
            storeBytes = Int64(GGUFWeightStore.estimatedBytes(file: file, config: config, compact: true))
            kvBytesPerToken = config.kvBytesPerToken
            unsupportedReason = nil
        } catch {
            contextLength = file.archInt("context_length") ?? 0
            layerCount = file.archInt("block_count") ?? 0
            embeddingSize = file.archInt("embedding_length") ?? 0
            vocabSize = file.metadata["tokenizer.ggml.tokens"]?.count ?? 0
            storeBytes = Int64(file.parameterCount)
            kvBytesPerToken = 0
            unsupportedReason = error.localizedDescription
        }
    }

    /// 既定で使うコンテキスト長（KV キャッシュが大きくなりすぎないように、学習時の長さと 4096 の小さいほう）
    var defaultContext: Int { max(256, min(contextLength, 4096)) }

    /// パラメータ数の短い表記（"494M"・"1.5B"）
    var parameterLabel: String {
        let count = Double(parameterCount)
        if count >= 1e9 { return String(format: "%.1fB", count / 1e9) }
        return String(format: "%.0fM", count / 1e6)
    }

    /// 動かすのに要るメモリの目安（変換後の重み + コンテキスト分の KV キャッシュ）
    func memoryBytes(context: Int) -> Int64 {
        storeBytes + Int64(kvBytesPerToken) * Int64(context) + 64 << 20
    }
}

/// GGUF のファイル名から分かること。
enum GGUFNaming {
    static func isGGUF(_ path: String) -> Bool {
        path.lowercased().hasSuffix(".gguf")
    }

    /// 画像用の部品（mmproj）や、何個かに分かれたファイルの 1 つは、単体では動かせない
    static func isStandaloneModel(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent.lowercased()
        guard name.hasSuffix(".gguf") else { return false }
        if name.contains("mmproj") || name.contains("imatrix") { return false }
        return name.range(of: #"-\d{5}-of-\d{5}\.gguf$"#, options: .regularExpression) == nil
    }

    /// ファイル名に入っている量子化の名前（"Q4_K_M"・"IQ3_XS"・"F16" など）
    static func quantization(of path: String) -> String? {
        let name = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        let pattern = #"(?i)(?:^|[-._])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|F16|F32|BF16|FP16|FP32|TQ\d_\d)(?:$|[-._])"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.matches(in: name, range: NSRange(name.startIndex..., in: name)).last,
              let range = Range(match.range(at: 1), in: name) else { return nil }
        return name[range].uppercased()
    }

    /// 量子化の名前から、このアプリで読める形式か（名前で分かる範囲で）
    static func isReadable(quantization: String?) -> Bool {
        guard let quantization else { return true }
        return !(quantization.hasPrefix("IQ1") || quantization.hasPrefix("TQ"))
    }

    /// 並べる順（小さいほど先）: まず定番の Q4_K_M、そのあとは軽い順
    static func rank(of path: String) -> Int {
        let order = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_L", "Q3_K_S", "IQ4_XS",
                     "IQ3_M", "Q2_K", "F16", "BF16", "F32"]
        guard let quantization = quantization(of: path) else { return order.count }
        if let index = order.firstIndex(of: quantization) { return index }
        return order.firstIndex { quantization.hasPrefix($0) } ?? order.count
    }
}
