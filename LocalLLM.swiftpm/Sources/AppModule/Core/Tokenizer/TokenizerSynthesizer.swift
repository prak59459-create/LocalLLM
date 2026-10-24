import Foundation

/// tokenizer.json が無いモデルのために、ほかの形式のトークナイザのファイルから同じ内容の tokenizer.json を作る。
///
/// - tokenizer.model（SentencePiece の BPE。Llama 2・Mistral・Gemma など）
/// - vocab.json + merges.txt（GPT-2 形式のバイト単位の BPE。Qwen など）
/// - *.tiktoken / tiktoken 形式の tokenizer.model（Llama 3 の original など）
///
/// どれも Hugging Face の `transformers` が「遅いトークナイザ」から tokenizer.json に変換するときと同じ規則で作る。
enum TokenizerSynthesizer {
    enum Source {
        case sentencePiece(Data)
        case vocabMerges(vocab: Data, merges: String)
        case tiktoken(String)
    }

    /// 見つけたトークナイザのファイル
    struct Files: Sendable {
        var tokenizerJSON: URL?
        var tokenizerModel: URL?
        var vocabJSON: URL?
        var mergesText: URL?
        var tiktoken: URL?
        var addedTokensJSON: URL?

        var canBuild: Bool { tokenizerJSON != nil || tokenizerModel != nil || (vocabJSON != nil && mergesText != nil) || tiktoken != nil }

        /// tokenizer.json を作るのに使えるファイルがあるか
        var canSynthesize: Bool { tokenizerModel != nil || (vocabJSON != nil && mergesText != nil) || tiktoken != nil }
    }

    /// tokenizer.json の代わりになるファイルの名前
    static let alternativeFileNames: Set<String> = [
        "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.json", "merges.txt", "added_tokens.json",
        "qwen.tiktoken", "tokenizer.tiktoken", "cl100k_base.tiktoken", "o200k_base.tiktoken",
    ]

    /// ファイル名の一覧から、トークナイザを作れるか
    static func canBuild(fileNames: Set<String>) -> Bool {
        fileNames.contains("tokenizer.json") || fileNames.contains("tokenizer.model") || fileNames.contains("spiece.model")
            || fileNames.contains("sentencepiece.bpe.model") || (fileNames.contains("vocab.json") && fileNames.contains("merges.txt"))
            || fileNames.contains { $0.hasSuffix(".tiktoken") }
    }

    /// ファイルの中身から、どの形式かを決める
    static func source(for files: Files) throws -> Source? {
        if let url = files.tiktoken { return .tiktoken(try String(contentsOf: url, encoding: .utf8)) }
        if let url = files.tokenizerModel {
            let data = try Data(contentsOf: url)
            // tiktoken 形式（"base64 rank" の行）でも tokenizer.model という名前のことがある
            if isTiktokenText(data) { return .tiktoken(String(decoding: data, as: UTF8.self)) }
            return .sentencePiece(data)
        }
        if let vocab = files.vocabJSON, let merges = files.mergesText {
            return .vocabMerges(vocab: try Data(contentsOf: vocab), merges: try String(contentsOf: merges, encoding: .utf8))
        }
        return nil
    }

    static func isTiktokenText(_ data: Data) -> Bool {
        let head = data.prefix(200)
        guard let text = String(data: head, encoding: .utf8), let line = text.split(separator: "\n").first else { return false }
        let parts = line.split(separator: " ")
        return parts.count == 2 && Int(parts[1]) != nil && Data(base64Encoded: String(parts[0])) != nil
    }

    /// tokenizer.json の中身を作る
    static func tokenizerJSON(from source: Source, tokenizerConfig: Data? = nil, addedTokensJSON: Data? = nil) throws -> Data {
        let config = tokenizerConfig.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        var root: [String: Any]
        switch source {
        case .sentencePiece(let data):
            root = try sentencePiece(data, config: config)
        case .vocabMerges(let vocab, let merges):
            root = try byteLevelBPE(vocabJSON: vocab, merges: merges, config: config)
        case .tiktoken(let text):
            root = try tiktoken(text, config: config)
        }
        // tokenizer_config.json と added_tokens.json の追加トークンを足す（語彙にあるものは ID を合わせる）
        var added = root["added_tokens"] as? [[String: Any]] ?? []
        var known = Set(added.compactMap { $0["content"] as? String })
        for token in addedTokens(config: config, addedTokensJSON: addedTokensJSON) where !known.contains(token.content) {
            known.insert(token.content)
            added.append(token.json)
        }
        root["added_tokens"] = added.sorted { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        return try JSONSerialization.data(withJSONObject: root)
    }

    // MARK: - 追加トークン

    struct AddedToken {
        var id: Int
        var content: String
        var special: Bool
        var lstrip = false
        var rstrip = false
        var normalized = false

        var json: [String: Any] {
            ["id": id, "content": content, "single_word": false, "lstrip": lstrip, "rstrip": rstrip,
             "normalized": normalized, "special": special]
        }
    }

    static func addedTokens(config: [String: Any], addedTokensJSON: Data?) -> [AddedToken] {
        var result: [Int: AddedToken] = [:]
        if let decoder = config["added_tokens_decoder"] as? [String: Any] {
            for (key, value) in decoder {
                guard let id = Int(key), let entry = value as? [String: Any], let content = entry["content"] as? String else { continue }
                result[id] = AddedToken(id: id, content: content, special: entry["special"] as? Bool ?? true,
                                        lstrip: entry["lstrip"] as? Bool ?? false, rstrip: entry["rstrip"] as? Bool ?? false,
                                        normalized: entry["normalized"] as? Bool ?? false)
            }
        }
        if let data = addedTokensJSON, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (content, value) in object {
                guard let id = (value as? Int) ?? (value as? NSNumber)?.intValue, result[id] == nil else { continue }
                let special = content.hasPrefix("<") && content.hasSuffix(">")
                result[id] = AddedToken(id: id, content: content, special: special)
            }
        }
        return result.values.sorted { $0.id < $1.id }
    }

    // MARK: - SentencePiece

    /// SentencePiece の 1 つのピース
    struct Piece {
        enum Kind: Int { case normal = 1, unknown, control, userDefined, unused, byte }
        var text: String
        var score: Float
        var kind: Kind
    }

    struct SentencePieceModel {
        var pieces: [Piece]
        /// 1 = Unigram, 2 = BPE
        var modelType: Int
        var byteFallback: Bool
        var addDummyPrefix: Bool
        var bosID: Int?
        var eosID: Int?
        var unkID: Int?
    }

    /// tokenizer.model（protobuf の ModelProto）を読む
    static func parseSentencePiece(_ data: Data) throws -> SentencePieceModel {
        let fields = try ProtoWire.parse(data)
        var pieces: [Piece] = []
        var modelType = 1
        var byteFallback = false
        var addDummyPrefix = true
        var bos: Int? = 1, eos: Int? = 2, unk: Int? = 0
        for field in fields {
            switch field.number {
            case 1:
                guard let bytes = field.bytes else { continue }
                var text = ""
                var score: Float = 0
                var kind = Piece.Kind.normal
                for inner in try ProtoWire.parse(bytes) {
                    switch (inner.number, inner.value) {
                    case (1, .bytes(let value)): text = String(decoding: value, as: UTF8.self)
                    case (2, .fixed32(let bits)): score = Float(bitPattern: bits)
                    case (3, .varint(let value)): kind = Piece.Kind(rawValue: Int(value)) ?? .normal
                    default: break
                    }
                }
                pieces.append(Piece(text: text, score: score, kind: kind))
            case 2:
                guard let bytes = field.bytes else { continue }
                for inner in try ProtoWire.parse(bytes) {
                    guard let value = inner.varint else { continue }
                    let signed = Int(Int64(bitPattern: value))
                    switch inner.number {
                    case 3: modelType = signed
                    case 35: byteFallback = value != 0
                    case 40: unk = signed >= 0 ? signed : nil
                    case 41: bos = signed >= 0 ? signed : nil
                    case 42: eos = signed >= 0 ? signed : nil
                    default: break
                    }
                }
            case 3:
                guard let bytes = field.bytes else { continue }
                for inner in try ProtoWire.parse(bytes) where inner.number == 3 {
                    addDummyPrefix = (inner.varint ?? 1) != 0
                }
            default:
                break
            }
        }
        guard !pieces.isEmpty else { throw TokenizerError.invalidFormat(L("tokenizer.model にピースがありません", "tokenizer.model has no pieces")) }
        if !byteFallback { byteFallback = pieces.contains { $0.kind == .byte } }
        return SentencePieceModel(pieces: pieces, modelType: modelType, byteFallback: byteFallback, addDummyPrefix: addDummyPrefix,
                                  bosID: bos, eosID: eos, unkID: unk)
    }

    private static func sentencePiece(_ data: Data, config: [String: Any]) throws -> [String: Any] {
        let model = try parseSentencePiece(data)
        guard model.modelType == 2 else {
            throw TokenizerError.unsupported(L("SentencePiece の Unigram 形式（BPE のみ対応）", "SentencePiece Unigram models (only BPE is supported)"))
        }
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(model.pieces.count)
        for (id, piece) in model.pieces.enumerated() where vocab[piece.text] == nil { vocab[piece.text] = id }

        // 結合規則: 各ピースを 2 つに分けて、両方が語彙にあるもの。ピースのスコアの高い順（同じなら左・右の ID 順）
        var merges: [(left: String, right: String, score: Float, order: Int)] = []
        var order = 0
        for piece in model.pieces {
            let scalars = Array(piece.text.unicodeScalars)
            guard scalars.count > 1 else { continue }
            var local: [(String, String, Int, Int)] = []
            for split in 1..<scalars.count {
                let left = String(String.UnicodeScalarView(scalars[..<split]))
                let right = String(String.UnicodeScalarView(scalars[split...]))
                if let leftID = vocab[left], let rightID = vocab[right] { local.append((left, right, leftID, rightID)) }
            }
            local.sort { ($0.2, $0.3) < ($1.2, $1.3) }
            for item in local {
                merges.append((item.0, item.1, piece.score, order))
                order += 1
            }
        }
        merges.sort { $0.score != $1.score ? $0.score > $1.score : $0.order < $1.order }

        var added: [[String: Any]] = []
        for (id, piece) in model.pieces.enumerated() where piece.kind == .control || piece.kind == .unknown || piece.kind == .userDefined {
            added.append(AddedToken(id: id, content: piece.text, special: piece.kind != .userDefined).json)
        }
        let unkToken = model.unkID.flatMap { $0 < model.pieces.count ? model.pieces[$0].text : nil }
        var normalizers: [[String: Any]] = []
        if model.addDummyPrefix { normalizers.append(["type": "Prepend", "prepend": "▁"]) }
        normalizers.append(["type": "Replace", "pattern": ["String": " "], "content": "▁"])
        var decoders: [[String: Any]] = [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
            ["type": "ByteFallback"],
            ["type": "Fuse"],
        ]
        if model.addDummyPrefix { decoders.append(["type": "Strip", "content": " ", "start": 1, "stop": 0]) }

        var root: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": added,
            "normalizer": ["type": "Sequence", "normalizers": normalizers],
            "pre_tokenizer": NSNull(),
            "decoder": ["type": "Sequence", "decoders": decoders],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": unkToken.map { $0 as Any } ?? NSNull(),
                "continuing_subword_prefix": NSNull(), "end_of_word_suffix": NSNull(), "fuse_unk": true,
                "byte_fallback": model.byteFallback, "ignore_merges": false,
                "vocab": vocab, "merges": merges.map { [$0.left, $0.right] },
            ] as [String: Any],
        ]
        let addBos = config["add_bos_token"] as? Bool ?? true
        if addBos, let bos = model.bosID, bos < model.pieces.count {
            let token = model.pieces[bos].text
            root["post_processor"] = [
                "type": "TemplateProcessing",
                "single": [["SpecialToken": ["id": token, "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]]],
                "pair": [["SpecialToken": ["id": token, "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]],
                         ["SpecialToken": ["id": token, "type_id": 1]], ["Sequence": ["id": "B", "type_id": 1]]],
                "special_tokens": [token: ["id": token, "ids": [bos], "tokens": [token]]],
            ] as [String: Any]
        } else {
            root["post_processor"] = NSNull()
        }
        return root
    }

    // MARK: - バイト単位の BPE（vocab.json + merges.txt）

    /// Qwen の前処理の正規表現
    static let qwenPattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    /// Llama 3 / tiktoken（cl100k）の前処理の正規表現
    static let llama3Pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private static func byteLevelBPE(vocabJSON: Data, merges: String, config: [String: Any]) throws -> [String: Any] {
        guard let vocab = try JSONSerialization.jsonObject(with: vocabJSON) as? [String: Any] else {
            throw TokenizerError.invalidFormat(L("vocab.json が読めません", "Cannot read vocab.json"))
        }
        var mergeList: [String] = []
        for line in merges.split(whereSeparator: { $0 == "\n" || $0 == "\r\n" }) {
            let text = line.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
            if text.isEmpty || text.hasPrefix("#version") { continue }
            mergeList.append(text)
        }
        let tokenizerClass = (config["tokenizer_class"] as? String ?? "").lowercased()
        let addPrefixSpace = config["add_prefix_space"] as? Bool ?? false
        let preTokenizer: [String: Any]
        var normalizer: Any = NSNull()
        if tokenizerClass.hasPrefix("qwen2") || tokenizerClass.hasPrefix("qwen3") {
            normalizer = ["type": "NFC"]
            preTokenizer = splitThenByteLevel(qwenPattern)
        } else {
            preTokenizer = ["type": "ByteLevel", "add_prefix_space": addPrefixSpace, "trim_offsets": true, "use_regex": true]
        }
        return [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": [[String: Any]](),
            "normalizer": normalizer,
            "pre_tokenizer": preTokenizer,
            "post_processor": NSNull(),
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": mergeList,
            ] as [String: Any],
        ]
    }

    private static func splitThenByteLevel(_ pattern: String) -> [String: Any] {
        [
            "type": "Sequence",
            "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": pattern], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
            ],
        ]
    }

    // MARK: - tiktoken

    private static func tiktoken(_ text: String, config: [String: Any]) throws -> [String: Any] {
        var ranks: [[UInt8]: Int] = [:]
        var ordered: [([UInt8], Int)] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: " ")
            guard parts.count == 2, let rank = Int(parts[1]), let bytes = Data(base64Encoded: String(parts[0])) else { continue }
            let token = [UInt8](bytes)
            ranks[token] = rank
            ordered.append((token, rank))
        }
        guard !ordered.isEmpty else { throw TokenizerError.invalidFormat(L("tiktoken のファイルが読めません", "Cannot read the tiktoken file")) }
        ordered.sort { $0.1 < $1.1 }
        let unicode = byteToUnicode
        func string(_ bytes: [UInt8]) -> String {
            var view = String.UnicodeScalarView()
            for byte in bytes { view.append(unicode[Int(byte)]) }
            return String(view)
        }
        var vocab: [String: Int] = [:]
        var merges: [(String, String, Int, Int, Int)] = []
        for (token, rank) in ordered {
            vocab[string(token)] = rank
            guard token.count > 1 else { continue }
            var local: [(String, String, Int, Int, Int)] = []
            for split in 1..<token.count {
                let left = Array(token[..<split]), right = Array(token[split...])
                if let leftRank = ranks[left], let rightRank = ranks[right] {
                    local.append((string(left), string(right), rank, leftRank, rightRank))
                }
            }
            local.sort { ($0.3, $0.4) < ($1.3, $1.4) }
            merges.append(contentsOf: local)
        }
        // 結合後のトークンの順位の順（Swift の sort は安定でないので、同じ順位の中の順番も比べる）
        merges.sort { ($0.2, $0.3, $0.4) < ($1.2, $1.3, $1.4) }
        return [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": [[String: Any]](),
            "normalizer": NSNull(),
            "pre_tokenizer": splitThenByteLevel(llama3Pattern),
            "post_processor": NSNull(),
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                "vocab": vocab, "merges": merges.map { [$0.0, $0.1] },
            ] as [String: Any],
        ]
    }

    /// GPT-2 のバイト → 文字の対応（バイト単位の BPE が語彙に使う文字）
    static let byteToUnicode: [Unicode.Scalar] = {
        var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var characters = bytes
        var extra = 0
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            characters.append(256 + extra)
            extra += 1
        }
        var table = [Unicode.Scalar](repeating: " ", count: 256)
        for (byte, character) in zip(bytes, characters) {
            table[byte] = Unicode.Scalar(UInt32(character)) ?? " "
        }
        return table
    }()
}

/// tokenizer.json が手に入らないとき、同じトークナイザを持つ公開リポジトリを探す。
enum TokenizerFallback {
    /// 系列の名前（小文字）→ トークナイザを取ってくるリポジトリ。上から順に、より細かい名前を先に調べる。
    static let families: [(keys: [String], repo: String)] = [
        (["qwen3"], "Qwen/Qwen3-0.6B"),
        (["qwen2.5-coder", "qwen2_5-coder", "qwen25-coder"], "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        (["deepseek-r1-distill-qwen"], "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        (["vibethinker"], "WeiboAI/VibeThinker-1.5B"),
        (["qwen2.5", "qwen2_5", "qwen25", "qwen2"], "Qwen/Qwen2.5-0.5B-Instruct"),
        (["llama-3", "llama3", "llama_3"], "unsloth/Llama-3.2-1B-Instruct"),
        (["gemma-3", "gemma3"], "unsloth/gemma-3-1b-it"),
        (["gemma-2", "gemma2"], "unsloth/gemma-2-2b-it"),
        (["mistral"], "unsloth/mistral-7b-instruct-v0.3"),
        (["phi-3", "phi3"], "microsoft/Phi-3-mini-4k-instruct"),
        (["smollm2"], "HuggingFaceTB/SmolLM2-360M-Instruct"),
        (["smollm"], "HuggingFaceTB/SmolLM-360M-Instruct"),
        (["tinyllama"], "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        (["exaone"], "LGAI-EXAONE/EXAONE-4.0-1.2B"),
    ]

    /// 試す順のリポジトリ（元のモデル → 名前から推測した系列）
    static func candidates(repoID: String, baseModels: [String] = [], names: [String] = []) -> [String] {
        var result: [String] = []
        for base in baseModels where !base.isEmpty && base != repoID && !result.contains(base) { result.append(base) }
        let text = ([repoID] + names + baseModels).joined(separator: " ").lowercased()
        for family in families where family.keys.contains(where: { text.contains($0) }) {
            if !result.contains(family.repo) { result.append(family.repo) }
            break
        }
        return result
    }
}

extension TokenizerFallback {
    /// いっしょに取ってくる付属ファイル
    static let companionFiles = ["tokenizer_config.json", "chat_template.jinja", "generation_config.json", "special_tokens_map.json"]

    /// 候補のリポジトリから tokenizer.json（と付属ファイル）を取ってきて folder に置く。取れたリポジトリを返す。
    static func fetch(client: HFClient, candidates: [String], into folder: URL) async -> String? {
        for repo in candidates {
            guard let data = try? await client.fetchData(repo: repo, path: "tokenizer.json", revision: "main"), !data.isEmpty,
                  (try? JSONSerialization.jsonObject(with: data)) is [String: Any] else { continue }
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            guard (try? data.write(to: folder.appendingPathComponent("tokenizer.json"), options: .atomic)) != nil else { return nil }
            for name in companionFiles {
                if let extra = try? await client.fetchData(repo: repo, path: name, revision: "main"), !extra.isEmpty {
                    try? extra.write(to: folder.appendingPathComponent(name), options: .atomic)
                }
            }
            try? Data(repo.utf8).write(to: folder.appendingPathComponent("SOURCE.txt"))
            return repo
        }
        return nil
    }
}

