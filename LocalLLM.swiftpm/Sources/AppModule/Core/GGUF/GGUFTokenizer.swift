import Foundation

/// GGUF に入っているトークナイザの情報（語彙・結合規則・チャットテンプレート）から、
/// このアプリのトークナイザが読める tokenizer.json などを作る。llama.cpp の llama-vocab.cpp と同じ規則。
enum GGUFTokenizer {
    /// 作ったファイルの中身
    struct Files {
        var tokenizerJSON: Data
        var tokenizerConfigJSON: Data
        var generationConfigJSON: Data
    }

    /// GGUF の語彙の種類（tokenizer.ggml.token_type）
    private enum TokenType: Int {
        case normal = 1, unknown, control, userDefined, unused, byte
    }

    static func files(from file: GGUFFile) throws -> Files {
        Files(tokenizerJSON: try tokenizerJSON(from: file),
              tokenizerConfigJSON: try tokenizerConfigJSON(from: file),
              generationConfigJSON: try generationConfigJSON(from: file))
    }

    // MARK: - tokenizer.json

    static func tokenizerJSON(from file: GGUFFile) throws -> Data {
        guard let tokens = file.metadata["tokenizer.ggml.tokens"]?.stringArray, !tokens.isEmpty else {
            throw GGUFError.missingMetadata("tokenizer.ggml.tokens")
        }
        let model = file.string("tokenizer.ggml.model") ?? "llama"
        let types = file.metadata["tokenizer.ggml.token_type"]?.intArray ?? []
        let root: [String: Any]
        switch model {
        case "llama":
            root = sentencePiece(file: file, tokens: tokens, types: types)
        case "gpt2":
            root = byteLevel(file: file, tokens: tokens, types: types)
        default:
            throw TokenizerError.unsupported(L("GGUF のトークナイザ「\(model)」", "the GGUF tokenizer \"\(model)\""))
        }
        return try JSONSerialization.data(withJSONObject: root)
    }

    /// Phi-3 は、特別なトークンのあとの空白を取り除く（llama.cpp も tokenizer.json も同じ）
    private static func stripsAfterSpecial(_ file: GGUFFile) -> Bool {
        let name = (file.name ?? "").lowercased()
        return name.contains("phi-3") || name.contains("phi3")
    }

    private static func sentencePiece(file: GGUFFile, tokens: [String], types: [Int]) -> [String: Any] {
        let scores = file.metadata["tokenizer.ggml.scores"]?.floatArray ?? []
        var pieces: [TokenizerSynthesizer.Piece] = []
        pieces.reserveCapacity(tokens.count)
        for (index, text) in tokens.enumerated() {
            let kind = TokenizerSynthesizer.Piece.Kind(rawValue: index < types.count ? types[index] : 1) ?? .normal
            pieces.append(TokenizerSynthesizer.Piece(text: text, score: index < scores.count ? scores[index] : 0, kind: kind))
        }
        let model = TokenizerSynthesizer.SentencePieceModel(
            pieces: pieces, modelType: 2,
            byteFallback: pieces.contains { $0.kind == .byte },
            addDummyPrefix: file.bool("tokenizer.ggml.add_space_prefix") ?? true,
            bosID: file.int("tokenizer.ggml.bos_token_id") ?? 1,
            eosID: file.int("tokenizer.ggml.eos_token_id") ?? 2,
            unkID: file.int("tokenizer.ggml.unknown_token_id") ?? 0)
        var root = TokenizerSynthesizer.sentencePieceJSON(model, addBos: file.bool("tokenizer.ggml.add_bos_token") ?? true)
        // 結合規則がそのまま入っていれば（古い変換の GGUF）、スコアから作り直さずにそれを使う
        if let merges = file.metadata["tokenizer.ggml.merges"]?.stringArray, !merges.isEmpty,
           var bpe = root["model"] as? [String: Any] {
            bpe["merges"] = merges
            root["model"] = bpe
        }
        // Mistral の v3 トークナイザ（[INST] が制御トークン）は、文の先頭にだけ（空白で始まっていなければ）▁ を付ける
        if model.addDummyPrefix, let inst = tokens.firstIndex(of: "[INST]"), inst < types.count, types[inst] == TokenType.control.rawValue {
            root["normalizer"] = NSNull()
            root["pre_tokenizer"] = ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false]
        }
        if stripsAfterSpecial(file), var added = root["added_tokens"] as? [[String: Any]] {
            // llama.cpp と同じ: 特別なトークンは後ろの空白を取り込む（<unk>・<s>・<|endoftext|> は除く）
            let keep: Set<String> = ["<unk>", "<s>", "<|endoftext|>"]
            for index in added.indices {
                guard let content = added[index]["content"] as? String, !keep.contains(content) else { continue }
                added[index]["rstrip"] = true
            }
            root["added_tokens"] = added
        }
        return root
    }

    // MARK: - バイト単位の BPE

    /// 数字を 1 つずつ分けてから GPT-2 の規則で分ける系統（tokenizer.ggml.pre）
    private static let digitFamilies: Set<String> = ["smollm", "starcoder", "refact", "command-r", "codeshell", "smaug-bpe", "exaone"]
    private static let llama3Families: Set<String> = ["llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2"]
    private static let qwenFamilies: Set<String> = ["qwen2", "deepseek-r1-qwen", "megrez", "kormo", "qwen35", "stablelm2", "hunyuan"]
    static let tekkenPattern = #"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    static let gpt4oPattern = #"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private static func byteLevel(file: GGUFFile, tokens: [String], types: [Int]) -> [String: Any] {
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(tokens.count)
        for (index, token) in tokens.enumerated() where vocab[token] == nil { vocab[token] = index }
        let merges = file.metadata["tokenizer.ggml.merges"]?.stringArray ?? []
        let pre = file.string("tokenizer.ggml.pre") ?? "default"

        var normalizer: Any = NSNull()
        var ignoreMerges = false
        let plain: [String: Any] = ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true]
        let preTokenizer: [String: Any]
        if qwenFamilies.contains(pre) {
            normalizer = ["type": "NFC"]
            preTokenizer = TokenizerSynthesizer.splitThenByteLevel(TokenizerSynthesizer.qwenPattern)
        } else if llama3Families.contains(pre) {
            ignoreMerges = true
            preTokenizer = TokenizerSynthesizer.splitThenByteLevel(TokenizerSynthesizer.llama3Pattern)
        } else if pre == "tekken" {
            ignoreMerges = true
            preTokenizer = TokenizerSynthesizer.splitThenByteLevel(tekkenPattern)
        } else if pre == "gpt-4o" || pre == "llama4" {
            preTokenizer = TokenizerSynthesizer.splitThenByteLevel(gpt4oPattern)
        } else if digitFamilies.contains(pre) {
            preTokenizer = ["type": "Sequence", "pretokenizers": [["type": "Digits", "individual_digits": true], plain]]
        } else {
            preTokenizer = plain
        }

        var added: [[String: Any]] = []
        for (index, token) in tokens.enumerated() where index < types.count {
            guard let type = TokenType(rawValue: types[index]), type == .control || type == .userDefined else { continue }
            added.append(TokenizerSynthesizer.AddedToken(id: index, content: token, special: type == .control).json)
        }

        var root: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": added,
            "normalizer": normalizer,
            "pre_tokenizer": preTokenizer,
            "post_processor": NSNull(),
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": ignoreMerges,
                "vocab": vocab, "merges": merges,
            ] as [String: Any],
        ]
        // 先頭に BOS を付ける設定（Llama 3 系は既定で付ける）
        let addBos = file.bool("tokenizer.ggml.add_bos_token") ?? llama3Families.contains(pre)
        if addBos, let bos = file.int("tokenizer.ggml.bos_token_id"), bos >= 0, bos < tokens.count {
            let token = tokens[bos]
            root["post_processor"] = [
                "type": "TemplateProcessing",
                "single": [["SpecialToken": ["id": token, "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]]],
                "pair": [["SpecialToken": ["id": token, "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]],
                         ["Sequence": ["id": "B", "type_id": 1]]],
                "special_tokens": [token: ["id": token, "ids": [bos], "tokens": [token]]],
            ] as [String: Any]
        }
        return root
    }

    // MARK: - 付属の設定

    /// tokenizer_config.json に当たるもの（BOS / EOS の文字列とチャットテンプレート）
    static func tokenizerConfigJSON(from file: GGUFFile) throws -> Data {
        let tokens = file.metadata["tokenizer.ggml.tokens"]?.stringArray ?? []
        func token(_ key: String) -> String? {
            guard let id = file.int("tokenizer.ggml.\(key)"), id >= 0, id < tokens.count else { return nil }
            return tokens[id]
        }
        var config: [String: Any] = [:]
        // SentencePiece 系は、番号が書かれていなければ決まった位置（1 / 2 / 0）
        let isSentencePiece = (file.string("tokenizer.ggml.model") ?? "llama") == "llama"
        config["bos_token"] = token("bos_token_id") ?? (isSentencePiece && tokens.count > 1 ? tokens[1] : nil)
        config["eos_token"] = token("eos_token_id") ?? (isSentencePiece && tokens.count > 2 ? tokens[2] : nil)
        config["unk_token"] = token("unknown_token_id")
        config["pad_token"] = token("padding_token_id")
        if let addBos = file.bool("tokenizer.ggml.add_bos_token") { config["add_bos_token"] = addBos }
        if let template = file.string("tokenizer.chat_template") { config["chat_template"] = template }
        return try JSONSerialization.data(withJSONObject: config.compactMapValues { $0 })
    }

    /// 生成を止めるトークン（EOS・ターンの終わり・メッセージの終わり）
    static func generationConfigJSON(from file: GGUFFile) throws -> Data {
        var ids: [Int] = []
        for key in ["eos_token_id", "eot_token_id", "eom_token_id"] {
            if let id = file.int("tokenizer.ggml.\(key)"), id >= 0, !ids.contains(id) { ids.append(id) }
        }
        return try JSONSerialization.data(withJSONObject: ["eos_token_id": ids])
    }
}
