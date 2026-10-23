import Foundation

enum TokenizerError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let name):
            return L("トークナイザのファイルが見つかりません: \(name)", "Tokenizer file not found: \(name)")
        case .invalidFormat(let detail):
            return L("tokenizer.json の形式が想定外です: \(detail)", "Unexpected tokenizer.json format: \(detail)")
        case .unsupported(let detail):
            return L("このトークナイザには対応していません: \(detail)", "This tokenizer is not supported: \(detail)")
        }
    }
}

/// tokenizer.json の added_tokens（<|im_end|> など）1 つ分。
struct TokAddedToken {
    var id: Int
    var content: String
    var scalars: [Unicode.Scalar]
    var special: Bool
    var lstrip: Bool
    var rstrip: Bool
    var singleWord: Bool
    var normalized: Bool
}

/// 追加トークンを入力文字列から探す（最も左で、同じ位置なら最も長いもの）。
final class TokAddedTokenMatcher {
    private struct Node {
        var children: [UInt32: Int] = [:]
        var tokenIndex: Int?
    }

    private var nodes: [Node] = [Node()]
    private let tokens: [TokAddedToken]

    init(tokens: [TokAddedToken]) {
        self.tokens = tokens
        for (index, token) in tokens.enumerated() where !token.scalars.isEmpty {
            var node = 0
            for scalar in token.scalars {
                if let child = nodes[node].children[scalar.value] {
                    node = child
                } else {
                    nodes.append(Node())
                    nodes[node].children[scalar.value] = nodes.count - 1
                    node = nodes.count - 1
                }
            }
            nodes[node].tokenIndex = index
        }
    }

    var isEmpty: Bool { tokens.isEmpty }

    /// 文字列を「追加トークン」と「それ以外の文字列」に分ける。
    /// 戻り値: (トークンID または nil, 文字列, 元の文字列の先頭から始まるか)
    func split(_ text: String, isOriginalStart: Bool) -> [(id: Int?, text: String, isStart: Bool)] {
        let scalars = Array(text.unicodeScalars)
        if tokens.isEmpty || scalars.isEmpty {
            return scalars.isEmpty ? [] : [(nil, text, isOriginalStart)]
        }
        var result: [(id: Int?, text: String, isStart: Bool)] = []
        var segmentStart = 0
        var position = 0
        while position < scalars.count {
            guard let (tokenIndex, matchEnd) = longestMatch(scalars, from: position) else {
                position += 1
                continue
            }
            let token = tokens[tokenIndex]
            var start = position
            var stop = matchEnd
            if token.singleWord {
                let startOK = start == 0 || !scalars[start - 1].tokIsAlphanumeric
                let stopOK = stop == scalars.count || !scalars[stop].tokIsAlphanumeric
                if !(startOK && stopOK) {
                    position += 1
                    continue
                }
            }
            if token.lstrip {
                var newStart = start
                while newStart > 0 && scalars[newStart - 1].tokIsWhitespace { newStart -= 1 }
                start = max(newStart, segmentStart)
            }
            if token.rstrip {
                while stop < scalars.count && scalars[stop].tokIsWhitespace { stop += 1 }
            }
            if segmentStart < start {
                result.append((nil, Self.string(scalars[segmentStart..<start]), isOriginalStart && segmentStart == 0))
            }
            result.append((token.id, token.content, false))
            segmentStart = stop
            position = stop
        }
        if segmentStart < scalars.count {
            result.append((nil, Self.string(scalars[segmentStart...]), isOriginalStart && segmentStart == 0))
        }
        return result
    }

    private func longestMatch(_ scalars: [Unicode.Scalar], from start: Int) -> (Int, Int)? {
        var node = 0
        var best: (Int, Int)?
        var index = start
        while index < scalars.count, let child = nodes[node].children[scalars[index].value] {
            node = child
            index += 1
            if let tokenIndex = nodes[node].tokenIndex { best = (tokenIndex, index) }
        }
        return best
    }

    private static func string(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }
}

/// Hugging Face の tokenizer.json を読む BPE トークナイザ（純 Swift）。
/// Python の `tokenizers` と同じ結果になるように作ってある（テストで確認）。
final class HFTokenizer: TextTokenizer {
    private let model: TokBPEModel
    private let normalizer: TokNormalizer?
    private let preTokenizer: TokPreTokenizer?
    private let decoder: TokDecoder?
    private let postProcessor: TokPostProcessor?
    private let addedTokens: [TokAddedToken]
    private let addedByID: [Int: TokAddedToken]
    private let addedByContent: [TokByteKey: Int]
    private let rawMatcher: TokAddedTokenMatcher
    private let normalizedMatcher: TokAddedTokenMatcher

    let vocabSize: Int
    let specialTokenIDs: Set<Int>
    let bosToken: String?
    let eosToken: String?
    let padToken: String?
    let unkToken: String?
    let addBosToken: Bool?
    let chatTemplate: String?

    init(tokenizerJSON: Data, tokenizerConfigJSON: Data?) throws {
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: tokenizerJSON)
        } catch {
            throw TokenizerError.invalidFormat(L("JSON として読めません", "Not valid JSON"))
        }
        guard let root = parsed as? [String: Any], let modelJSON = root["model"] as? [String: Any] else {
            throw TokenizerError.invalidFormat(L("model がありません", "model is missing"))
        }
        let type = modelJSON["type"] as? String ?? "BPE"
        guard type == "BPE" else {
            throw TokenizerError.unsupported(L("\(type) 形式のトークナイザ（BPE のみ対応）", "\(type) tokenizers (only BPE is supported)"))
        }
        model = try TokBPEModel(json: modelJSON, vocabEntries: TokVocabScanner.scanVocab(tokenizerJSON))
        normalizer = try TokNormalizer.parse(root["normalizer"])
        preTokenizer = try TokPreTokenizer.parse(root["pre_tokenizer"])
        decoder = try TokDecoder.parse(root["decoder"])

        var added: [TokAddedToken] = []
        for entry in root["added_tokens"] as? [[String: Any]] ?? [] {
            guard let id = entry["id"] as? Int, let content = entry["content"] as? String else { continue }
            added.append(TokAddedToken(id: id, content: content, scalars: Array(content.unicodeScalars),
                                       special: entry["special"] as? Bool ?? false,
                                       lstrip: entry["lstrip"] as? Bool ?? false,
                                       rstrip: entry["rstrip"] as? Bool ?? false,
                                       singleWord: entry["single_word"] as? Bool ?? false,
                                       normalized: entry["normalized"] as? Bool ?? !(entry["special"] as? Bool ?? false)))
        }
        addedTokens = added
        var byID: [Int: TokAddedToken] = [:]
        var byContent: [TokByteKey: Int] = [:]
        for token in added {
            byID[token.id] = token
            byContent[TokByteKey(token.content)] = token.id
        }
        addedByID = byID
        addedByContent = byContent
        rawMatcher = TokAddedTokenMatcher(tokens: added.filter { !$0.normalized })
        normalizedMatcher = TokAddedTokenMatcher(tokens: added.filter { $0.normalized })
        specialTokenIDs = Set(added.filter(\.special).map(\.id))
        vocabSize = max(model.maxID, added.map(\.id).max() ?? -1) + 1

        let vocab = model.vocab
        let lookup: (String) -> Int? = { name in byContent[TokByteKey(name)] ?? vocab[TokByteKey(name)] }
        postProcessor = try TokPostProcessor.parse(root["post_processor"], tokenID: lookup)

        // tokenizer_config.json
        var config: [String: Any] = [:]
        if let data = tokenizerConfigJSON,
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            config = object
        }
        func tokenString(_ value: Any?) -> String? {
            if let text = value as? String { return text }
            if let object = value as? [String: Any] { return object["content"] as? String }
            return nil
        }
        bosToken = tokenString(config["bos_token"])
        eosToken = tokenString(config["eos_token"])
        padToken = tokenString(config["pad_token"])
        unkToken = tokenString(config["unk_token"])
        addBosToken = config["add_bos_token"] as? Bool
        if let text = config["chat_template"] as? String {
            chatTemplate = text
        } else if let list = config["chat_template"] as? [[String: Any]] {
            let preferred = list.first { ($0["name"] as? String) == "default" } ?? list.first
            chatTemplate = preferred?["template"] as? String
        } else {
            chatTemplate = nil
        }
    }

    /// ディレクトリ内の tokenizer.json（と tokenizer_config.json）を読む。
    static func load(directory: URL) throws -> HFTokenizer {
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: tokenizerURL) else {
            throw TokenizerError.fileNotFound("tokenizer.json")
        }
        let config = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
        return try HFTokenizer(tokenizerJSON: data, tokenizerConfigJSON: config)
    }

    // MARK: - TextTokenizer

    func encode(_ text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        // 1. 正規化しない追加トークンを、元の文字列から切り出す
        for part in rawMatcher.split(text, isOriginalStart: true) {
            if let id = part.id {
                ids.append(id)
                continue
            }
            // 2. 残りを正規化してから、正規化する追加トークンを切り出す
            let normalized = normalizer?.apply(part.text) ?? part.text
            if normalized.isEmpty { continue }
            for sub in normalizedMatcher.split(normalized, isOriginalStart: part.isStart) {
                if let id = sub.id {
                    ids.append(id)
                    continue
                }
                // 3. 前処理で単語に分け、BPE でトークンにする
                let pieces = preTokenizer?.apply([TokPiece(text: sub.text, isOriginalStart: sub.isStart)])
                    ?? [TokPiece(text: sub.text, isOriginalStart: sub.isStart)]
                for piece in pieces where !piece.text.isEmpty {
                    ids.append(contentsOf: model.tokenize(piece.text))
                }
            }
        }
        if addSpecialTokens, let postProcessor {
            ids = postProcessor.process(ids, tokenID: { [weak self] in self?.tokenID(for: $0) })
        }
        return ids
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String {
        var tokens: [String] = []
        tokens.reserveCapacity(ids.count)
        for id in ids {
            if skipSpecialTokens && specialTokenIDs.contains(id) { continue }
            if let added = addedByID[id] {
                tokens.append(added.content)
            } else if let token = model.idToToken[id] {
                tokens.append(token)
            }
        }
        guard let decoder else { return tokens.joined(separator: " ") }
        return decoder.decodeChain(tokens).joined()
    }

    func tokenID(for token: String) -> Int? {
        let key = TokByteKey(token)
        return addedByContent[key] ?? model.vocab[key]
    }

    func token(for id: Int) -> String? {
        addedByID[id]?.content ?? model.idToToken[id]
    }
}
