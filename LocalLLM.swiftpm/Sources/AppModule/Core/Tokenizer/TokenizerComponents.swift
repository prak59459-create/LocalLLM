import Foundation

/// 文字列パターン（Hugging Face の {"String": ...} / {"Regex": ...}）
enum TokPattern {
    case string(String)
    case regex(NSRegularExpression)

    static func parse(_ json: Any?) throws -> TokPattern {
        guard let object = json as? [String: Any] else {
            throw TokenizerError.invalidFormat(L("pattern の形式が不正です", "Invalid pattern"))
        }
        if let text = object["String"] as? String { return .string(text) }
        if let text = object["Regex"] as? String { return .regex(try TokRegex.make(text)) }
        throw TokenizerError.invalidFormat(L("pattern の形式が不正です", "Invalid pattern"))
    }

    var regex: NSRegularExpression {
        switch self {
        case .string(let text): return TokRegex.literal(text)
        case .regex(let regex): return regex
        }
    }

    /// すべての一致を content に置き換える（文字単位で厳密に比べる）。
    func replaceAll(in text: String, with content: String) -> String {
        switch self {
        case .string(let pattern):
            return TokText.replacing(pattern, with: content, in: text)
        case .regex(let regex):
            let ns = text as NSString
            let template = NSRegularExpression.escapedTemplate(for: content)
            return regex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: ns.length),
                                                  withTemplate: template)
        }
    }
}

/// Unicode スカラー単位の文字列操作（String の比較は正規等価なので使わない）。
enum TokText {
    static func hasPrefix(_ text: String, _ prefix: String) -> Bool {
        text.unicodeScalars.starts(with: prefix.unicodeScalars)
    }

    static func replacing(_ pattern: String, with content: String, in text: String) -> String {
        let source = Array(text.unicodeScalars)
        let target = Array(pattern.unicodeScalars)
        guard !target.isEmpty, source.count >= target.count else { return text }
        var result = String.UnicodeScalarView()
        var index = 0
        while index < source.count {
            if index + target.count <= source.count && source[index] == target[0]
                && Array(source[index..<(index + target.count)]) == target {
                result.append(contentsOf: content.unicodeScalars)
                index += target.count
            } else {
                result.append(source[index])
                index += 1
            }
        }
        return String(result)
    }
}

// MARK: - 正規化

indirect enum TokNormalizer {
    case nfc, nfd, nfkc, nfkd, lowercase
    case prepend(String)
    case replace(TokPattern, String)
    case strip(left: Bool, right: Bool)
    case stripAccents
    case sequence([TokNormalizer])

    static func parse(_ json: Any?) throws -> TokNormalizer? {
        guard let object = json as? [String: Any], let type = object["type"] as? String else { return nil }
        switch type {
        case "NFC": return .nfc
        case "NFD": return .nfd
        case "NFKC": return .nfkc
        case "NFKD": return .nfkd
        case "Lowercase": return .lowercase
        case "StripAccents": return .stripAccents
        case "Prepend":
            return .prepend(object["prepend"] as? String ?? "▁")
        case "Replace":
            return .replace(try TokPattern.parse(object["pattern"]), object["content"] as? String ?? "")
        case "Strip":
            return .strip(left: object["strip_left"] as? Bool ?? true, right: object["strip_right"] as? Bool ?? true)
        case "Sequence":
            let items = object["normalizers"] as? [Any] ?? []
            return .sequence(try items.compactMap { try TokNormalizer.parse($0) })
        case "BertNormalizer":
            // 主な挙動だけ（制御文字の除去は省略）: 小文字化とアクセント除去
            var steps: [TokNormalizer] = []
            if object["lowercase"] as? Bool ?? true {
                steps.append(.nfd)
                steps.append(.stripAccents)
                steps.append(.lowercase)
            }
            return .sequence(steps)
        default:
            throw TokenizerError.unsupported(L("未対応の normalizer: \(type)", "Unsupported normalizer: \(type)"))
        }
    }

    func apply(_ text: String) -> String {
        switch self {
        case .nfc: return text.precomposedStringWithCanonicalMapping
        case .nfd: return text.decomposedStringWithCanonicalMapping
        case .nfkc: return text.precomposedStringWithCompatibilityMapping
        case .nfkd: return text.decomposedStringWithCompatibilityMapping
        case .lowercase: return text.lowercased()
        case .prepend(let prefix): return text.isEmpty ? text : prefix + text
        case .replace(let pattern, let content): return pattern.replaceAll(in: text, with: content)
        case .strip(let left, let right):
            var scalars = Array(text.unicodeScalars)[...]
            if left { while let first = scalars.first, first.tokIsWhitespace { scalars = scalars.dropFirst() } }
            if right { while let last = scalars.last, last.tokIsWhitespace { scalars = scalars.dropLast() } }
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars)
            return String(view)
        case .stripAccents:
            var view = String.UnicodeScalarView()
            for scalar in text.unicodeScalars where scalar.properties.generalCategory != .nonspacingMark {
                view.append(scalar)
            }
            return String(view)
        case .sequence(let items):
            return items.reduce(text) { $1.apply($0) }
        }
    }
}

// MARK: - 前処理（pre-tokenizer）

indirect enum TokPreTokenizer {
    case byteLevel(addPrefixSpace: Bool, useRegex: Bool)
    case split(TokPattern, TokSplitBehavior, invert: Bool)
    case metaspace(replacement: String, prependScheme: String, split: Bool)
    case digits(individual: Bool)
    case punctuation(TokSplitBehavior)
    case whitespace
    case whitespaceSplit
    case bert
    case sequence([TokPreTokenizer])

    /// GPT-2 の分割用正規表現
    static let gpt2Regex: NSRegularExpression = {
        try! TokRegex.make("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")
    }()

    static let whitespaceRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "\\w+|[^\\w\\s]+", options: [])
    }()

    static func parse(_ json: Any?) throws -> TokPreTokenizer? {
        guard let object = json as? [String: Any], let type = object["type"] as? String else { return nil }
        switch type {
        case "ByteLevel":
            return .byteLevel(addPrefixSpace: object["add_prefix_space"] as? Bool ?? true,
                              useRegex: object["use_regex"] as? Bool ?? true)
        case "Split":
            let behavior = TokSplitBehavior(rawValue: object["behavior"] as? String ?? "Isolated") ?? .isolated
            return .split(try TokPattern.parse(object["pattern"]), behavior, invert: object["invert"] as? Bool ?? false)
        case "Metaspace":
            let replacement = object["replacement"] as? String ?? "▁"
            var scheme = object["prepend_scheme"] as? String
            if scheme == nil {
                scheme = (object["add_prefix_space"] as? Bool ?? true) ? "always" : "never"
            }
            return .metaspace(replacement: replacement, prependScheme: scheme ?? "always", split: object["split"] as? Bool ?? true)
        case "Digits":
            return .digits(individual: object["individual_digits"] as? Bool ?? false)
        case "Punctuation":
            return .punctuation(TokSplitBehavior(rawValue: object["behavior"] as? String ?? "Isolated") ?? .isolated)
        case "Whitespace":
            return .whitespace
        case "WhitespaceSplit":
            return .whitespaceSplit
        case "BertPreTokenizer":
            return .bert
        case "Sequence":
            let items = object["pretokenizers"] as? [Any] ?? []
            return .sequence(try items.compactMap { try TokPreTokenizer.parse($0) })
        default:
            throw TokenizerError.unsupported(L("未対応の pre_tokenizer: \(type)", "Unsupported pre_tokenizer: \(type)"))
        }
    }

    func apply(_ pieces: [TokPiece]) -> [TokPiece] {
        switch self {
        case .sequence(let items):
            return items.reduce(pieces) { $1.apply($0) }
        case .byteLevel(let addPrefixSpace, let useRegex):
            var result: [TokPiece] = []
            for var piece in pieces {
                if addPrefixSpace && !TokText.hasPrefix(piece.text, " ") {
                    piece.text = " " + piece.text
                }
                let parts = useRegex
                    ? TokSplitter.apply(TokSplitter.spans(regex: Self.gpt2Regex, in: piece.text), behavior: .isolated, invert: false, in: piece)
                    : [piece]
                for part in parts {
                    result.append(TokPiece(text: TokByteLevel.encode(part.text), isOriginalStart: part.isOriginalStart))
                }
            }
            return result.filter { !$0.text.isEmpty }
        case .split(let pattern, let behavior, let invert):
            let regex = pattern.regex
            return pieces.flatMap { piece in
                TokSplitter.apply(TokSplitter.spans(regex: regex, in: piece.text), behavior: behavior, invert: invert, in: piece)
            }
        case .metaspace(let replacement, let scheme, let split):
            var result: [TokPiece] = []
            for var piece in pieces {
                piece.text = TokText.replacing(" ", with: replacement, in: piece.text)
                if !TokText.hasPrefix(piece.text, replacement) {
                    if scheme == "always" || (scheme == "first" && piece.isOriginalStart) {
                        piece.text = replacement + piece.text
                    }
                }
                if split, let mark = replacement.unicodeScalars.first {
                    let spans = TokSplitter.spans(in: piece.text) { $0 == mark }
                    result.append(contentsOf: TokSplitter.apply(spans, behavior: .mergedWithNext, invert: false, in: piece))
                } else if !piece.text.isEmpty {
                    result.append(piece)
                }
            }
            return result
        case .digits(let individual):
            return pieces.flatMap { piece in
                TokSplitter.apply(TokSplitter.spans(in: piece.text) { $0.tokIsNumeric },
                                  behavior: individual ? .isolated : .contiguous, invert: false, in: piece)
            }
        case .punctuation(let behavior):
            return pieces.flatMap { piece in
                TokSplitter.apply(TokSplitter.spans(in: piece.text) { $0.tokIsPunctuation }, behavior: behavior, invert: false, in: piece)
            }
        case .whitespace:
            return pieces.flatMap { piece in
                // 一致した部分（単語・記号の並び）だけを残す
                TokSplitter.apply(TokSplitter.spans(regex: Self.whitespaceRegex, in: piece.text), behavior: .removed, invert: true, in: piece)
            }
        case .whitespaceSplit:
            return pieces.flatMap { piece in
                TokSplitter.apply(TokSplitter.spans(in: piece.text) { $0.tokIsWhitespace }, behavior: .removed, invert: false, in: piece)
            }
        case .bert:
            let words = TokPreTokenizer.whitespaceSplit.apply(pieces)
            return TokPreTokenizer.punctuation(.isolated).apply(words)
        }
    }
}

// MARK: - 復号（decoder）

indirect enum TokDecoder {
    case byteLevel
    case metaspace(replacement: Unicode.Scalar, prependScheme: String)
    case replace(TokPattern, String)
    case byteFallback
    case fuse
    case strip(content: Unicode.Scalar, start: Int, stop: Int)
    case wordPiece(prefix: String)
    case bpeSuffix(String)
    case sequence([TokDecoder])

    static func parse(_ json: Any?) throws -> TokDecoder? {
        guard let object = json as? [String: Any], let type = object["type"] as? String else { return nil }
        switch type {
        case "ByteLevel": return .byteLevel
        case "Metaspace":
            let replacement = (object["replacement"] as? String ?? "▁").unicodeScalars.first ?? "▁"
            var scheme = object["prepend_scheme"] as? String
            if scheme == nil { scheme = (object["add_prefix_space"] as? Bool ?? true) ? "always" : "never" }
            return .metaspace(replacement: replacement, prependScheme: scheme ?? "always")
        case "Replace":
            return .replace(try TokPattern.parse(object["pattern"]), object["content"] as? String ?? "")
        case "ByteFallback": return .byteFallback
        case "Fuse": return .fuse
        case "Strip":
            let content = (object["content"] as? String ?? " ").unicodeScalars.first ?? " "
            return .strip(content: content, start: object["start"] as? Int ?? 0, stop: object["stop"] as? Int ?? 0)
        case "WordPiece":
            return .wordPiece(prefix: object["prefix"] as? String ?? "##")
        case "BPEDecoder":
            return .bpeSuffix(object["suffix"] as? String ?? "</w>")
        case "Sequence":
            let items = object["decoders"] as? [Any] ?? []
            return .sequence(try items.compactMap { try TokDecoder.parse($0) })
        default:
            throw TokenizerError.unsupported(L("未対応の decoder: \(type)", "Unsupported decoder: \(type)"))
        }
    }

    /// Hugging Face の decode_chain と同じ処理（トークン列 → 文字列の列）。
    func decodeChain(_ tokens: [String]) -> [String] {
        switch self {
        case .sequence(let items):
            return items.reduce(tokens) { $1.decodeChain($0) }
        case .byteLevel:
            var bytes: [UInt8] = []
            for token in tokens {
                var tokenBytes: [UInt8] = []
                var ok = true
                for scalar in token.unicodeScalars {
                    if let byte = TokByteLevel.scalarToByte[scalar.value] {
                        tokenBytes.append(byte)
                    } else {
                        ok = false
                        break
                    }
                }
                if ok { bytes.append(contentsOf: tokenBytes) } else { bytes.append(contentsOf: Array(token.utf8)) }
            }
            return [String(decoding: bytes, as: UTF8.self)]
        case .metaspace(let replacement, let scheme):
            return tokens.enumerated().map { index, token in
                var view = String.UnicodeScalarView()
                for scalar in token.unicodeScalars {
                    if scalar == replacement {
                        if index == 0 && scheme != "never" { continue }
                        view.append(" ")
                    } else {
                        view.append(scalar)
                    }
                }
                return String(view)
            }
        case .replace(let pattern, let content):
            return tokens.map { pattern.replaceAll(in: $0, with: content) }
        case .byteFallback:
            var result: [String] = []
            var pending: [UInt8] = []
            func flush() {
                guard !pending.isEmpty else { return }
                if let text = String(validatingUTF8Bytes: pending) {
                    result.append(text)
                } else {
                    result.append(contentsOf: repeatElement("\u{FFFD}", count: pending.count))
                }
                pending.removeAll()
            }
            for token in tokens {
                if let byte = Self.fallbackByte(token) {
                    pending.append(byte)
                } else {
                    flush()
                    result.append(token)
                }
            }
            flush()
            return result
        case .fuse:
            return [tokens.joined()]
        case .strip(let content, let start, let stop):
            return tokens.map { token in
                let scalars = Array(token.unicodeScalars)
                var startCut = 0
                while startCut < start && startCut < scalars.count && scalars[startCut] == content { startCut += 1 }
                var stopCut = 0
                while stopCut < stop && stopCut < scalars.count - startCut && scalars[scalars.count - 1 - stopCut] == content { stopCut += 1 }
                var view = String.UnicodeScalarView()
                view.append(contentsOf: scalars[startCut..<(scalars.count - stopCut)])
                return String(view)
            }
        case .wordPiece(let prefix):
            return tokens.enumerated().map { index, token in
                if TokText.hasPrefix(token, prefix) {
                    return String(token.unicodeScalars.dropFirst(prefix.unicodeScalars.count))
                }
                return index == 0 ? token : " " + token
            }
        case .bpeSuffix(let suffix):
            return tokens.enumerated().map { index, token in
                TokText.replacing(suffix, with: index == tokens.count - 1 ? "" : " ", in: token)
            }
        }
    }

    /// "<0xE3>" のようなバイトのトークンなら、その値を返す。
    static func fallbackByte(_ token: String) -> UInt8? {
        let utf8 = Array(token.utf8)
        guard utf8.count == 6, utf8[0] == UInt8(ascii: "<"), utf8[1] == UInt8(ascii: "0"), utf8[2] == UInt8(ascii: "x"),
              utf8[5] == UInt8(ascii: ">") else { return nil }
        let hex = String(decoding: utf8[3...4], as: UTF8.self)
        return UInt8(hex, radix: 16)
    }
}

extension String {
    /// 正しい UTF-8 のときだけ文字列にする（不正なら nil）。
    init?(validatingUTF8Bytes bytes: [UInt8]) {
        var iterator = bytes.makeIterator()
        var decoder = UTF8()
        var view = String.UnicodeScalarView()
        loop: while true {
            switch decoder.decode(&iterator) {
            case .scalarValue(let scalar): view.append(scalar)
            case .emptyInput: break loop
            case .error: return nil
            }
        }
        self = String(view)
    }
}

// MARK: - 後処理（post_processor）

indirect enum TokPostProcessor {
    enum Item {
        case special(String)
        case sequence
    }

    case template(single: [Item], specialTokens: [String: [Int]])
    case wrap(prefix: [Int], suffix: [Int])   // RobertaProcessing / BertProcessing
    case none
    case sequence([TokPostProcessor])

    static func parse(_ json: Any?, tokenID: (String) -> Int?) throws -> TokPostProcessor? {
        guard let object = json as? [String: Any], let type = object["type"] as? String else { return nil }
        switch type {
        case "TemplateProcessing":
            var items: [Item] = []
            for entry in object["single"] as? [[String: Any]] ?? [] {
                if let special = entry["SpecialToken"] as? [String: Any], let id = special["id"] as? String {
                    items.append(.special(id))
                } else if entry["Sequence"] != nil {
                    items.append(.sequence)
                }
            }
            var specials: [String: [Int]] = [:]
            for (key, value) in object["special_tokens"] as? [String: Any] ?? [:] {
                if let entry = value as? [String: Any], let ids = entry["ids"] as? [Int] {
                    specials[key] = ids
                }
            }
            return .template(single: items, specialTokens: specials)
        case "ByteLevel":
            return TokPostProcessor.none
        case "RobertaProcessing", "BertProcessing":
            func id(_ key: String) -> Int? {
                guard let pair = object[key] as? [Any] else { return nil }
                if let number = pair.last as? Int { return number }
                if let text = pair.first as? String { return tokenID(text) }
                return nil
            }
            return .wrap(prefix: id("cls").map { [$0] } ?? [], suffix: id("sep").map { [$0] } ?? [])
        case "Sequence":
            let items = object["processors"] as? [Any] ?? []
            return .sequence(try items.compactMap { try TokPostProcessor.parse($0, tokenID: tokenID) })
        default:
            throw TokenizerError.unsupported(L("未対応の post_processor: \(type)", "Unsupported post_processor: \(type)"))
        }
    }

    func process(_ ids: [Int], tokenID: (String) -> Int?) -> [Int] {
        switch self {
        case .none:
            return ids
        case .sequence(let items):
            return items.reduce(ids) { $1.process($0, tokenID: tokenID) }
        case .wrap(let prefix, let suffix):
            return prefix + ids + suffix
        case .template(let single, let specials):
            var result: [Int] = []
            for item in single {
                switch item {
                case .sequence:
                    result.append(contentsOf: ids)
                case .special(let name):
                    if let values = specials[name] {
                        result.append(contentsOf: values)
                    } else if let id = tokenID(name) {
                        result.append(id)
                    }
                }
            }
            return result
        }
    }
}
