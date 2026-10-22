import Foundation

/// 語彙の検索に使うキー（UTF-8 のバイト列）。
/// Swift の String は「見た目が同じ文字」を等しいとみなす（例: U+1F77 と U+03AF）ため、
/// 語彙の辞書には String ではなくバイト列を使う。
struct TokByteKey: Hashable {
    let bytes: [UInt8]

    init(_ string: String) {
        bytes = Array(string.utf8)
    }

    init(bytes: [UInt8]) {
        self.bytes = bytes
    }

    init(scalars: ArraySlice<Unicode.Scalar>) {
        var result: [UInt8] = []
        result.reserveCapacity(scalars.count * 3)
        for scalar in scalars {
            UTF8.encode(scalar) { result.append($0) }
        }
        bytes = result
    }
}

/// GPT-2 のバイトレベル BPE で使う「バイト ⇔ 表示用文字」の対応表。
enum TokByteLevel {
    /// 0〜255 のバイト → 文字
    static let byteToScalar: [Unicode.Scalar] = {
        var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var scalars = bytes
        var extra = 0
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(256 + extra)
            extra += 1
        }
        var table = [Unicode.Scalar](repeating: " ", count: 256)
        for (byte, scalar) in zip(bytes, scalars) {
            table[byte] = Unicode.Scalar(UInt32(scalar))!
        }
        return table
    }()

    /// 文字 → バイト
    static let scalarToByte: [UInt32: UInt8] = {
        var map: [UInt32: UInt8] = [:]
        for (byte, scalar) in byteToScalar.enumerated() {
            map[scalar.value] = UInt8(byte)
        }
        return map
    }()

    /// 文字列の UTF-8 バイトを表示用文字に置き換える。
    static func encode(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        for byte in text.utf8 {
            scalars.append(byteToScalar[Int(byte)])
        }
        return String(scalars)
    }
}

/// 正規表現まわりの補助。
enum TokRegex {
    /// Hugging Face（Oniguruma / Rust regex）の書き方を ICU（NSRegularExpression）向けに直す。
    /// \s は Unicode の空白（White_Space）全体を指すようにする（ICU の \s には U+000B や U+0085 が含まれないため）。
    static func translate(_ pattern: String) -> String {
        let whitespace = "\\t-\\r\\u0085\\p{Z}"
        var result = ""
        var inClass = 0
        var iterator = Array(pattern.unicodeScalars)[...]
        while let scalar = iterator.first {
            iterator = iterator.dropFirst()
            if scalar == "\\", let next = iterator.first {
                iterator = iterator.dropFirst()
                switch next {
                case "s":
                    result += inClass > 0 ? whitespace : "[" + whitespace + "]"
                case "S":
                    result += inClass > 0 ? "[^" + whitespace + "]" : "[^" + whitespace + "]"
                default:
                    result.unicodeScalars.append("\\")
                    result.unicodeScalars.append(next)
                }
                continue
            }
            if scalar == "[" { inClass += 1 }
            if scalar == "]" && inClass > 0 { inClass -= 1 }
            result.unicodeScalars.append(scalar)
        }
        return result
    }

    static func make(_ pattern: String) throws -> NSRegularExpression {
        do {
            return try NSRegularExpression(pattern: translate(pattern), options: [])
        } catch {
            throw TokenizerError.unsupported(L("正規表現を解釈できません: \(pattern)", "Unsupported regular expression: \(pattern)"))
        }
    }

    /// 文字列パターンをそのまま一致させる正規表現。
    static func literal(_ text: String) -> NSRegularExpression {
        // escapedPattern は失敗しない
        return try! NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: text), options: [])
    }
}

/// 分割のしかた（Hugging Face の SplitDelimiterBehavior）。
enum TokSplitBehavior: String {
    case removed = "Removed"
    case isolated = "Isolated"
    case mergedWithPrevious = "MergedWithPrevious"
    case mergedWithNext = "MergedWithNext"
    case contiguous = "Contiguous"
}

/// 前処理で扱う文字列の断片。isOriginalStart は「元の入力の先頭から始まるか」（Metaspace の prepend_scheme: first 用）。
struct TokPiece {
    var text: String
    var isOriginalStart: Bool
}

enum TokSplitter {
    /// 区間（UTF-16 オフセット）と「一致した部分か」の列から、HF と同じ規則で断片を作る。
    static func apply(_ spans: [(Range<Int>, Bool)], behavior: TokSplitBehavior, invert: Bool,
                      in piece: TokPiece) -> [TokPiece] {
        let ns = piece.text as NSString
        var marked = spans
        if invert {
            marked = marked.map { ($0.0, !$0.1) }
        }
        var output: [(Range<Int>, Bool)] = []   // (区間, 取り除くか)
        switch behavior {
        case .isolated:
            output = marked.map { ($0.0, false) }
        case .removed:
            output = marked
        case .mergedWithPrevious:
            var previousMatch = false
            for (range, isMatch) in marked {
                if isMatch && !previousMatch, let last = output.last {
                    output[output.count - 1] = (last.0.lowerBound..<range.upperBound, false)
                } else {
                    output.append((range, false))
                }
                previousMatch = isMatch
            }
        case .mergedWithNext:
            var previousMatch = false
            var reversed: [(Range<Int>, Bool)] = []
            for (range, isMatch) in marked.reversed() {
                if isMatch && !previousMatch, let last = reversed.last {
                    reversed[reversed.count - 1] = (range.lowerBound..<last.0.upperBound, false)
                } else {
                    reversed.append((range, false))
                }
                previousMatch = isMatch
            }
            output = reversed.reversed()
        case .contiguous:
            var previousMatch = false
            for (range, isMatch) in marked {
                if isMatch == previousMatch, let last = output.last {
                    output[output.count - 1] = (last.0.lowerBound..<range.upperBound, false)
                } else {
                    output.append((range, false))
                }
                previousMatch = isMatch
            }
        }
        var pieces: [TokPiece] = []
        for (range, remove) in output where !remove && !range.isEmpty {
            let text = ns.substring(with: NSRange(location: range.lowerBound, length: range.count))
            pieces.append(TokPiece(text: text, isOriginalStart: piece.isOriginalStart && range.lowerBound == 0))
        }
        return pieces
    }

    /// 正規表現の一致から区間の列を作る。
    static func spans(regex: NSRegularExpression, in text: String) -> [(Range<Int>, Bool)] {
        let ns = text as NSString
        let length = ns.length
        if length == 0 { return [(0..<0, false)] }
        var result: [(Range<Int>, Bool)] = []
        var previous = 0
        for match in regex.matches(in: text, options: [], range: NSRange(location: 0, length: length)) {
            let range = match.range
            if range.length == 0 { continue }
            if previous != range.location {
                result.append((previous..<range.location, false))
            }
            result.append((range.location..<(range.location + range.length), true))
            previous = range.location + range.length
        }
        if previous != length {
            result.append((previous..<length, false))
        }
        return result
    }

    /// 1 文字ずつ条件を調べて区間の列を作る（一致した文字はそれぞれ別の区間）。
    static func spans(in text: String, where predicate: (Unicode.Scalar) -> Bool) -> [(Range<Int>, Bool)] {
        if text.isEmpty { return [(0..<0, false)] }
        var result: [(Range<Int>, Bool)] = []
        var offset = 0
        var previous = 0
        for scalar in text.unicodeScalars {
            let width = scalar.utf16.count
            if predicate(scalar) {
                if previous != offset { result.append((previous..<offset, false)) }
                result.append((offset..<(offset + width), true))
                previous = offset + width
            }
            offset += width
        }
        if previous != offset { result.append((previous..<offset, false)) }
        return result
    }
}

extension Unicode.Scalar {
    /// Rust の char::is_whitespace と同じ（Unicode の White_Space）
    var tokIsWhitespace: Bool { properties.isWhitespace }

    /// Rust の char::is_numeric と同じ（一般カテゴリ Nd / Nl / No）
    var tokIsNumeric: Bool {
        switch properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }

    /// Rust の char::is_alphanumeric と同じ
    var tokIsAlphanumeric: Bool { properties.isAlphabetic || tokIsNumeric }

    /// Hugging Face の is_punc（ASCII の記号 + Unicode の P* カテゴリ）
    var tokIsPunctuation: Bool {
        if isASCII {
            let v = value
            if (33...47).contains(v) || (58...64).contains(v) || (91...96).contains(v) || (123...126).contains(v) { return true }
        }
        switch properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }
}
