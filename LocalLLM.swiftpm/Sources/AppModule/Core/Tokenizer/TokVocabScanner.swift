import Foundation

/// tokenizer.json の model.vocab だけを、元のバイト列から直接読む小さな JSON 読み取り器。
///
/// JSONSerialization で [String: Any] にすると、Swift の String は正規等価な文字列（例: "ά" U+03AC と U+1F71、
/// "年" U+5E74 と互換漢字 U+F98E）を同じキーとみなすので、語彙の一部が消えてしまう。
/// そこで vocab のキーはバイト列のまま取り出す。
enum TokVocabScanner {
    /// model.vocab の (トークンの UTF-8 バイト列, ID) の一覧。見つからなければ nil。
    static func scanVocab(_ data: Data) -> [([UInt8], Int)]? {
        data.withUnsafeBytes { raw -> [([UInt8], Int)]? in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return nil }
            var scanner = Scanner(bytes: base, count: raw.count)
            return scanner.findModelVocab()
        }
    }

    private struct Scanner {
        let bytes: UnsafePointer<UInt8>
        let count: Int
        var index = 0

        init(bytes: UnsafePointer<UInt8>, count: Int) {
            self.bytes = bytes
            self.count = count
        }

        mutating func findModelVocab() -> [([UInt8], Int)]? {
            skipWhitespace()
            guard peek() == UInt8(ascii: "{") else { return nil }
            index += 1
            // 一番外側のオブジェクトから "model" を探す
            while true {
                skipWhitespace()
                if peek() == UInt8(ascii: "}") { return nil }
                guard let key = readString() else { return nil }
                skipWhitespace()
                guard peek() == UInt8(ascii: ":") else { return nil }
                index += 1
                skipWhitespace()
                if key == Array("model".utf8) {
                    return findVocab()
                }
                guard skipValue() else { return nil }
                skipWhitespace()
                if peek() == UInt8(ascii: ",") { index += 1 } else { return nil }
            }
        }

        mutating func findVocab() -> [([UInt8], Int)]? {
            guard peek() == UInt8(ascii: "{") else { return nil }
            index += 1
            while true {
                skipWhitespace()
                if peek() == UInt8(ascii: "}") { return nil }
                guard let key = readString() else { return nil }
                skipWhitespace()
                guard peek() == UInt8(ascii: ":") else { return nil }
                index += 1
                skipWhitespace()
                if key == Array("vocab".utf8) {
                    return readVocabObject()
                }
                guard skipValue() else { return nil }
                skipWhitespace()
                if peek() == UInt8(ascii: ",") { index += 1 } else { return nil }
            }
        }

        mutating func readVocabObject() -> [([UInt8], Int)]? {
            guard peek() == UInt8(ascii: "{") else { return nil }
            index += 1
            var entries: [([UInt8], Int)] = []
            skipWhitespace()
            if peek() == UInt8(ascii: "}") { return entries }
            while true {
                skipWhitespace()
                guard let key = readString() else { return nil }
                skipWhitespace()
                guard peek() == UInt8(ascii: ":") else { return nil }
                index += 1
                skipWhitespace()
                guard let number = readInteger() else { return nil }
                entries.append((key, number))
                skipWhitespace()
                let next = peek()
                if next == UInt8(ascii: ",") {
                    index += 1
                } else if next == UInt8(ascii: "}") {
                    index += 1
                    return entries
                } else {
                    return nil
                }
            }
        }

        // MARK: - 基本の読み取り

        func peek() -> UInt8? {
            index < count ? bytes[index] : nil
        }

        mutating func skipWhitespace() {
            while index < count {
                let byte = bytes[index]
                if byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09 { index += 1 } else { break }
            }
        }

        mutating func readInteger() -> Int? {
            var negative = false
            if peek() == UInt8(ascii: "-") {
                negative = true
                index += 1
            }
            var value = 0
            var digits = 0
            while index < count {
                let byte = bytes[index]
                guard byte >= 0x30 && byte <= 0x39 else { break }
                value = value * 10 + Int(byte - 0x30)
                digits += 1
                index += 1
            }
            guard digits > 0 else { return nil }
            return negative ? -value : value
        }

        /// 文字列を読んで、エスケープを戻した UTF-8 バイト列を返す。
        mutating func readString() -> [UInt8]? {
            guard peek() == UInt8(ascii: "\"") else { return nil }
            index += 1
            var result: [UInt8] = []
            while index < count {
                let byte = bytes[index]
                index += 1
                if byte == UInt8(ascii: "\"") { return result }
                if byte != UInt8(ascii: "\\") {
                    result.append(byte)
                    continue
                }
                guard index < count else { return nil }
                let escape = bytes[index]
                index += 1
                switch escape {
                case UInt8(ascii: "\""): result.append(UInt8(ascii: "\""))
                case UInt8(ascii: "\\"): result.append(UInt8(ascii: "\\"))
                case UInt8(ascii: "/"): result.append(UInt8(ascii: "/"))
                case UInt8(ascii: "b"): result.append(0x08)
                case UInt8(ascii: "f"): result.append(0x0C)
                case UInt8(ascii: "n"): result.append(0x0A)
                case UInt8(ascii: "r"): result.append(0x0D)
                case UInt8(ascii: "t"): result.append(0x09)
                case UInt8(ascii: "u"):
                    guard var code = readHex4() else { return nil }
                    // サロゲートペア
                    if (0xD800...0xDBFF).contains(code), index + 1 < count,
                       bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u") {
                        let saved = index
                        index += 2
                        if let low = readHex4(), (0xDC00...0xDFFF).contains(low) {
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            index = saved
                        }
                    }
                    let scalar = Unicode.Scalar(code) ?? "\u{FFFD}"
                    UTF8.encode(scalar) { result.append($0) }
                default:
                    return nil
                }
            }
            return nil
        }

        mutating func readHex4() -> UInt32? {
            guard index + 4 <= count else { return nil }
            var value: UInt32 = 0
            for _ in 0..<4 {
                let byte = bytes[index]
                index += 1
                let digit: UInt32
                switch byte {
                case 0x30...0x39: digit = UInt32(byte - 0x30)
                case 0x41...0x46: digit = UInt32(byte - 0x41 + 10)
                case 0x61...0x66: digit = UInt32(byte - 0x61 + 10)
                default: return nil
                }
                value = value * 16 + digit
            }
            return value
        }

        /// 値を読み飛ばす（中身は作らない）。
        mutating func skipValue() -> Bool {
            guard let first = peek() else { return false }
            switch first {
            case UInt8(ascii: "\""):
                return skipString()
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                var depth = 0
                while index < count {
                    let byte = bytes[index]
                    if byte == UInt8(ascii: "\"") {
                        guard skipString() else { return false }
                        continue
                    }
                    index += 1
                    if byte == UInt8(ascii: "{") || byte == UInt8(ascii: "[") {
                        depth += 1
                    } else if byte == UInt8(ascii: "}") || byte == UInt8(ascii: "]") {
                        depth -= 1
                        if depth == 0 { return true }
                    }
                }
                return false
            default:
                // 数値・true / false / null
                while index < count {
                    let byte = bytes[index]
                    if byte == UInt8(ascii: ",") || byte == UInt8(ascii: "}") || byte == UInt8(ascii: "]")
                        || byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09 { break }
                    index += 1
                }
                return true
            }
        }

        mutating func skipString() -> Bool {
            index += 1   // 開きの "
            while index < count {
                let byte = bytes[index]
                index += 1
                if byte == UInt8(ascii: "\\") {
                    index += 1
                } else if byte == UInt8(ascii: "\"") {
                    return true
                }
            }
            return false
        }
    }
}
