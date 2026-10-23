import Foundation

/// キーの順番を保つ JSON の値（ツールの引数・スキーマ、MCP のやり取りに使う）。
indirect enum JSONValue: Hashable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([(String, JSONValue)])

    static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.int(let a), .int(let b)): return a == b
        case (.double(let a), .double(let b)): return a == b
        case (.int(let a), .double(let b)), (.double(let b), .int(let a)): return Double(a) == b
        case (.string(let a), .string(let b)): return a == b
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)):
            guard a.count == b.count else { return false }
            for (key, value) in a where b.first(where: { $0.0 == key })?.1 != value { return false }
            return true
        default: return false
        }
    }

    func hash(into hasher: inout Hasher) {
        switch self {
        case .null: hasher.combine(0)
        case .bool(let value): hasher.combine(value)
        case .int(let value): hasher.combine(Double(value))
        case .double(let value): hasher.combine(value)
        case .string(let value): hasher.combine(value)
        case .array(let values): hasher.combine(values)
        case .object(let pairs):
            hasher.combine(pairs.count)
            for (key, _) in pairs.sorted(by: { $0.0 < $1.0 }) { hasher.combine(key) }
        }
    }

    // MARK: - 取り出し

    subscript(key: String) -> JSONValue? {
        if case .object(let pairs) = self { return pairs.first { $0.0 == key }?.1 }
        return nil
    }

    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var intValue: Int? {
        switch self {
        case .int(let value): return value
        case .double(let value): return value.rounded() == value ? Int(value) : nil
        case .string(let value): return Int(value.trimmingCharacters(in: .whitespaces))
        default: return nil
        }
    }

    var doubleValue: Double? {
        switch self {
        case .int(let value): return Double(value)
        case .double(let value): return value
        case .string(let value): return Double(value.trimmingCharacters(in: .whitespaces))
        default: return nil
        }
    }

    var boolValue: Bool? {
        switch self {
        case .bool(let value): return value
        case .string(let value):
            switch value.lowercased() {
            case "true", "yes", "1": return true
            case "false", "no", "0": return false
            default: return nil
            }
        case .int(let value): return value != 0
        default: return nil
        }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let values) = self { return values }
        return nil
    }

    var objectValue: [(String, JSONValue)]? {
        if case .object(let pairs) = self { return pairs }
        return nil
    }

    /// 値を文字列として（文字列ならそのまま、それ以外は JSON）
    var textValue: String {
        if case .string(let value) = self { return value }
        return serialized()
    }

    // MARK: - 書き出し

    /// JSON 文字列（日本語はそのまま）
    func serialized(pretty: Bool = false) -> String {
        var output = ""
        write(into: &output, pretty: pretty, indent: 0)
        return output
    }

    private func write(into output: inout String, pretty: Bool, indent: Int) {
        switch self {
        case .null: output += "null"
        case .bool(let value): output += value ? "true" : "false"
        case .int(let value): output += String(value)
        case .double(let value):
            if value.isFinite {
                if value.rounded() == value && abs(value) < 1e15 {
                    output += String(Int(value)) + ".0"
                } else {
                    output += String(value)
                }
            } else {
                output += "null"
            }
        case .string(let value): Self.writeString(value, into: &output)
        case .array(let values):
            if values.isEmpty { output += "[]"; return }
            output += "["
            for (index, value) in values.enumerated() {
                if index > 0 { output += pretty ? "," : ", " }
                if pretty { output += "\n" + String(repeating: "  ", count: indent + 1) }
                value.write(into: &output, pretty: pretty, indent: indent + 1)
            }
            if pretty { output += "\n" + String(repeating: "  ", count: indent) }
            output += "]"
        case .object(let pairs):
            if pairs.isEmpty { output += "{}"; return }
            output += "{"
            for (index, (key, value)) in pairs.enumerated() {
                if index > 0 { output += pretty ? "," : ", " }
                if pretty { output += "\n" + String(repeating: "  ", count: indent + 1) }
                Self.writeString(key, into: &output)
                output += ": "
                value.write(into: &output, pretty: pretty, indent: indent + 1)
            }
            if pretty { output += "\n" + String(repeating: "  ", count: indent) }
            output += "}"
        }
    }

    private static func writeString(_ value: String, into output: inout String) {
        output += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            default:
                if scalar.value < 0x20 {
                    output += String(format: "\\u%04x", scalar.value)
                } else {
                    output.unicodeScalars.append(scalar)
                }
            }
        }
        output += "\""
    }

    // MARK: - Foundation との変換

    init(any value: Any?) {
        guard let value, !(value is NSNull) else {
            self = .null
            return
        }
        if let string = value as? String {
            self = .string(string)
        } else if let number = value as? NSNumber {
            // JSONSerialization の真偽値は objCType が "c"
            let type = String(cString: number.objCType)
            if type == "c" || type == "B" {
                self = .bool(number.boolValue)
            } else if ["q", "i", "l", "s", "Q", "I", "L", "S"].contains(type) {
                self = .int(number.intValue)
            } else {
                self = .double(number.doubleValue)
            }
        } else if let array = value as? [Any] {
            self = .array(array.map { JSONValue(any: $0) })
        } else if let dictionary = value as? [String: Any] {
            self = .object(dictionary.keys.sorted().map { ($0, JSONValue(any: dictionary[$0])) })
        } else {
            self = .string(String(describing: value))
        }
    }

    /// Jinja のテンプレートに渡す値
    var jinjaValue: JinjaValue {
        switch self {
        case .null: return .none
        case .bool(let value): return .bool(value)
        case .int(let value): return .int(value)
        case .double(let value): return .double(value)
        case .string(let value): return .string(value)
        case .array(let values): return .array(values.map(\.jinjaValue))
        case .object(let pairs): return .object(JinjaObject(pairs.map { ($0.0, $0.1.jinjaValue) }))
        }
    }

    // MARK: - 読み取り

    enum ParseError: Error, LocalizedError {
        case invalid(String, Int)

        var errorDescription: String? {
            switch self {
            case .invalid(let reason, let offset): return "JSON: \(reason) (at \(offset))"
            }
        }
    }

    /// JSON を読む。lenient なら、小さなモデルが書きがちな崩れ（末尾のカンマ、' の文字列、引用符の無いキー、
    /// True / None、閉じ忘れ）もできるだけ読む。
    static func parse(_ text: String, lenient: Bool = false) throws -> JSONValue {
        var parser = JSONParser(Array(text.unicodeScalars), lenient: lenient)
        parser.skipWhitespace()
        let value = try parser.parseValue()
        parser.skipWhitespace()
        if !parser.atEnd && !lenient { throw ParseError.invalid("extra characters", parser.position) }
        return value
    }

    /// 文章の中の最初の { … }（または [ … ]）を読む。前後の説明文は無視する。
    static func firstObject(in text: String) -> (value: JSONValue, range: Range<String.Index>)? {
        let scalars = Array(text.unicodeScalars)
        var start = 0
        while start < scalars.count {
            if scalars[start] == "{" || scalars[start] == "[" {
                var parser = JSONParser(scalars, lenient: true)
                parser.position = start
                if let value = try? parser.parseValue(), value.isNonEmptyContainer {
                    let base = text.unicodeScalars.startIndex
                    let lower = text.unicodeScalars.index(base, offsetBy: start)
                    let upper = text.unicodeScalars.index(base, offsetBy: parser.position)
                    return (value, lower..<upper)
                }
            }
            start += 1
        }
        return nil
    }

    var isNonEmptyContainer: Bool {
        switch self {
        case .object(let pairs): return !pairs.isEmpty
        case .array(let values): return !values.isEmpty
        default: return false
        }
    }
}

extension JSONValue: Codable {
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let text = try container.decode(String.self)
        self = (try? JSONValue.parse(text)) ?? .string(text)
    }

    /// JSON 文字列として保存する（キーの順番を保つため）
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(serialized())
    }
}

private struct JSONParser {
    let scalars: [Unicode.Scalar]
    let lenient: Bool
    var position = 0

    init(_ scalars: [Unicode.Scalar], lenient: Bool) {
        self.scalars = scalars
        self.lenient = lenient
    }

    var atEnd: Bool { position >= scalars.count }

    mutating func skipWhitespace() {
        while position < scalars.count {
            let c = scalars[position]
            if c == " " || c == "\n" || c == "\r" || c == "\t" {
                position += 1
            } else if lenient && c == "/" && position + 1 < scalars.count && scalars[position + 1] == "/" {
                while position < scalars.count && scalars[position] != "\n" { position += 1 }
            } else {
                return
            }
        }
    }

    func fail(_ reason: String) -> JSONValue.ParseError {
        .invalid(reason, position)
    }

    mutating func parseValue() throws -> JSONValue {
        skipWhitespace()
        guard position < scalars.count else { throw fail("unexpected end") }
        let c = scalars[position]
        switch c {
        case "{": return try parseObject()
        case "[": return try parseArray()
        case "\"": return .string(try parseString(quote: "\""))
        case "'" where lenient: return .string(try parseString(quote: "'"))
        case "-", "0"..."9": return try parseNumber()
        default:
            if let word = parseWord() {
                switch word {
                case "true": return .bool(true)
                case "false": return .bool(false)
                case "null": return .null
                case "True" where lenient: return .bool(true)
                case "False" where lenient: return .bool(false)
                case "None" where lenient: return .null
                default:
                    if lenient { return .string(word) }
                }
            }
            throw fail("unexpected character")
        }
    }

    mutating func parseWord() -> String? {
        let start = position
        while position < scalars.count {
            let c = scalars[position]
            if c.properties.isAlphabetic || ("0"..."9").contains(c) || c == "_" || c == "-" || c == "." {
                position += 1
            } else {
                break
            }
        }
        guard position > start else { return nil }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars[start..<position])
        return String(view)
    }

    mutating func parseObject() throws -> JSONValue {
        position += 1
        var pairs: [(String, JSONValue)] = []
        var afterComma = false
        while true {
            skipWhitespace()
            guard position < scalars.count else {
                if lenient { return .object(pairs) }
                throw fail("unterminated object")
            }
            if scalars[position] == "}" {
                if afterComma && !lenient { throw fail("trailing comma") }
                position += 1
                return .object(pairs)
            }
            let key: String
            if scalars[position] == "\"" {
                key = try parseString(quote: "\"")
            } else if lenient && scalars[position] == "'" {
                key = try parseString(quote: "'")
            } else if lenient, let word = parseWord() {
                key = word
            } else {
                throw fail("expected key")
            }
            skipWhitespace()
            guard position < scalars.count, scalars[position] == ":" else { throw fail("expected ':'") }
            position += 1
            let value = try parseValue()
            pairs.append((key, value))
            afterComma = false
            skipWhitespace()
            guard position < scalars.count else {
                if lenient { return .object(pairs) }
                throw fail("unterminated object")
            }
            if scalars[position] == "," {
                position += 1
                afterComma = true
            } else if scalars[position] != "}" {
                throw fail("expected ',' or '}'")
            }
        }
    }

    mutating func parseArray() throws -> JSONValue {
        position += 1
        var values: [JSONValue] = []
        var afterComma = false
        while true {
            skipWhitespace()
            guard position < scalars.count else {
                if lenient { return .array(values) }
                throw fail("unterminated array")
            }
            if scalars[position] == "]" {
                if afterComma && !lenient { throw fail("trailing comma") }
                position += 1
                return .array(values)
            }
            values.append(try parseValue())
            afterComma = false
            skipWhitespace()
            guard position < scalars.count else {
                if lenient { return .array(values) }
                throw fail("unterminated array")
            }
            if scalars[position] == "," {
                position += 1
                afterComma = true
            } else if scalars[position] != "]" {
                throw fail("expected ',' or ']'")
            }
        }
    }

    mutating func parseString(quote: Unicode.Scalar) throws -> String {
        position += 1
        var view = String.UnicodeScalarView()
        while position < scalars.count {
            let c = scalars[position]
            position += 1
            if c == quote { return String(view) }
            if c == "\\" {
                guard position < scalars.count else { break }
                let escaped = scalars[position]
                position += 1
                switch escaped {
                case "n": view.append("\n")
                case "t": view.append("\t")
                case "r": view.append("\r")
                case "b": view.append("\u{08}")
                case "f": view.append("\u{0C}")
                case "u":
                    guard position + 4 <= scalars.count,
                          var code = UInt32(String(String.UnicodeScalarView(scalars[position..<(position + 4)])), radix: 16) else {
                        throw fail("bad \\u escape")
                    }
                    position += 4
                    // サロゲートペア
                    if (0xD800...0xDBFF).contains(code), position + 6 <= scalars.count, scalars[position] == "\\",
                       scalars[position + 1] == "u",
                       let low = UInt32(String(String.UnicodeScalarView(scalars[(position + 2)..<(position + 6)])), radix: 16),
                       (0xDC00...0xDFFF).contains(low) {
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        position += 6
                    }
                    view.append(Unicode.Scalar(code) ?? "\u{FFFD}")
                default: view.append(escaped)
                }
            } else {
                view.append(c)
            }
        }
        if lenient { return String(view) }
        throw fail("unterminated string")
    }

    mutating func parseNumber() throws -> JSONValue {
        let start = position
        if scalars[position] == "-" { position += 1 }
        var isDouble = false
        while position < scalars.count {
            let c = scalars[position]
            if ("0"..."9").contains(c) {
                position += 1
            } else if c == "." || c == "e" || c == "E" || ((c == "+" || c == "-") && (scalars[position - 1] == "e" || scalars[position - 1] == "E")) {
                isDouble = true
                position += 1
            } else {
                break
            }
        }
        let text = String(String.UnicodeScalarView(scalars[start..<position]))
        if !isDouble, let value = Int(text) { return .int(value) }
        guard let value = Double(text) else { throw fail("bad number") }
        return .double(value)
    }
}
