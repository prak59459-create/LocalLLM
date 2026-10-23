import Foundation

enum JinjaError: Error, LocalizedError {
    case syntax(String)
    case runtime(String)
    /// テンプレート内の raise_exception("...")
    case raised(String)

    var errorDescription: String? {
        switch self {
        case .syntax(let message): return L("テンプレートの文法エラー: \(message)", "Template syntax error: \(message)")
        case .runtime(let message): return L("テンプレートの実行エラー: \(message)", "Template error: \(message)")
        case .raised(let message): return L("テンプレートがエラーを返しました: \(message)", "The template raised an error: \(message)")
        }
    }
}

/// 挿入順を保つ辞書。
struct JinjaObject {
    private(set) var keys: [String] = []
    private var storage: [String: JinjaValue] = [:]

    init() {}

    init(_ pairs: [(String, JinjaValue)]) {
        for (key, value) in pairs { self[key] = value }
    }

    subscript(key: String) -> JinjaValue? {
        get { storage[key] }
        set {
            if let newValue {
                if storage[key] == nil { keys.append(key) }
                storage[key] = newValue
            } else if storage.removeValue(forKey: key) != nil {
                keys.removeAll { $0 == key }
            }
        }
    }

    var count: Int { keys.count }

    var pairs: [(String, JinjaValue)] {
        keys.map { ($0, storage[$0]!) }
    }
}

/// namespace() の戻り値（属性を代入できる）。
final class JinjaNamespace {
    var attributes: JinjaObject

    init(_ attributes: JinjaObject) {
        self.attributes = attributes
    }
}

/// 組み込み関数・マクロ。
struct JinjaFunction {
    typealias Body = (_ arguments: [JinjaValue], _ keywords: [(String, JinjaValue)]) throws -> JinjaValue
    var name: String
    var body: Body
}

/// Jinja の値。
indirect enum JinjaValue {
    case undefined
    case none
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JinjaValue])
    /// タプル（表示が ( ) になる以外は配列と同じ）
    case tuple([JinjaValue])
    case object(JinjaObject)
    case namespace(JinjaNamespace)
    case function(JinjaFunction)

    /// JSON 風の Swift の値から作る。
    init(any value: Any?) {
        switch value {
        case nil: self = .none
        case is NSNull: self = .none
        case let value as JinjaValue: self = value
        case let value as Bool: self = .bool(value)
        case let value as Int: self = .int(value)
        case let value as Double: self = .double(value)
        case let value as Float: self = .double(Double(value))
        case let value as String: self = .string(value)
        case let value as [Any?]: self = .array(value.map { JinjaValue(any: $0) })
        case let value as [Any]: self = .array(value.map { JinjaValue(any: $0) })
        case let value as [(String, Any?)]:
            self = .object(JinjaObject(value.map { ($0.0, JinjaValue(any: $0.1)) }))
        case let value as [String: Any?]:
            self = .object(JinjaObject(value.keys.sorted().map { ($0, JinjaValue(any: value[$0] ?? nil)) }))
        case let value as [String: Any]:
            self = .object(JinjaObject(value.keys.sorted().map { ($0, JinjaValue(any: value[$0])) }))
        default:
            self = .string("\(value!)")
        }
    }

    var isUndefined: Bool {
        if case .undefined = self { return true }
        return false
    }

    var isNone: Bool {
        if case .none = self { return true }
        return false
    }

    /// Python の bool() と同じ真偽値。
    var truthy: Bool {
        switch self {
        case .undefined, .none: return false
        case .bool(let value): return value
        case .int(let value): return value != 0
        case .double(let value): return value != 0
        case .string(let value): return !value.isEmpty
        case .array(let value), .tuple(let value): return !value.isEmpty
        case .object(let value): return value.count > 0
        case .namespace, .function: return true
        }
    }

    /// 配列として扱える場合の要素（文字列は 1 文字ずつ、辞書はキー）。
    var iterationItems: [JinjaValue]? {
        switch self {
        case .array(let items), .tuple(let items): return items
        case .string(let text): return text.map { .string(String($0)) }
        case .object(let object): return object.keys.map { .string($0) }
        case .undefined: return []
        default: return nil
        }
    }

    var typeName: String {
        switch self {
        case .undefined: return "undefined"
        case .none: return "NoneType"
        case .bool: return "bool"
        case .int: return "int"
        case .double: return "float"
        case .string: return "str"
        case .array: return "list"
        case .tuple: return "tuple"
        case .object: return "dict"
        case .namespace: return "Namespace"
        case .function: return "function"
        }
    }

    // MARK: - 文字列化（Python の str() / repr() と同じ見た目）

    /// {{ value }} で出力される文字列。
    var renderedString: String {
        switch self {
        case .undefined: return ""
        case .string(let text): return text
        default: return pythonStr
        }
    }

    var pythonStr: String {
        switch self {
        case .undefined: return ""
        case .none: return "None"
        case .bool(let value): return value ? "True" : "False"
        case .int(let value): return String(value)
        case .double(let value): return JinjaValue.formatDouble(value)
        case .string(let text): return text
        case .array, .tuple, .object: return pythonRepr
        case .namespace: return "<Namespace>"
        case .function(let function): return "<function \(function.name)>"
        }
    }

    var pythonRepr: String {
        switch self {
        case .string(let text): return JinjaValue.reprString(text)
        case .array(let items): return "[" + items.map(\.pythonRepr).joined(separator: ", ") + "]"
        case .tuple(let items):
            if items.count == 1 { return "(" + items[0].pythonRepr + ",)" }
            return "(" + items.map(\.pythonRepr).joined(separator: ", ") + ")"
        case .object(let object):
            return "{" + object.pairs.map { JinjaValue.reprString($0.0) + ": " + $0.1.pythonRepr }.joined(separator: ", ") + "}"
        case .undefined: return ""
        default: return pythonStr
        }
    }

    static func formatDouble(_ value: Double) -> String {
        if value.isNaN { return "nan" }
        if value.isInfinite { return value > 0 ? "inf" : "-inf" }
        var text = "\(value)"
        if !text.contains(".") && !text.contains("e") && !text.contains("n") { text += ".0" }
        return text
    }

    /// Python の repr(str)
    static func reprString(_ text: String) -> String {
        let useDouble = text.contains("'") && !text.contains("\"")
        let quote: Character = useDouble ? "\"" : "'"
        var result = String(quote)
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case "'" where !useDouble: result += "\\'"
            default:
                if scalar.value < 0x20 || scalar.value == 0x7F {
                    result += String(format: "\\x%02x", scalar.value)
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        result.append(quote)
        return result
    }

    // MARK: - JSON（transformers の tojson と同じ: json.dumps(ensure_ascii=False)）

    func toJSON(indent: Int?, level: Int = 0, ensureASCII: Bool = false) throws -> String {
        let itemSeparator = indent == nil ? ", " : ","
        func newline(_ depth: Int) -> String {
            guard let indent else { return "" }
            return "\n" + String(repeating: " ", count: indent * depth)
        }
        switch self {
        case .undefined, .none: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .double(let value):
            if value.isNaN { return "NaN" }
            if value.isInfinite { return value > 0 ? "Infinity" : "-Infinity" }
            return JinjaValue.formatDouble(value)
        case .string(let text): return JinjaValue.jsonString(text, ensureASCII: ensureASCII)
        case .array(let items), .tuple(let items):
            if items.isEmpty { return "[]" }
            let parts = try items.map { newline(level + 1) + (try $0.toJSON(indent: indent, level: level + 1, ensureASCII: ensureASCII)) }
            return "[" + parts.joined(separator: itemSeparator) + newline(level) + "]"
        case .object(let object):
            if object.count == 0 { return "{}" }
            let parts = try object.pairs.map { key, value in
                newline(level + 1) + JinjaValue.jsonString(key, ensureASCII: ensureASCII) + ": "
                    + (try value.toJSON(indent: indent, level: level + 1, ensureASCII: ensureASCII))
            }
            return "{" + parts.joined(separator: itemSeparator) + newline(level) + "}"
        case .namespace(let namespace):
            return try JinjaValue.object(namespace.attributes).toJSON(indent: indent, level: level, ensureASCII: ensureASCII)
        case .function:
            throw JinjaError.runtime("Object of type function is not JSON serializable")
        }
    }

    static func jsonString(_ text: String, ensureASCII: Bool) -> String {
        var result = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case "\u{08}": result += "\\b"
            case "\u{0C}": result += "\\f"
            default:
                if scalar.value < 0x20 || (ensureASCII && scalar.value > 0x7E) {
                    if scalar.value > 0xFFFF {
                        let value = scalar.value - 0x10000
                        result += String(format: "\\u%04x\\u%04x", 0xD800 + (value >> 10), 0xDC00 + (value & 0x3FF))
                    } else {
                        result += String(format: "\\u%04x", scalar.value)
                    }
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        return result + "\""
    }

    // MARK: - 比較

    /// Python の == と同じ（1 == 1.0 は真、True == 1 も真）。
    func isEqual(to other: JinjaValue) -> Bool {
        switch (self, other) {
        case (.undefined, .undefined), (.none, .none): return true
        case (.string(let a), .string(let b)): return a.unicodeScalars.elementsEqual(b.unicodeScalars)
        case (.array(let a), .array(let b)), (.tuple(let a), .tuple(let b)):
            return a.count == b.count && zip(a, b).allSatisfy { $0.isEqual(to: $1) }
        case (.object(let a), .object(let b)):
            guard a.count == b.count else { return false }
            for key in a.keys {
                guard let left = a[key], let right = b[key], left.isEqual(to: right) else { return false }
            }
            return true
        case (.namespace(let a), .namespace(let b)): return a === b
        default:
            if let a = numericValue, let b = other.numericValue { return a == b }
            return false
        }
    }

    /// 数として扱える値（bool も 0/1）
    var numericValue: Double? {
        switch self {
        case .bool(let value): return value ? 1 : 0
        case .int(let value): return Double(value)
        case .double(let value): return value
        default: return nil
        }
    }

    func compare(_ other: JinjaValue) throws -> ComparisonResult {
        if let a = numericValue, let b = other.numericValue {
            return a < b ? .orderedAscending : (a > b ? .orderedDescending : .orderedSame)
        }
        switch (self, other) {
        case (.string(let a), .string(let b)):
            let left = Array(a.unicodeScalars).map(\.value)
            let right = Array(b.unicodeScalars).map(\.value)
            if left == right { return .orderedSame }
            return left.lexicographicallyPrecedes(right) ? .orderedAscending : .orderedDescending
        case (.array(let a), .array(let b)), (.tuple(let a), .tuple(let b)):
            for (left, right) in zip(a, b) {
                let result = try left.compare(right)
                if result != .orderedSame { return result }
            }
            return a.count < b.count ? .orderedAscending : (a.count > b.count ? .orderedDescending : .orderedSame)
        default:
            throw JinjaError.runtime("'<' not supported between instances of '\(typeName)' and '\(other.typeName)'")
        }
    }
}
