import Foundation

/// モデルに使わせるツールの定義（OpenAI / Hugging Face の function calling と同じ形）。
struct ToolDefinition: Codable, Hashable, Sendable {
    var name: String
    var description: String
    /// JSON Schema（{"type": "object", "properties": {...}, "required": [...]}）
    var parameters: JSONValue

    init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    /// よくある形（文字列の引数だけ）を簡単に作る。arguments は (名前, 説明, 必須か)
    init(name: String, description: String, arguments: [(String, String, Bool)]) {
        self.name = name
        self.description = description
        var properties: [(String, JSONValue)] = []
        var required: [JSONValue] = []
        for (argument, detail, isRequired) in arguments {
            properties.append((argument, .object([("type", .string("string")), ("description", .string(detail))])))
            if isRequired { required.append(.string(argument)) }
        }
        self.parameters = .object([("type", .string("object")), ("properties", .object(properties)), ("required", .array(required))])
    }

    /// チャットテンプレートの tools に入れる形
    var templateValue: JSONValue {
        .object([
            ("type", .string("function")),
            ("function", .object([
                ("name", .string(name)),
                ("description", .string(description)),
                ("parameters", parameters),
            ])),
        ])
    }

    /// 人が読む引数の説明（ツールに対応していないテンプレート用）
    var argumentSummary: String {
        guard let properties = parameters["properties"]?.objectValue, !properties.isEmpty else { return L("なし", "none") }
        let required = Set(parameters["required"]?.arrayValue?.compactMap(\.stringValue) ?? [])
        return properties.map { name, schema in
            let type = schema["type"]?.stringValue ?? "string"
            let detail = schema["description"]?.stringValue.map { " " + $0 } ?? ""
            return "\(name) (\(type)\(required.contains(name) ? L("・必須", ", required") : ""))\(detail)"
        }.joined(separator: "; ")
    }
}

/// モデルが書いたツールの呼び出し。
struct ToolCall: Codable, Hashable, Sendable, Identifiable {
    var id: String
    var name: String
    var arguments: JSONValue

    init(id: String = ToolCall.makeID(), name: String, arguments: JSONValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// 9 文字の英数字（Mistral のテンプレートがこの長さを求める）
    static func makeID() -> String {
        let characters = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        return String((0..<9).map { _ in characters[Int.random(in: 0..<characters.count)] })
    }

    func argument(_ key: String) -> String? {
        arguments[key]?.textValue
    }

    /// テンプレートに渡す形
    var templateValue: JSONValue {
        .object([
            ("id", .string(id)),
            ("type", .string("function")),
            ("function", .object([("name", .string(name)), ("arguments", arguments)])),
        ])
    }

    /// 画面に出す短い説明（例: web_search(query: "東京 天気")）
    var summary: String {
        guard let pairs = arguments.objectValue, !pairs.isEmpty else { return "\(name)()" }
        let parts = pairs.map { key, value -> String in
            var text = value.textValue
            if text.count > 60 { text = String(text.prefix(60)) + "…" }
            return "\(key): \(text)"
        }
        return "\(name)(\(parts.joined(separator: ", ")))"
    }
}

/// モデルの出力からツールの呼び出しを取り出す。いろいろなモデルの書き方に対応する:
/// - Qwen / Hermes: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
/// - Llama 3.x: `{"name": …, "parameters": {…}}`（`<|python_tag|>` が前に付くこともある）
/// - Mistral: `[TOOL_CALLS] [{"name": …, "arguments": {…}}]`
/// - Llama 3.1 の別形式: `<function=名前>{…}</function>`
/// - コードブロック: ```json {"name": …, "arguments": …} ```
enum ToolCallParser {
    struct Result: Equatable {
        /// 呼び出し以外の本文
        var content: String
        var calls: [ToolCall]
    }

    static let markers = ["<tool_call>", "<function=", "<|python_tag|>", "[TOOL_CALLS]"]

    /// knownTools にある名前の呼び出しだけを取り出す（関係ない JSON を誤って呼び出しにしないため）。
    static func parse(_ text: String, knownTools: Set<String>) -> Result {
        var content = text
        var calls: [ToolCall] = []

        // 1. <tool_call> … </tool_call>
        while let open = content.range(of: "<tool_call>") {
            let after = content[open.upperBound...]
            let close = after.range(of: "</tool_call>")
            let inner = String(close.map { after[..<$0.lowerBound] } ?? after)
            calls.append(contentsOf: interpret(inner, knownTools: knownTools))
            content.removeSubrange(open.lowerBound..<(close?.upperBound ?? content.endIndex))
        }
        // 2. <function=名前>{…}</function>
        while let open = content.range(of: "<function=") {
            let after = content[open.upperBound...]
            guard let nameEnd = after.firstIndex(of: ">") else { break }
            let name = String(after[..<nameEnd])
            let body = after[after.index(after: nameEnd)...]
            let close = body.range(of: "</function>")
            let inner = String(close.map { body[..<$0.lowerBound] } ?? body)
            if knownTools.contains(name) {
                let arguments = (try? JSONValue.parse(inner, lenient: true)) ?? .object([])
                calls.append(ToolCall(name: name, arguments: arguments))
            }
            content.removeSubrange(open.lowerBound..<(close?.upperBound ?? content.endIndex))
        }
        if !calls.isEmpty { return Result(content: tidy(content), calls: calls) }

        // 3. 本文全体が JSON（Llama / Mistral）
        var trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["<|python_tag|>", "[TOOL_CALLS]"] where trimmed.hasPrefix(prefix) {
            trimmed = String(trimmed.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if trimmed.hasPrefix("```") {
            // ```json … ```
            var inner = trimmed.drop { $0 != "\n" }
            if let end = inner.range(of: "```", options: .backwards) { inner = inner[..<end.lowerBound] }
            let found = interpret(String(inner), knownTools: knownTools)
            if !found.isEmpty { return Result(content: "", calls: found) }
        }
        if trimmed.hasPrefix("{") || trimmed.hasPrefix("[") {
            let found = interpret(trimmed, knownTools: knownTools)
            if !found.isEmpty { return Result(content: "", calls: found) }
        }

        // 4. 文章の途中のコードブロックや JSON（「検索します: {"name": …}」のような書き方）
        if let (value, range) = JSONValue.firstObject(in: content) {
            let found = interpret(value, knownTools: knownTools)
            if !found.isEmpty {
                var rest = content
                rest.removeSubrange(range)
                rest = rest.replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
                return Result(content: tidy(rest), calls: found)
            }
        }
        return Result(content: text, calls: [])
    }

    /// 生成中に画面に出す部分（ツールの呼び出しが始まったら、そこから先は隠す）
    static func visiblePart(of text: String) -> String {
        var cut = text.endIndex
        for marker in markers {
            if let range = text.range(of: marker), range.lowerBound < cut { cut = range.lowerBound }
        }
        var visible = String(text[..<cut])
        // 本文が JSON で始まるときも（Llama の書き方）隠す
        let trimmed = visible.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{\"name\"") || trimmed.hasPrefix("[{\"name\"") || trimmed.hasPrefix("{\"type\": \"function\"") {
            visible = ""
        }
        return visible
    }

    private static func interpret(_ text: String, knownTools: Set<String>) -> [ToolCall] {
        if let value = try? JSONValue.parse(text.trimmingCharacters(in: .whitespacesAndNewlines), lenient: true) {
            let found = interpret(value, knownTools: knownTools)
            if !found.isEmpty { return found }
        }
        if let (value, _) = JSONValue.firstObject(in: text) {
            return interpret(value, knownTools: knownTools)
        }
        return []
    }

    private static func interpret(_ value: JSONValue, knownTools: Set<String>) -> [ToolCall] {
        if case .array(let items) = value {
            return items.flatMap { interpret($0, knownTools: knownTools) }
        }
        guard case .object = value else { return [] }
        // {"tool_calls": [...]} / {"function": {...}}
        if let list = value["tool_calls"]?.arrayValue {
            return list.flatMap { interpret($0, knownTools: knownTools) }
        }
        let function = value["function"]
        let name = value["name"]?.stringValue ?? function?["name"]?.stringValue ?? value["tool"]?.stringValue
            ?? value["tool_name"]?.stringValue
        guard let name, knownTools.contains(name) else { return [] }
        var arguments = value["arguments"] ?? value["parameters"] ?? value["args"] ?? value["input"]
            ?? function?["arguments"] ?? function?["parameters"] ?? .object([])
        if case .string(let text) = arguments {
            arguments = (try? JSONValue.parse(text, lenient: true)) ?? .object([("input", .string(text))])
        }
        if case .object = arguments {} else { arguments = .object([("input", arguments)]) }
        let id = value["id"]?.stringValue ?? ToolCall.makeID()
        return [ToolCall(id: id, name: name, arguments: arguments)]
    }

    private static func tidy(_ text: String) -> String {
        text.replacingOccurrences(of: "<|python_tag|>", with: "")
            .replacingOccurrences(of: "[TOOL_CALLS]", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// ツールに対応していないテンプレート・書式のための、プロンプトでの代用。
enum ToolPrompt {
    /// システムプロンプトに付け足す説明
    static func instructions(for tools: [ToolDefinition]) -> String {
        var lines = [
            L("あなたは次のツールを使えます。ツールが必要なときは、本文を書かずに、次の形式だけを出力してください。",
              "You can use the following tools. When you need one, output only the following, with no other text:"),
            "<tool_call>",
            "{\"name\": \"" + L("ツール名", "tool_name") + "\", \"arguments\": {\"" + L("引数", "argument") + "\": \"" + L("値", "value") + "\"}}",
            "</tool_call>",
            L("ツールの結果は <tool_response> で渡されます。結果を受け取ったら、それをもとに答えてください。ツールが要らないときは、そのまま答えてください。",
              "Tool results come back in <tool_response>. Use them to answer. If no tool is needed, just answer."),
            "",
            L("ツール:", "Tools:"),
        ]
        for tool in tools {
            lines.append("- \(tool.name): \(tool.description) " + L("引数: ", "Arguments: ") + tool.argumentSummary)
        }
        return lines.joined(separator: "\n")
    }

    /// tool の役割やツール呼び出しを、普通の user / assistant の文章に書き直す
    static func flatten(_ messages: [ChatMessage]) -> [ChatMessage] {
        messages.map { message in
            switch message.role {
            case .tool:
                let name = message.toolName.map { " name=\"\($0)\"" } ?? ""
                return ChatMessage(role: .user, content: "<tool_response\(name)>\n\(message.content)\n</tool_response>")
            case .assistant:
                guard let calls = message.toolCalls, !calls.isEmpty else { return message }
                var text = message.content.isEmpty ? "" : message.content + "\n"
                for call in calls {
                    let body = JSONValue.object([("name", .string(call.name)), ("arguments", call.arguments)]).serialized()
                    text += "<tool_call>\n\(body)\n</tool_call>\n"
                }
                return ChatMessage(role: .assistant, content: text.trimmingCharacters(in: .newlines))
            default:
                return message
            }
        }
    }
}
