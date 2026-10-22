import Foundation

/// チャットのメッセージ 1 つ。
struct ChatMessage: Codable, Hashable, Sendable {
    enum Role: String, Codable, Sendable {
        case system, user, assistant
    }

    var role: Role
    var content: String

    init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// テンプレートが無い・描画できないときに使う組み込みの書式。
enum BuiltinChatFormat: String, Codable, CaseIterable, Sendable {
    case chatML
    case llama3
    case gemma
    case mistral
    case phi3
    case deepseek
    case exaone
    case plain

    /// テンプレート文字列や語彙に含まれるトークンから推定する。
    static func detect(template: String?, hasToken: (String) -> Bool) -> BuiltinChatFormat {
        let markers: [(String, BuiltinChatFormat)] = [
            ("<|im_start|>", .chatML),
            ("<|start_header_id|>", .llama3),
            ("<start_of_turn>", .gemma),
            ("<｜User｜>", .deepseek),
            ("[|user|]", .exaone),
            ("<|user|>", .phi3),
            ("[INST]", .mistral),
        ]
        if let template {
            for (marker, format) in markers where template.contains(marker) { return format }
        }
        for (marker, format) in markers where hasToken(marker) { return format }
        return .plain
    }

    /// このフォーマットでの「ターン終了」の目印
    var endOfTurnTokens: [String] {
        switch self {
        case .chatML: return ["<|im_end|>"]
        case .llama3: return ["<|eot_id|>", "<|eom_id|>"]
        case .gemma: return ["<end_of_turn>"]
        case .mistral: return ["</s>"]
        case .phi3: return ["<|end|>"]
        case .deepseek: return ["<｜end▁of▁sentence｜>"]
        case .exaone: return ["[|endofturn|]"]
        case .plain: return []
        }
    }

    func render(messages: [ChatMessage], addGenerationPrompt: Bool, bosToken: String?, eosToken: String?) -> String {
        var output = ""
        switch self {
        case .chatML:
            for message in messages {
                output += "<|im_start|>\(message.role.rawValue)\n\(message.content)<|im_end|>\n"
            }
            if addGenerationPrompt { output += "<|im_start|>assistant\n" }
        case .llama3:
            output += bosToken ?? "<|begin_of_text|>"
            for message in messages {
                output += "<|start_header_id|>\(message.role.rawValue)<|end_header_id|>\n\n\(message.content)<|eot_id|>"
            }
            if addGenerationPrompt { output += "<|start_header_id|>assistant<|end_header_id|>\n\n" }
        case .gemma:
            output += bosToken ?? "<bos>"
            for message in BuiltinChatFormat.mergingSystemIntoFirstUser(messages) {
                let role = message.role == .assistant ? "model" : "user"
                output += "<start_of_turn>\(role)\n\(message.content)<end_of_turn>\n"
            }
            if addGenerationPrompt { output += "<start_of_turn>model\n" }
        case .mistral:
            output += bosToken ?? "<s>"
            for message in BuiltinChatFormat.mergingSystemIntoFirstUser(messages) {
                if message.role == .user {
                    output += "[INST] \(message.content)[/INST]"
                } else {
                    output += " \(message.content)\(eosToken ?? "</s>")"
                }
            }
        case .phi3:
            for message in messages {
                output += "<|\(message.role.rawValue)|>\n\(message.content)<|end|>\n"
            }
            if addGenerationPrompt { output += "<|assistant|>\n" }
        case .deepseek:
            output += bosToken ?? "<｜begin▁of▁sentence｜>"
            for message in messages {
                switch message.role {
                case .system: output += message.content
                case .user: output += "<｜User｜>\(message.content)"
                case .assistant: output += "<｜Assistant｜>\(message.content)\(eosToken ?? "<｜end▁of▁sentence｜>")"
                }
            }
            if addGenerationPrompt { output += "<｜Assistant｜><think>\n" }
        case .exaone:
            for message in messages {
                output += "[|\(message.role.rawValue)|]\(message.content)"
                output += message.role == .user ? "\n" : "[|endofturn|]\n"
            }
            if addGenerationPrompt { output += "[|assistant|]" }
        case .plain:
            for message in messages {
                switch message.role {
                case .system: output += "\(message.content)\n\n"
                case .user: output += "User: \(message.content)\n"
                case .assistant: output += "Assistant: \(message.content)\n"
                }
            }
            if addGenerationPrompt { output += "Assistant:" }
        }
        return output
    }

    /// system を最初の user にまとめる（system を受け付けない書式用）
    static func mergingSystemIntoFirstUser(_ messages: [ChatMessage]) -> [ChatMessage] {
        guard let first = messages.first, first.role == .system else { return messages }
        var rest = Array(messages.dropFirst())
        if let index = rest.firstIndex(where: { $0.role == .user }) {
            rest[index].content = first.content + "\n\n" + rest[index].content
            return rest
        }
        return [ChatMessage(role: .user, content: first.content)] + rest
    }
}

/// チャットテンプレートの描画。Jinja で描画できなければ組み込みの書式にする。
struct ChatTemplateRenderer {
    let template: String?
    let bosToken: String?
    let eosToken: String?
    let fallbackFormat: BuiltinChatFormat
    private let compiled: JinjaTemplate?

    init(template: String?, bosToken: String?, eosToken: String?, fallbackFormat: BuiltinChatFormat? = nil) {
        self.template = template
        self.bosToken = bosToken
        self.eosToken = eosToken
        self.fallbackFormat = fallbackFormat ?? BuiltinChatFormat.detect(template: template, hasToken: { _ in false })
        if let template, !template.isEmpty {
            compiled = try? JinjaTemplate(template)
        } else {
            compiled = nil
        }
    }

    /// 実際に Jinja のテンプレートを使うか（false なら組み込みの書式）
    var usesJinja: Bool { compiled != nil }

    /// テンプレートから推定した「ターン終了」トークン
    var endOfTurnTokens: [String] {
        var tokens: [String] = []
        let source = template ?? ""
        let candidates = ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|end|>", "[|endofturn|]",
                          "<｜end▁of▁sentence｜>", "<|endoftext|>", "<|end_of_text|>"]
        for candidate in candidates where source.contains(candidate) { tokens.append(candidate) }
        for candidate in fallbackFormat.endOfTurnTokens where !tokens.contains(candidate) && (compiled == nil || source.contains(candidate)) {
            tokens.append(candidate)
        }
        return tokens
    }

    /// - Parameters:
    ///   - enableThinking: テンプレート変数 enable_thinking（nil なら渡さない）
    ///   - extraContext: 追加の変数（テスト用の date_string など）
    func render(messages: [ChatMessage], addGenerationPrompt: Bool, enableThinking: Bool?,
                extraContext: [String: JinjaValue] = [:]) throws -> String {
        guard let compiled else {
            return renderBuiltin(messages: messages, addGenerationPrompt: addGenerationPrompt)
        }
        var context: [String: JinjaValue] = [
            "messages": .array(messages.map { message in
                .object(JinjaObject([("role", .string(message.role.rawValue)), ("content", .string(message.content))]))
            }),
            "add_generation_prompt": .bool(addGenerationPrompt),
            "tools": .none,
            "documents": .none,
        ]
        if let bosToken { context["bos_token"] = .string(bosToken) }
        if let eosToken { context["eos_token"] = .string(eosToken) }
        if let enableThinking { context["enable_thinking"] = .bool(enableThinking) }
        for (key, value) in extraContext { context[key] = value }
        do {
            return try compiled.render(context)
        } catch let error as JinjaError {
            if case .raised = error { throw error }
            // テンプレートが未対応の機能を使っていた → 組み込みの書式で代用
            return renderBuiltin(messages: messages, addGenerationPrompt: addGenerationPrompt)
        }
    }

    func renderBuiltin(messages: [ChatMessage], addGenerationPrompt: Bool) -> String {
        fallbackFormat.render(messages: messages, addGenerationPrompt: addGenerationPrompt, bosToken: bosToken, eosToken: eosToken)
    }
}

/// 生成テキストから <think>…</think> の思考部分を分ける。
struct ThinkingSplit: Equatable {
    var thinking: String?
    var answer: String
    var isThinkingOpen: Bool

    static let openTag = "<think>"
    static let closeTag = "</think>"

    static func split(_ text: String) -> ThinkingSplit {
        let ns = text as NSString
        let open = ns.range(of: openTag, options: .literal)
        if open.location != NSNotFound {
            let before = ns.substring(to: open.location)
            if before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let afterOpen = ns.substring(from: open.location + open.length) as NSString
                let close = afterOpen.range(of: closeTag, options: .literal)
                if close.location == NSNotFound {
                    return ThinkingSplit(thinking: trimThinking(afterOpen as String), answer: "", isThinkingOpen: true)
                }
                let thinking = afterOpen.substring(to: close.location)
                let answer = afterOpen.substring(from: close.location + close.length)
                return ThinkingSplit(thinking: trimThinking(thinking), answer: trimAnswer(answer), isThinkingOpen: false)
            }
        }
        // DeepSeek R1 など: プロンプト側に <think> があり、出力には </think> だけが出る
        let close = ns.range(of: closeTag, options: .literal)
        if close.location != NSNotFound {
            let thinking = ns.substring(to: close.location)
            let answer = ns.substring(from: close.location + close.length)
            return ThinkingSplit(thinking: trimThinking(thinking), answer: trimAnswer(answer), isThinkingOpen: false)
        }
        return ThinkingSplit(thinking: nil, answer: text, isThinkingOpen: false)
    }

    private static func trimThinking(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func trimAnswer(_ text: String) -> String {
        String(text.drop { $0.isWhitespace || $0.isNewline })
    }
}

/// システムプロンプトと会話履歴から、コンテキストに収まるプロンプトを作る。
struct PromptAssembler {
    struct Result {
        var text: String
        var tokens: [Int]
        var droppedMessageCount: Int
    }

    static func assemble(systemPrompt: String?, history: [ChatMessage], renderer: ChatTemplateRenderer,
                         tokenizer: TextTokenizer, contextLength: Int, reserveTokens: Int,
                         enableThinking: Bool?) throws -> Result {
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        // 過去のアシスタントの発話から思考部分を除き、同じ役割が続くものはまとめる
        var cleaned: [ChatMessage] = []
        for message in history where message.role != .system {
            var item = message
            if item.role == .assistant { item.content = ThinkingSplit.split(item.content).answer }
            if let last = cleaned.last, last.role == item.role {
                cleaned[cleaned.count - 1].content += "\n\n" + item.content
            } else {
                cleaned.append(item)
            }
        }
        let budget = max(1, contextLength - max(0, reserveTokens))
        var dropped = 0
        while true {
            var messages: [ChatMessage] = []
            if let system, !system.isEmpty { messages.append(ChatMessage(role: .system, content: system)) }
            messages.append(contentsOf: cleaned[dropped...])
            let text = try renderLeniently(messages, renderer: renderer, enableThinking: enableThinking)
            var tokens = tokenizer.encode(text, addSpecialTokens: false)
            let remaining = cleaned.count - dropped
            if tokens.count <= budget || remaining <= 1 {
                if tokens.count > budget {
                    // 最後の発言だけでも長すぎる: 先頭側を切り詰める
                    tokens = Array(tokens.suffix(budget))
                }
                return Result(text: text, tokens: tokens, droppedMessageCount: dropped)
            }
            // 古いものから捨てる（user から始まるように、続く assistant も一緒に）
            dropped += 1
            while dropped < cleaned.count - 1 && cleaned[dropped].role != .user { dropped += 1 }
        }
    }

    /// テンプレートが system を拒否したら最初の user にまとめて再挑戦し、それでもだめなら組み込みの書式にする。
    private static func renderLeniently(_ messages: [ChatMessage], renderer: ChatTemplateRenderer,
                                        enableThinking: Bool?) throws -> String {
        do {
            return try renderer.render(messages: messages, addGenerationPrompt: true, enableThinking: enableThinking)
        } catch {
            let merged = BuiltinChatFormat.mergingSystemIntoFirstUser(messages)
            if let text = try? renderer.render(messages: merged, addGenerationPrompt: true, enableThinking: enableThinking) {
                return text
            }
            return renderer.renderBuiltin(messages: messages, addGenerationPrompt: true)
        }
    }
}
