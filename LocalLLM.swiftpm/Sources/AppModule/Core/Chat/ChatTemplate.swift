import Foundation

/// チャットのメッセージ 1 つ。
struct ChatMessage: Codable, Hashable, Sendable {
    enum Role: String, Codable, Sendable {
        case system, user, assistant, tool
    }

    var role: Role
    var content: String
    /// assistant がツールを呼び出したとき
    var toolCalls: [ToolCall]?
    /// tool の結果のとき: ツールの名前と、呼び出しの ID
    var toolName: String?
    var toolCallID: String?

    init(role: Role, content: String, toolCalls: [ToolCall]? = nil, toolName: String? = nil, toolCallID: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolName = toolName
        self.toolCallID = toolCallID
    }

    /// チャットテンプレートの messages に入れる形（Hugging Face と同じキー）
    var templateValue: JinjaValue {
        var pairs: [(String, JinjaValue)] = [("role", .string(role.rawValue)), ("content", .string(content))]
        if let toolCalls, !toolCalls.isEmpty {
            pairs.append(("tool_calls", .array(toolCalls.map { $0.templateValue.jinjaValue })))
        }
        if let toolName { pairs.append(("name", .string(toolName))) }
        if let toolCallID { pairs.append(("tool_call_id", .string(toolCallID))) }
        return .object(JinjaObject(pairs))
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

    func render(messages original: [ChatMessage], addGenerationPrompt: Bool, bosToken: String?, eosToken: String?) -> String {
        // 組み込みの書式には tool の役割が無いので、普通の文章に書き直す
        let messages = ToolPrompt.flatten(original)
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
                case .user, .tool: output += "<｜User｜>\(message.content)"
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
                case .user, .tool: output += "User: \(message.content)\n"
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

    /// テンプレートがツール（tools 変数と tool の役割）を扱えるか
    var supportsTools: Bool {
        guard compiled != nil, let template else { return false }
        return template.contains("tools") && (template.contains("tool_calls") || template.contains("tool_call"))
    }

    /// 思考モード（テンプレートの enable_thinking）を切り替えられるか、と既定の向き。
    /// nil なら切り替えられない（オン・オフで同じプロンプトになる）。true なら既定で考える（Qwen3）、false なら既定は考えない（Gemma 4）
    var thinkingDefault: Bool? {
        guard compiled != nil else { return nil }
        let probe = [ChatMessage(role: .user, content: "hi")]
        func prompt(_ value: Bool?) -> String? {
            try? self.render(messages: probe, addGenerationPrompt: true, enableThinking: value)
        }
        guard let on = prompt(true), let off = prompt(false), on != off else { return nil }
        return prompt(nil) == on
    }

    /// テンプレートから推定した「ターン終了」トークン
    var endOfTurnTokens: [String] {
        var tokens: [String] = []
        let source = template ?? ""
        // <|tool_response> は Gemma 4: ツールを呼んだあと、結果を待つ印（ここで止めて、アプリが結果を渡す）
        let candidates = ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|end|>", "[|endofturn|]",
                          "<｜end▁of▁sentence｜>", "<|endoftext|>", "<|end_of_text|>", "<turn|>", Gemma4ToolSyntax.responseMarker]
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
                tools: [ToolDefinition] = [], extraContext: [String: JinjaValue] = [:]) throws -> String {
        guard let compiled else {
            return renderBuiltin(messages: messages, addGenerationPrompt: addGenerationPrompt)
        }
        var context: [String: JinjaValue] = [
            "messages": .array(messages.map(\.templateValue)),
            "add_generation_prompt": .bool(addGenerationPrompt),
            "tools": tools.isEmpty ? .none : .array(tools.map { $0.templateValue.jinjaValue }),
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

/// 生成テキストから <think>…</think> の思考部分を分ける（Gemma 4 の <|channel>thought … <channel|> も）。
struct ThinkingSplit: Equatable {
    var thinking: String?
    var answer: String
    var isThinkingOpen: Bool

    static let openTag = "<think>"
    static let closeTag = "</think>"
    /// Gemma 4 の思考の始まりと終わり
    static let channelOpenTag = "<|channel>thought"
    static let channelCloseTag = "<channel|>"

    static func split(_ text: String) -> ThinkingSplit {
        // ツールの結果のあとはプロンプト側に <|channel>thought があり、出力には <channel|> だけが出る
        if text.contains(channelOpenTag) || text.contains(channelCloseTag) {
            return split(text, open: channelOpenTag, close: channelCloseTag)
        }
        return split(text, open: openTag, close: closeTag)
    }

    private static func split(_ text: String, open openTag: String, close closeTag: String) -> ThinkingSplit {
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

    /// 描画に要るもの（何度か描き直すので、まとめておく）
    private struct Parts {
        var system: String?
        var cleaned: [ChatMessage]
        var renderer: ChatTemplateRenderer
        var tokenizer: TextTokenizer
        var enableThinking: Bool?
        var tools: [ToolDefinition]

        /// 先頭の dropped 個を捨てて描画する（last があれば、最後の発言をそれに替える）
        func render(dropping dropped: Int, last: ChatMessage? = nil) throws -> Result {
            var messages: [ChatMessage] = []
            if let system, !system.isEmpty { messages.append(ChatMessage(role: .system, content: system)) }
            var kept: [ChatMessage] = Array(cleaned[min(dropped, cleaned.count)...])
            if let last, !kept.isEmpty { kept[kept.count - 1] = last }
            messages.append(contentsOf: kept)
            let text: String = try PromptAssembler.renderLeniently(messages, renderer: renderer, enableThinking: enableThinking,
                                                                   tools: tools)
            let tokens: [Int] = tokenizer.encode(text, addSpecialTokens: false)
            return Result(text: text, tokens: tokens, droppedMessageCount: dropped)
        }
    }

    static func assemble(systemPrompt: String?, history: [ChatMessage], renderer: ChatTemplateRenderer,
                         tokenizer: TextTokenizer, contextLength: Int, reserveTokens: Int,
                         enableThinking: Bool?, tools: [ToolDefinition] = []) throws -> Result {
        let parts: Parts = prepare(systemPrompt: systemPrompt, history: history, renderer: renderer, tokenizer: tokenizer,
                                   enableThinking: enableThinking, tools: tools)
        let budget: Int = max(1, contextLength - max(0, reserveTokens))
        let full: Result = try parts.render(dropping: 0)
        if full.tokens.count <= budget || parts.cleaned.isEmpty { return full }
        if let fitted = try dropOldMessages(parts, full: full, budget: budget) { return fitted }
        return try shortenLastMessage(parts, budget: budget)
    }

    /// システムプロンプトとツールの説明、思考を除いてまとめたやり取りを用意する
    private static func prepare(systemPrompt: String?, history: [ChatMessage], renderer: ChatTemplateRenderer,
                                tokenizer: TextTokenizer, enableThinking: Bool?, tools: [ToolDefinition]) -> Parts {
        var system: String? = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        // テンプレートがツールに対応していなければ、説明をシステムプロンプトに書き、ツールのやり取りを文章にする
        let nativeTools: Bool = !tools.isEmpty && renderer.supportsTools
        var source: [ChatMessage] = history
        if !tools.isEmpty && !nativeTools {
            let instructions: String = ToolPrompt.instructions(for: tools)
            if let current = system, !current.isEmpty {
                system = current + "\n\n" + instructions
            } else {
                system = instructions
            }
            source = ToolPrompt.flatten(history)
        }
        // 過去のアシスタントの発話から思考部分を除き、同じ役割が続くものはまとめる（ツールのやり取りはまとめない）
        var cleaned: [ChatMessage] = []
        for message in source where message.role != .system {
            var item: ChatMessage = message
            if item.role == .assistant { item.content = ThinkingSplit.split(item.content).answer }
            if let last = cleaned.last, last.role == item.role, item.role != .tool,
               last.toolCalls == nil, item.toolCalls == nil {
                cleaned[cleaned.count - 1].content += "\n\n" + item.content
            } else {
                cleaned.append(item)
            }
        }
        return Parts(system: system, cleaned: cleaned, renderer: renderer, tokenizer: tokenizer,
                     enableThinking: enableThinking, tools: nativeTools ? tools : [])
    }

    /// 古いものから捨てる（user から始まるように、続く assistant / tool も一緒に）。入らなければ nil。
    /// 捨てる量は「入りきらない分」ちょうどではなく、budget の 1/3 ごとの切りのよい量にする。
    /// ちょうどにすると、会話が続くたびに捨てる位置が 1 つずつずれてプロンプトの先頭が毎回変わり、
    /// KV キャッシュが使えずに毎回会話全体を読み直すことになる（長い会話で毎回「読み込み」がとても長くなっていた）。
    /// 切りのよい量にしておけば、次に捨てる位置が変わるまでの何回かは、前の読み込みをそのまま使える
    private static func dropOldMessages(_ parts: Parts, full: Result, budget: Int) throws -> Result? {
        let candidates: [Int] = boundaries(parts.cleaned)
        // 発言ごとのトークン数の見積もり（テンプレートの印の分を少し足す）。先頭からの合計は、前の発言が変わらない限り同じ
        var before: [Int] = [0]
        for message in parts.cleaned {
            let count: Int = parts.tokenizer.encode(message.content, addSpecialTokens: false).count
            before.append(before[before.count - 1] + count + 4)
        }
        let quantum: Int = max(64, budget / 3)
        let excess: Int = full.tokens.count - budget
        let target: Int = (excess + quantum - 1) / quantum * quantum
        var first: Int = max(0, candidates.count - 1)
        for (position, boundary) in candidates.enumerated() where before[boundary] >= target {
            first = position
            break
        }
        for dropped in candidates[first...] {
            let attempt: Result = try parts.render(dropping: dropped)
            if attempt.tokens.count <= budget { return attempt }
        }
        return nil
    }

    /// 捨ててよい位置（user の発言の始まりか、最後の発言だけを残す位置）
    private static func boundaries(_ cleaned: [ChatMessage]) -> [Int] {
        var result: [Int] = []
        let last: Int = cleaned.count - 1
        for index in 1..<max(1, cleaned.count) where cleaned[index].role == .user || index == last {
            result.append(index)
        }
        return result
    }

    /// 最後の発言だけでも長すぎる: 発言の中ほどを省いて縮める（プロンプトの先頭（BOS やシステムプロンプト）を切ると、
    /// テンプレートの形が崩れて答えがおかしくなるので、切らない）
    private static func shortenLastMessage(_ parts: Parts, budget: Int) throws -> Result {
        let dropped: Int = boundaries(parts.cleaned).last ?? 0
        var last: ChatMessage = parts.cleaned[parts.cleaned.count - 1]
        var attempt: Result = try parts.render(dropping: dropped)
        for _ in 0..<4 where attempt.tokens.count > budget {
            let characters: [Character] = Array(last.content)
            let contentTokens: Int = max(1, parts.tokenizer.encode(last.content, addSpecialTokens: false).count)
            let keepTokens: Int = max(0, contentTokens - (attempt.tokens.count - budget) - 16)
            let keep: Int = characters.count * keepTokens / contentTokens
            guard keep > 0 else { break }
            last.content = shortened(characters, keep: keep)
            attempt = try parts.render(dropping: dropped, last: last)
        }
        // それでも入らない（テンプレートそのものが長すぎる）ときだけ、後ろを残して切る
        if attempt.tokens.count > budget { attempt.tokens = Array(attempt.tokens.suffix(budget)) }
        return attempt
    }

    /// 文章の頭（4 割）と終わり（6 割）を残して、keep 文字にする
    private static func shortened(_ characters: [Character], keep: Int) -> String {
        let head: Int = keep * 2 / 5
        let opening = String(characters[..<head])
        let closing = String(characters[(characters.count - (keep - head))...])
        let marker: String = "\n…（" + L("中略", "omitted") + "）…\n"
        return opening + marker + closing
    }

    /// テンプレートが system を拒否したら最初の user にまとめて再挑戦し、それでもだめなら組み込みの書式にする。
    private static func renderLeniently(_ messages: [ChatMessage], renderer: ChatTemplateRenderer,
                                        enableThinking: Bool?, tools: [ToolDefinition]) throws -> String {
        do {
            return try renderer.render(messages: messages, addGenerationPrompt: true, enableThinking: enableThinking,
                                       tools: tools)
        } catch {
            let merged = BuiltinChatFormat.mergingSystemIntoFirstUser(messages)
            if let text = try? renderer.render(messages: merged, addGenerationPrompt: true, enableThinking: enableThinking,
                                               tools: tools) {
                return text
            }
            return renderer.renderBuiltin(messages: messages, addGenerationPrompt: true)
        }
    }
}
