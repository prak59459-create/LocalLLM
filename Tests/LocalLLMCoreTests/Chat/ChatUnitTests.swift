import XCTest
@testable import LocalLLMCore

/// 1 文字 = 1 トークンの簡単なトークナイザ（テスト用）
final class CharacterMockTokenizer: TextTokenizer {
    func encode(_ text: String, addSpecialTokens: Bool) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0)) }))
    }
    func tokenID(for token: String) -> Int? { token.unicodeScalars.count == 1 ? Int(token.unicodeScalars.first!.value) : nil }
    func token(for id: Int) -> String? { Unicode.Scalar(UInt32(id)).map { String($0) } }
}

final class ChatUnitTests: XCTestCase {
    func testThinkingSplit() {
        XCTAssertEqual(ThinkingSplit.split("<think>\n考える\n</think>\n\n答え"),
                       ThinkingSplit(thinking: "考える", answer: "答え", isThinkingOpen: false))
        XCTAssertEqual(ThinkingSplit.split("<think>途中"), ThinkingSplit(thinking: "途中", answer: "", isThinkingOpen: true))
        XCTAssertEqual(ThinkingSplit.split("ふつうの答え"), ThinkingSplit(thinking: nil, answer: "ふつうの答え", isThinkingOpen: false))
        // DeepSeek R1: プロンプトに <think> があるので出力は </think> だけ
        XCTAssertEqual(ThinkingSplit.split("推論\n</think>\n\n結論"), ThinkingSplit(thinking: "推論", answer: "結論", isThinkingOpen: false))
        // 本文の途中に <think> と書いてあるだけなら思考扱いしない
        XCTAssertEqual(ThinkingSplit.split("タグは <think> です").thinking, nil)
    }

    func testDetectFormats() {
        XCTAssertEqual(BuiltinChatFormat.detect(template: "<|im_start|>{{ x }}", hasToken: { _ in false }), .chatML)
        XCTAssertEqual(BuiltinChatFormat.detect(template: nil, hasToken: { $0 == "<start_of_turn>" }), .gemma)
        XCTAssertEqual(BuiltinChatFormat.detect(template: nil, hasToken: { _ in false }), .plain)
    }

    func testBuiltinFormatsRender() {
        let messages = [ChatMessage(role: .system, content: "S"), ChatMessage(role: .user, content: "U")]
        XCTAssertEqual(BuiltinChatFormat.chatML.render(messages: messages, addGenerationPrompt: true, bosToken: nil, eosToken: nil),
                       "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(BuiltinChatFormat.gemma.render(messages: messages, addGenerationPrompt: true, bosToken: "<bos>", eosToken: nil),
                       "<bos><start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\n")
        for format in BuiltinChatFormat.allCases {
            let text = format.render(messages: messages, addGenerationPrompt: true, bosToken: nil, eosToken: nil)
            XCTAssertTrue(text.contains("U"), "\(format)")
        }
    }

    func testRendererFallsBackWithoutTemplate() throws {
        let renderer = ChatTemplateRenderer(template: nil, bosToken: nil, eosToken: nil, fallbackFormat: .chatML)
        XCTAssertFalse(renderer.usesJinja)
        let text = try renderer.render(messages: [ChatMessage(role: .user, content: "hi")], addGenerationPrompt: true, enableThinking: nil)
        XCTAssertEqual(text, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(renderer.endOfTurnTokens, ["<|im_end|>"])
    }

    func testRendererSurfacesRaiseException() {
        let renderer = ChatTemplateRenderer(template: "{{ raise_exception('no system') }}", bosToken: nil, eosToken: nil)
        XCTAssertThrowsError(try renderer.render(messages: [], addGenerationPrompt: true, enableThinking: nil)) { error in
            guard case JinjaError.raised(let message) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(message, "no system")
        }
    }

    func testPromptAssemblerDropsOldMessages() throws {
        let renderer = ChatTemplateRenderer(template: nil, bosToken: nil, eosToken: nil, fallbackFormat: .plain)
        let tokenizer = CharacterMockTokenizer()
        let history = [
            ChatMessage(role: .user, content: String(repeating: "a", count: 50)),
            ChatMessage(role: .assistant, content: String(repeating: "b", count: 50)),
            ChatMessage(role: .user, content: "last question"),
        ]
        let full = try PromptAssembler.assemble(systemPrompt: "SYS", history: history, renderer: renderer, tokenizer: tokenizer,
                                                contextLength: 1000, reserveTokens: 100, enableThinking: nil)
        XCTAssertEqual(full.droppedMessageCount, 0)
        XCTAssertEqual(full.tokens, tokenizer.encode(full.text, addSpecialTokens: false))

        let trimmed = try PromptAssembler.assemble(systemPrompt: "SYS", history: history, renderer: renderer, tokenizer: tokenizer,
                                                   contextLength: 100, reserveTokens: 20, enableThinking: nil)
        XCTAssertEqual(trimmed.droppedMessageCount, 2)
        XCTAssertTrue(trimmed.text.contains("SYS"))
        XCTAssertTrue(trimmed.text.contains("last question"))
        XCTAssertFalse(trimmed.text.contains("aaaa"))
        XCTAssertLessThanOrEqual(trimmed.tokens.count, 80)
    }

    /// 最後の発言だけでも長すぎるときは、発言の中ほどを省く（テンプレートの形と、発言の頭と終わりは残す）
    func testPromptAssemblerTruncatesTooLongLastMessage() throws {
        let renderer = ChatTemplateRenderer(template: nil, bosToken: nil, eosToken: nil, fallbackFormat: .plain)
        let content = "HEAD" + String(repeating: "x", count: 500) + "TAIL"
        let history = [ChatMessage(role: .user, content: content)]
        let tokenizer = CharacterMockTokenizer()
        let result = try PromptAssembler.assemble(systemPrompt: "SYS", history: history, renderer: renderer,
                                                  tokenizer: tokenizer, contextLength: 100, reserveTokens: 10,
                                                  enableThinking: nil)
        XCTAssertLessThanOrEqual(result.tokens.count, 90)
        XCTAssertGreaterThan(result.tokens.count, 50)
        XCTAssertEqual(result.droppedMessageCount, 0)
        XCTAssertEqual(result.tokens, tokenizer.encode(result.text, addSpecialTokens: false))
        // 先頭（システムプロンプト）も、発言の頭と終わりも、生成の始まりの印も残る
        XCTAssertTrue(result.text.hasPrefix("SYS\n\nUser: HEAD"), result.text)
        XCTAssertTrue(result.text.contains("TAIL\nAssistant:"), result.text)
        XCTAssertTrue(result.text.contains("…（"), result.text)
    }

    /// 長い会話で古い発言を捨てるとき、捨てる位置は会話が続いてもしばらく変わらない
    /// （毎回 1 つずつずらすと、プロンプトの先頭が毎回変わって KV キャッシュが使えず、毎回会話全体を読み直すことになる）
    func testPromptAssemblerKeepsTheSamePrefixAcrossTurns() throws {
        let renderer = ChatTemplateRenderer(template: nil, bosToken: nil, eosToken: nil, fallbackFormat: .plain)
        let tokenizer = CharacterMockTokenizer()
        var messages: [ChatMessage] = []
        var previous: PromptAssembler.Result?
        var reused = 0, turns = 0
        for turn in 0..<60 {
            messages.append(ChatMessage(role: .user, content: "question \(turn) " + String(repeating: "q", count: 20)))
            let result = try PromptAssembler.assemble(systemPrompt: "SYS", history: messages, renderer: renderer, tokenizer: tokenizer,
                                                      contextLength: 1200, reserveTokens: 200, enableThinking: nil)
            XCTAssertLessThanOrEqual(result.tokens.count, 1000)
            XCTAssertTrue(result.text.hasPrefix("SYS\n\n"))
            XCTAssertTrue(result.text.contains("question \(turn) "))
            if let previous, result.droppedMessageCount > 0 {
                turns += 1
                // 前の回のプロンプト（「Assistant:」まで）が、そのまま今回の先頭にあれば、前の読み込みを使える
                if result.text.hasPrefix(previous.text) { reused += 1 }
            }
            previous = result
            messages.append(ChatMessage(role: .assistant, content: "answer \(turn) " + String(repeating: "a", count: 20)))
        }
        XCTAssertGreaterThan(turns, 30)
        // 捨てる位置が変わった回以外は、前のプロンプトの続きになっている（1 回の発言が budget の 1/12 なら、6 割以上の回で使い回せる。
        // 前は捨てる位置が毎回ずれて、0 回だった）
        XCTAssertGreaterThan(reused, turns * 6 / 10, "reused \(reused) of \(turns)")
    }

    func testPromptAssemblerStripsThinkingAndMergesSystemForStrictTemplates() throws {
        // system を拒否するテンプレート（Gemma のような）
        let template = "{% for m in messages %}{% if m.role == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}[{{ m.role }}]{{ m.content }}{% endfor %}"
        let renderer = ChatTemplateRenderer(template: template, bosToken: nil, eosToken: nil)
        let history = [
            ChatMessage(role: .user, content: "Q1"),
            ChatMessage(role: .assistant, content: "<think>secret</think>A1"),
            ChatMessage(role: .user, content: "Q2"),
        ]
        let result = try PromptAssembler.assemble(systemPrompt: "SYS", history: history, renderer: renderer,
                                                  tokenizer: CharacterMockTokenizer(), contextLength: 1000, reserveTokens: 0,
                                                  enableThinking: nil)
        XCTAssertEqual(result.text, "[user]SYS\n\nQ1[assistant]A1[user]Q2")
    }

    func testJinjaValueFromSwift() {
        let value = JinjaValue(any: ["name": "x", "list": [1, 2.5, true, NSNull()]] as [String: Any])
        XCTAssertEqual(value.pythonRepr, "{'list': [1, 2.5, True, None], 'name': 'x'}")
    }
}
