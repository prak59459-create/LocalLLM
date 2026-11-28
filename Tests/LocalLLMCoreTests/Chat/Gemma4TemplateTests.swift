import XCTest
@testable import LocalLLMCore

/// Gemma 4（E2B・E4B など）のチャットテンプレートと、ツールの呼び出しの書き方。
///
/// テンプレートの正解は Tools/fixtures/gemma4_template_expectations.py が jinja2 で描画したもの
/// （Tests/GGUFFixtures/gemma4-chat-template-expected.json。テンプレートは GGUF に入っているもの）。
final class Gemma4TemplateTests: XCTestCase {
    struct Case {
        var name: String
        var messages: [ChatMessage]
        var tools: Bool
        var thinking: Bool?
        var generation: Bool
        var output: String
    }

    static func template() throws -> String {
        try String(contentsOf: GGUFFixtures.root.appendingPathComponent("gemma4-chat-template.jinja"), encoding: .utf8)
    }

    static func message(_ value: JSONValue) -> ChatMessage {
        let calls = value["tool_calls"]?.arrayValue?.map { call in
            ToolCall(id: call["id"]?.stringValue ?? "", name: call["function"]?["name"]?.stringValue ?? "",
                     arguments: call["function"]?["arguments"] ?? .object([]))
        }
        return ChatMessage(role: ChatMessage.Role(rawValue: value["role"]?.stringValue ?? "user") ?? .user,
                           content: value["content"]?.stringValue ?? "", toolCalls: calls,
                           toolName: value["name"]?.stringValue, toolCallID: value["tool_call_id"]?.stringValue)
    }

    static func expectations() throws -> (tools: [ToolDefinition], cases: [Case]) {
        let text = try String(contentsOf: GGUFFixtures.root.appendingPathComponent("gemma4-chat-template-expected.json"),
                              encoding: .utf8)
        let root = try JSONValue.parse(text)
        let tools = (root["tools"]?.arrayValue ?? []).compactMap { item -> ToolDefinition? in
            guard let function = item["function"], let name = function["name"]?.stringValue else { return nil }
            return ToolDefinition(name: name, description: function["description"]?.stringValue ?? "",
                                  parameters: function["parameters"] ?? .object([]))
        }
        let cases = (root["cases"]?.arrayValue ?? []).map { item in
            Case(name: item["name"]?.stringValue ?? "", messages: (item["messages"]?.arrayValue ?? []).map(message),
                 tools: item["tools"]?.boolValue ?? false, thinking: item["enableThinking"]?.boolValue,
                 generation: item["addGenerationPrompt"]?.boolValue ?? true, output: item["output"]?.stringValue ?? "")
        }
        return (tools, cases)
    }

    /// ツールの宣言・呼び出し・結果・思考を含む会話を、jinja2 と同じに描画する
    func testTemplateMatchesJinja2() throws {
        let renderer = ChatTemplateRenderer(template: try Self.template(), bosToken: "<bos>", eosToken: "<turn|>")
        XCTAssertTrue(renderer.usesJinja)
        XCTAssertTrue(renderer.supportsTools)
        let (tools, cases) = try Self.expectations()
        XCTAssertEqual(cases.count, 12)
        for item in cases {
            let output = try renderer.render(messages: item.messages, addGenerationPrompt: item.generation,
                                             enableThinking: item.thinking, tools: item.tools ? tools : [])
            XCTAssertEqual(output, item.output, item.name)
        }
    }

    /// ターンの終わり（<turn|>）と、ツールの結果を待つ印（<|tool_response>）で止める
    func testStopsAtTurnEndAndToolResponse() throws {
        let renderer = ChatTemplateRenderer(template: try Self.template(), bosToken: "<bos>", eosToken: "<turn|>")
        XCTAssertTrue(renderer.endOfTurnTokens.contains("<turn|>"))
        XCTAssertTrue(renderer.endOfTurnTokens.contains("<|tool_response>"))
        // ほかのモデルでは <|tool_response> で止めない
        let qwen = ChatTemplateRenderer(template: "{% for m in messages %}<|im_start|>{{ m.content }}<|im_end|>{% endfor %}",
                                        bosToken: nil, eosToken: nil)
        XCTAssertFalse(qwen.endOfTurnTokens.contains("<|tool_response>"))
    }

    /// Gemma 4 の書き方の呼び出し: call:名前{キー:値,…}（文字列は <|"|> で囲む）
    func testParsesGemma4ToolCalls() {
        let known: Set<String> = ["web_search", "read_file"]
        var result = ToolCallParser.parse("<|tool_call>call:web_search{query:<|\"|>東京 天気<|\"|>}<tool_call|>", knownTools: known)
        XCTAssertEqual(result.calls.map(\.name), ["web_search"])
        XCTAssertEqual(result.calls.first?.argument("query"), "東京 天気")
        XCTAssertEqual(result.content, "")

        // いろいろな型の引数・文字列の中のカンマや括弧・引用符、前置きの文、生成が止まった印
        let text = "読みます。<|tool_call>call:read_file{all:true,offset:3,path:<|\"|>a \"b\" {c}, d.txt<|\"|>,"
            + "range:{from:1,to:2},tags:[<|\"|>x<|\"|>,<|\"|>y<|\"|>],none:null,ratio:-1.5}<tool_call|><|tool_response>"
        result = ToolCallParser.parse(text, knownTools: known)
        XCTAssertEqual(result.content, "読みます。")
        let arguments = result.calls.first?.arguments
        XCTAssertEqual(arguments?["path"], .string("a \"b\" {c}, d.txt"))
        XCTAssertEqual(arguments?["offset"], .int(3))
        XCTAssertEqual(arguments?["all"], .bool(true))
        XCTAssertEqual(arguments?["range"], .object([("from", .int(1)), ("to", .int(2))]))
        XCTAssertEqual(arguments?["tags"], .array([.string("x"), .string("y")]))
        XCTAssertEqual(arguments?["none"], .null)
        XCTAssertEqual(arguments?["ratio"], .double(-1.5))

        // 2 つ続けて呼ぶ・閉じる印の前で止まった・普通の引用符で書いた
        result = ToolCallParser.parse("<|tool_call>call:web_search{query:<|\"|>a<|\"|>}<tool_call|>"
                                      + "<|tool_call>call:read_file{path:\"x.txt\"}", knownTools: known)
        XCTAssertEqual(result.calls.map(\.name), ["web_search", "read_file"])
        XCTAssertEqual(result.calls.last?.argument("path"), "x.txt")

        // 知らないツールは呼び出しにしない（本文からは消す）
        result = ToolCallParser.parse("<|tool_call>call:delete_all{}<tool_call|>", knownTools: known)
        XCTAssertTrue(result.calls.isEmpty)
    }

    /// 生成中は、呼び出しの印から先を画面に出さない
    func testHidesGemma4ToolCallWhileStreaming() {
        XCTAssertEqual(ToolCallParser.visiblePart(of: "調べます。<|tool_call>call:web_search{query:<|\"|>東"), "調べます。")
        XCTAssertEqual(ToolCallParser.visiblePart(of: "<|tool_call>call:web"), "")
    }

    /// 呼び出しを履歴に書き戻すと、テンプレートが Gemma 4 の書き方で描く（引数は型のまま）
    func testToolCallRoundTripsThroughTemplate() throws {
        let renderer = ChatTemplateRenderer(template: try Self.template(), bosToken: "<bos>", eosToken: "<turn|>")
        let generated = "<|tool_call>call:read_file{offset:3,path:<|\"|>x.txt<|\"|>}<tool_call|>"
        let parsed = ToolCallParser.parse(generated, knownTools: ["read_file"])
        let messages = [ChatMessage(role: .user, content: "読んで"),
                        ChatMessage(role: .assistant, content: parsed.content, toolCalls: parsed.calls)]
        let output = try renderer.render(messages: messages, addGenerationPrompt: true, enableThinking: nil,
                                         tools: [ToolDefinition(name: "read_file", description: "Read",
                                                                arguments: [("path", "Path", true)])])
        XCTAssertTrue(output.hasSuffix("<|turn>model\n" + generated + "<|tool_response>"), output)
    }

    /// 思考モードは切り替えられ、既定は考えない（オンにするとシステムのターンの先頭に <|think|> が入る）
    func testThinkingCanBeSwitchedAndIsOffByDefault() throws {
        let renderer = ChatTemplateRenderer(template: try Self.template(), bosToken: "<bos>", eosToken: "<turn|>")
        XCTAssertEqual(renderer.thinkingDefault, false)
        let messages = [ChatMessage(role: .system, content: "S"), ChatMessage(role: .user, content: "Q")]
        let on = try renderer.render(messages: messages, addGenerationPrompt: true, enableThinking: true)
        XCTAssertTrue(on.hasPrefix("<bos><|turn>system\n<|think|>\nS<turn|>\n"), on)
        // 既定の向きに合わせる: Qwen3 のようなテンプレートは既定で考える
        let qwen = ChatTemplateRenderer(template: "{% for m in messages %}{{ m.content }}{% endfor %}"
                                            + "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}",
                                        bosToken: nil, eosToken: nil)
        XCTAssertEqual(qwen.thinkingDefault, true)
        // enable_thinking を使わないテンプレートは切り替えられない
        let plain = ChatTemplateRenderer(template: "{% for m in messages %}{{ m.content }}{% endfor %}", bosToken: nil, eosToken: nil)
        XCTAssertNil(plain.thinkingDefault)
    }

    /// 思考（<|channel>thought … <channel|>）を答えと分ける
    func testSplitsGemma4Thinking() {
        var split = ThinkingSplit.split("<|channel>thought\n2 と 2 を足す<channel|>4 です。")
        XCTAssertEqual(split.thinking, "2 と 2 を足す")
        XCTAssertEqual(split.answer, "4 です。")
        XCTAssertFalse(split.isThinkingOpen)
        split = ThinkingSplit.split("<|channel>thought\n考え中")
        XCTAssertEqual(split.thinking, "考え中")
        XCTAssertTrue(split.isThinkingOpen)
        XCTAssertEqual(split.answer, "")
        // 思考が空のとき
        split = ThinkingSplit.split("<|channel>thought\n<channel|>答え")
        XCTAssertEqual(split.answer, "答え")
        // <think> は今までどおり
        XCTAssertEqual(ThinkingSplit.split("<think>a</think>b").answer, "b")
    }
}
