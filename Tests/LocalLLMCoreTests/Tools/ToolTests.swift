import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

final class JSONValueTests: XCTestCase {
    func testParseKeepsOrderAndRoundTrips() throws {
        let text = #"{"b": 1, "a": [true, null, 1.5, "x\n\"y\""], "c": {"z": "日本語😀"}}"#
        let value = try JSONValue.parse(text)
        XCTAssertEqual(value.objectValue?.map(\.0), ["b", "a", "c"])
        XCTAssertEqual(value["a"]?.arrayValue?.count, 4)
        XCTAssertEqual(value["c"]?["z"]?.stringValue, "日本語😀")
        XCTAssertEqual(try JSONValue.parse(value.serialized()), value)
        XCTAssertEqual(JSONValue.object([("k", .double(2))]).serialized(), #"{"k": 2.0}"#)
        XCTAssertEqual(try JSONValue.parse(#""😀""#).stringValue, "😀")
        XCTAssertThrowsError(try JSONValue.parse("{\"a\": 1,}"))
        XCTAssertThrowsError(try JSONValue.parse("[1, 2] x"))
    }

    func testLenientParsing() throws {
        let value = try JSONValue.parse("{name: 'web_search', arguments: {'query': \"東京\",}, flag: True, n: None", lenient: true)
        XCTAssertEqual(value["name"]?.stringValue, "web_search")
        XCTAssertEqual(value["arguments"]?["query"]?.stringValue, "東京")
        XCTAssertEqual(value["flag"]?.boolValue, true)
        XCTAssertEqual(value["n"], .null)
        let found = try XCTUnwrap(JSONValue.firstObject(in: "了解です。{\"a\": [1, 2]} 以上"))
        XCTAssertEqual(found.value["a"]?.arrayValue?.count, 2)
        XCTAssertNil(JSONValue.firstObject(in: "{} no json"))
    }

    func testCodableAndJinja() throws {
        let value = JSONValue.object([("z", .int(1)), ("a", .array([.string("x")]))])
        let decoded = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value))
        XCTAssertEqual(decoded.objectValue?.map(\.0), ["z", "a"])
        let rendered = try JinjaTemplate("{{ v | tojson }}").render(["v": value.jinjaValue])
        XCTAssertEqual(rendered, #"{"z": 1, "a": ["x"]}"#)
        XCTAssertEqual(JSONValue(any: ["b": true, "a": 3, "c": [1.5, "s"]] as [String: Any]),
                       .object([("a", .int(3)), ("b", .bool(true)), ("c", .array([.double(1.5), .string("s")]))]))
    }
}

final class ToolCallParserTests: XCTestCase {
    let known: Set<String> = ["web_search", "read_file"]

    func testQwenHermesFormat() {
        let text = "調べます。\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"東京 天気\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"a.txt\\\"}\"}\n</tool_call>"
        let result = ToolCallParser.parse(text, knownTools: known)
        XCTAssertEqual(result.content, "調べます。")
        XCTAssertEqual(result.calls.map(\.name), ["web_search", "read_file"])
        XCTAssertEqual(result.calls[0].argument("query"), "東京 天気")
        XCTAssertEqual(result.calls[1].argument("path"), "a.txt")
        XCTAssertEqual(ToolCallParser.visiblePart(of: text), "調べます。\n")
    }

    func testLlamaMistralAndFunctionTags() {
        let llama = ToolCallParser.parse("<|python_tag|>{\"name\": \"web_search\", \"parameters\": {\"query\": \"swift\"}}", knownTools: known)
        XCTAssertEqual(llama.calls.first?.argument("query"), "swift")
        XCTAssertEqual(llama.content, "")
        let mistral = ToolCallParser.parse("[{\"name\": \"read_file\", \"arguments\": {\"path\": \"x\"}}]", knownTools: known)
        XCTAssertEqual(mistral.calls.first?.name, "read_file")
        let function = ToolCallParser.parse("<function=web_search>{\"query\": \"a\"}</function>", knownTools: known)
        XCTAssertEqual(function.calls.first?.argument("query"), "a")
        let fenced = ToolCallParser.parse("```json\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"q\"}}\n```", knownTools: known)
        XCTAssertEqual(fenced.calls.first?.argument("query"), "q")
        let inline = ToolCallParser.parse("検索します {\"tool\": \"web_search\", \"args\": {\"query\": \"q2\"}} お待ちください", knownTools: known)
        XCTAssertEqual(inline.calls.first?.argument("query"), "q2")
        XCTAssertEqual(inline.content, "検索します  お待ちください")
        let truncated = ToolCallParser.parse("<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"切れた", knownTools: known)
        XCTAssertEqual(truncated.calls.first?.argument("query"), "切れた")
    }

    func testIgnoresUnknownToolsAndPlainJSON() {
        let unknown = ToolCallParser.parse("{\"name\": \"delete_everything\", \"arguments\": {}}", knownTools: known)
        XCTAssertTrue(unknown.calls.isEmpty)
        let answer = "例: {\"name\": \"太郎\", \"age\": 3} のように書きます"
        XCTAssertEqual(ToolCallParser.parse(answer, knownTools: known), .init(content: answer, calls: []))
        XCTAssertEqual(ToolCallParser.visiblePart(of: "{\"name\": \"web_search\""), "")
        XCTAssertEqual(ToolCallParser.visiblePart(of: "普通の答え"), "普通の答え")
    }

    func testFallbackPromptForTemplatesWithoutTools() throws {
        let tool = ToolDefinition(name: "web_search", description: "Search the web",
                                  arguments: [("query", "words to search", true)])
        XCTAssertTrue(ToolPrompt.instructions(for: [tool]).contains("web_search: Search the web"))
        let renderer = ChatTemplateRenderer(template: nil, bosToken: nil, eosToken: nil, fallbackFormat: .chatML)
        XCTAssertFalse(renderer.supportsTools)
        let history = [
            ChatMessage(role: .user, content: "Q"),
            ChatMessage(role: .assistant, content: "", toolCalls: [ToolCall(id: "1", name: "web_search", arguments: .object([("query", .string("q"))]))]),
            ChatMessage(role: .tool, content: "R", toolName: "web_search", toolCallID: "1"),
        ]
        let result = try PromptAssembler.assemble(systemPrompt: "S", history: history, renderer: renderer,
                                                  tokenizer: CharacterMockTokenizer(), contextLength: 4096, reserveTokens: 16,
                                                  enableThinking: nil, tools: [tool])
        XCTAssertTrue(result.text.contains("<|im_start|>system\nS\n\n"))
        XCTAssertTrue(result.text.contains("<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"q\"}}\n</tool_call>"))
        XCTAssertTrue(result.text.contains("<|im_start|>user\n<tool_response name=\"web_search\">\nR\n</tool_response><|im_end|>"))
    }

    func testNativeToolTemplate() throws {
        let template = "{% if tools %}TOOLS:{% for t in tools %}{{ t.function.name }};{% endfor %}{% endif %}{% for m in messages %}[{{ m.role }}:{{ m.content }}{% if m.tool_calls %}{% for c in m.tool_calls %}<{{ c.function.name }} {{ c.function.arguments | tojson }}>{% endfor %}{% endif %}]{% endfor %}"
        let renderer = ChatTemplateRenderer(template: template + "{# tool_call #}", bosToken: nil, eosToken: nil)
        XCTAssertTrue(renderer.supportsTools)
        let tool = ToolDefinition(name: "web_search", description: "d", arguments: [("query", "q", true)])
        let call = ToolCall(id: "1", name: "web_search", arguments: .object([("query", .string("x"))]))
        let text = try renderer.render(messages: [ChatMessage(role: .assistant, content: "", toolCalls: [call]),
                                                  ChatMessage(role: .tool, content: "R")],
                                       addGenerationPrompt: true, enableThinking: nil, tools: [tool])
        XCTAssertEqual(text, "TOOLS:web_search;[assistant:<web_search {\"query\": \"x\"}>][tool:R]")
    }
}

final class WebToolsTests: XCTestCase {
    static let duckDuckGoHTML = """
    <div class="result results_links results_links_deep web-result ">
      <div class="links_main links_deep result__body">
        <h2 class="result__title">
          <a rel="nofollow" class="result__a" href="https://developer.apple.com/documentation/coreml">Core ML | Apple Developer Documentation</a>
        </h2>
        <a class="result__snippet" href="https://developer.apple.com/documentation/coreml">Use <b>Core</b> <b>ML</b> to integrate models on a person&#x27;s device.</a>
      </div>
    </div>
    <div class="result results_links result--ad ">
      <a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_domain=x">Ad</a>
      <a class="result__snippet" href="#">buy</a>
    </div>
    <div class="result results_links">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1&amp;rut=abc">Example &amp; Co</a>
    </div>
    """

    func testDuckDuckGoParser() {
        let results = DuckDuckGoParser.parse(Self.duckDuckGoHTML)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0], WebSearchResult(title: "Core ML | Apple Developer Documentation",
                                                   url: "https://developer.apple.com/documentation/coreml",
                                                   snippet: "Use Core ML to integrate models on a person's device."))
        XCTAssertEqual(results[1].url, "https://example.com/page?a=1")
        XCTAssertEqual(results[1].title, "Example & Co")
        XCTAssertTrue(WebSearchClient.format(results).hasPrefix("[1] Core ML"))
    }

    func testHTMLTextExtraction() {
        let html = """
        <html><head><title>テスト &amp; ページ</title><style>body{color:red}</style>
        <script>var x = "<p>not text</p>";</script></head>
        <body><nav>menu</nav><h1>見出し</h1><p>段落の<b>文章</b>です。&nbsp;次&#x3002;</p>
        <!-- comment --><ul><li>一</li><li>二</li></ul><SCRIPT>alert(1)</SCRIPT><p>最後</p></body></html>
        """
        let page = HTMLText.extract(Data(html.utf8))
        XCTAssertEqual(page.title, "テスト & ページ")
        XCTAssertEqual(page.text, "見出し\n段落の文章です。 次。\n一\n二\n最後")
        XCTAssertEqual(HTMLText.decodeEntities("a &lt;b&gt; &#65;&unknown; &"), "a <b> A&unknown; &")
        // 大きなページでも速い（最適化なしのビルドで）
        let big = Data(String(repeating: "<div><p>テキスト<span>abc</span></p></div>\n", count: 20_000).utf8)
        let start = Date()
        XCTAssertFalse(HTMLText.extract(big).text.isEmpty)
        XCTAssertLessThan(Date().timeIntervalSince(start), 3)
    }

    func testWikipediaAndBraveParsers() {
        let wiki = Data(#"{"query": {"search": [{"title": "東京 都", "snippet": "<span class=\"searchmatch\">東京</span>は日本の首都"}]}}"#.utf8)
        XCTAssertEqual(WikipediaParser.parseSearch(wiki, language: "ja"),
                       [WebSearchResult(title: "東京 都", url: "https://ja.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC_%E9%83%BD", snippet: "東京は日本の首都")])
        let brave = Data(#"{"web": {"results": [{"title": "T <strong>x</strong>", "url": "https://t", "description": "D"}]}}"#.utf8)
        XCTAssertEqual(BraveParser.parse(brave).first?.title, "T x")
        XCTAssertEqual(SearXNGParser.parse(Data(#"{"results": [{"title": "S", "url": "u", "content": "c"}]}"#.utf8)).count, 1)
    }
}

/// MCP サーバーのまね（URLProtocol）
final class MockMCPProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _received: [JSONValue] = []
    nonisolated(unsafe) static var useEventStream = false

    static var received: [JSONValue] {
        lock.lock(); defer { lock.unlock() }
        return _received
    }

    static func reset() {
        lock.lock()
        _received = []
        lock.unlock()
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "mcp.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        var body = request.httpBody ?? Data()
        if body.isEmpty, let stream = request.httpBodyStream {
            stream.open()
            var buffer = [UInt8](repeating: 0, count: 65_536)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                body.append(buffer, count: count)
            }
            stream.close()
        }
        let message = (try? JSONValue.parse(String(decoding: body, as: UTF8.self))) ?? .null
        Self.lock.lock()
        Self._received.append(message)
        Self.lock.unlock()
        let method = message["method"]?.stringValue ?? ""
        var headers = ["Content-Type": Self.useEventStream ? "text/event-stream" : "application/json"]
        if method == "initialize" { headers["Mcp-Session-Id"] = "session-1" }
        guard let id = message["id"] else {
            let response = HTTPURLResponse(url: request.url!, statusCode: 202, httpVersion: "HTTP/1.1", headerFields: [:])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let result: JSONValue
        switch method {
        case "initialize":
            result = .object([("protocolVersion", .string("2025-06-18")), ("serverInfo", .object([("name", .string("mock"))]))])
        case "tools/list":
            if message["params"]?["cursor"] == nil {
                result = .object([("tools", .array([.object([("name", .string("echo")), ("description", .string("Echo text")),
                                                             ("inputSchema", .object([("type", .string("object"))]))])])),
                                  ("nextCursor", .string("page2"))])
            } else {
                result = .object([("tools", .array([.object([("name", .string("add")), ("description", .string("Add"))])]))])
            }
        case "tools/call":
            let text = message["params"]?["arguments"]?["text"]?.stringValue ?? ""
            result = .object([("content", .array([.object([("type", .string("text")), ("text", .string("echo: " + text))])])),
                              ("isError", .bool(false))])
        default:
            let error = JSONValue.object([("jsonrpc", .string("2.0")), ("id", id),
                                          ("error", .object([("code", .int(-32601)), ("message", .string("Method not found"))]))])
            send(error.serialized(), headers: headers)
            return
        }
        let reply = JSONValue.object([("jsonrpc", .string("2.0")), ("id", id), ("result", result)])
        if Self.useEventStream {
            send("event: message\ndata: {\"jsonrpc\": \"2.0\", \"method\": \"notifications/progress\"}\n\nevent: message\ndata: \(reply.serialized())\n\n",
                 headers: headers)
        } else {
            send(reply.serialized(), headers: headers)
        }
    }

    private func send(_ text: String, headers: [String: String]) {
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(text.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}

final class MCPClientTests: XCTestCase {
    func makeClient() -> MCPClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockMCPProtocol.self]
        return MCPClient(endpoint: URL(string: "https://mcp.test/mcp")!, bearerToken: "secret",
                         session: URLSession(configuration: configuration))
    }

    func testListAndCallTools() async throws {
        for eventStream in [false, true] {
            MockMCPProtocol.reset()
            MockMCPProtocol.useEventStream = eventStream
            let client = makeClient()
            let tools = try await client.listTools()
            XCTAssertEqual(tools.map(\.name), ["echo", "add"])
            XCTAssertEqual(tools[0].definition.description, "Echo text")
            let result = try await client.callTool("echo", arguments: .object([("text", .string("こんにちは"))]))
            XCTAssertEqual(result.text, "echo: こんにちは")
            XCTAssertFalse(result.isError)
            let serverName = await client.serverName
            XCTAssertEqual(serverName, "mock")
            let methods = MockMCPProtocol.received.compactMap { $0["method"]?.stringValue }
            XCTAssertEqual(methods, ["initialize", "notifications/initialized", "tools/list", "tools/list", "tools/call"])
        }
        MockMCPProtocol.useEventStream = false
    }

    func testRPCErrors() async throws {
        MockMCPProtocol.reset()
        let client = makeClient()
        try await client.initialize()
        do {
            _ = try await client.callTool("missing", arguments: .object([]))
        } catch {
            XCTFail("tools/call is handled by the mock: \(error)")
        }
        XCTAssertEqual(MCPClient.text(fromToolResult: .object([("content", .array([
            .object([("type", .string("text")), ("text", .string("a"))]),
            .object([("type", .string("resource")), ("resource", .object([("text", .string("b"))]))]),
        ]))])), "a\nb")
        XCTAssertEqual(MCPClient.parseEventStream("data: {\"a\": 1}\n\ndata: [1,\ndata: 2]\n\n").count, 3)
    }
}

/// 本物の Web に接続するテスト（LOCALLLM_LIVE_TESTS=1 のときだけ）
final class WebLiveTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["LOCALLLM_LIVE_TESTS"] == "1" else {
            throw XCTSkip("set LOCALLLM_LIVE_TESTS=1 to run tests against the web")
        }
    }

    func testWikipediaSearchAndFetch() async throws {
        let client = WebSearchClient(provider: .wikipedia, language: "en")
        let results: [WebSearchResult]
        do {
            results = try await client.search("Swift programming language", limit: 3)
        } catch WebToolError.http(429) {
            // 共有の IP からだと Wikimedia に回数制限されることがある
            throw XCTSkip("Wikipedia rate-limited this runner")
        }
        XCTAssertFalse(results.isEmpty)
        let page = try await client.fetchPage("https://en.wikipedia.org/wiki/Swift_(programming_language)", maxCharacters: 2000)
        XCTAssertTrue(page.title.contains("Swift"), page.title)
        XCTAssertGreaterThan(page.text.count, 500)
    }

    func testDuckDuckGo() async throws {
        do {
            let results = try await WebSearchClient(provider: .duckDuckGo).search("Core ML Apple", limit: 5)
            XCTAssertFalse(results.isEmpty)
        } catch WebToolError.blocked {
            throw XCTSkip("DuckDuckGo refused this runner")
        }
    }
}
