import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// MCP（Model Context Protocol）のツール 1 つ。
struct MCPTool: Codable, Hashable, Sendable {
    var name: String
    var title: String?
    var description: String
    var inputSchema: JSONValue

    var definition: ToolDefinition {
        ToolDefinition(name: name, description: description, parameters: inputSchema)
    }
}

enum MCPError: Error, LocalizedError {
    case http(Int, String)
    case rpc(Int, String)
    case invalidResponse(String)
    case unauthorized

    var errorDescription: String? {
        switch self {
        case .http(let status, let body):
            return L("MCP サーバーの応答がエラーです（HTTP \(status)）\(body.isEmpty ? "" : ": " + body)",
                     "MCP server error (HTTP \(status))\(body.isEmpty ? "" : ": " + body)")
        case .rpc(let code, let message): return L("MCP のエラー（\(code)）: \(message)", "MCP error (\(code)): \(message)")
        case .invalidResponse(let detail): return L("MCP の応答が読めません: \(detail)", "Unreadable MCP response: \(detail)")
        case .unauthorized: return L("MCP サーバーの認証に失敗しました。トークンを確認してください。",
                                     "MCP authentication failed. Check the token.")
        }
    }
}

/// リモートの MCP サーバー（Streamable HTTP）に接続するクライアント。
/// JSON-RPC 2.0 を POST で送り、応答は JSON か Server-Sent Events で受け取る。
actor MCPClient {
    static let protocolVersion = "2025-06-18"

    let endpoint: URL
    private let headers: [String: String]
    private let session: URLSession
    private var sessionID: String?
    private var negotiatedVersion: String?
    private var nextID = 1
    private(set) var serverName: String?
    private(set) var instructions: String?

    init(endpoint: URL, bearerToken: String? = nil, extraHeaders: [String: String] = [:], session: URLSession = .shared) {
        self.endpoint = endpoint
        var headers = extraHeaders
        if let token = bearerToken?.trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty {
            headers["Authorization"] = "Bearer \(token)"
        }
        self.headers = headers
        self.session = session
    }

    var isInitialized: Bool { negotiatedVersion != nil }

    /// 接続を始める（initialize → notifications/initialized）。
    func initialize() async throws {
        let result = try await request("initialize", params: .object([
            ("protocolVersion", .string(Self.protocolVersion)),
            ("capabilities", .object([])),
            ("clientInfo", .object([("name", .string("LocalLLM")), ("version", .string("1.0"))])),
        ]))
        negotiatedVersion = result["protocolVersion"]?.stringValue ?? Self.protocolVersion
        serverName = result["serverInfo"]?["name"]?.stringValue
        instructions = result["instructions"]?.stringValue
        try await notify("notifications/initialized")
    }

    /// ツールの一覧（ページングもたどる）
    func listTools() async throws -> [MCPTool] {
        if !isInitialized { try await initialize() }
        var tools: [MCPTool] = []
        var cursor: String?
        for _ in 0..<20 {
            let params: JSONValue = cursor.map { .object([("cursor", .string($0))]) } ?? .object([])
            let result = try await request("tools/list", params: params)
            for item in result["tools"]?.arrayValue ?? [] {
                guard let name = item["name"]?.stringValue else { continue }
                tools.append(MCPTool(name: name, title: item["title"]?.stringValue,
                                     description: item["description"]?.stringValue ?? "",
                                     inputSchema: item["inputSchema"] ?? .object([("type", .string("object"))])))
            }
            cursor = result["nextCursor"]?.stringValue
            if cursor == nil { break }
        }
        return tools
    }

    /// ツールを呼ぶ。結果の文章（エラーなら isError）を返す。
    func callTool(_ name: String, arguments: JSONValue) async throws -> (text: String, isError: Bool) {
        if !isInitialized { try await initialize() }
        let result = try await request("tools/call", params: .object([("name", .string(name)), ("arguments", arguments)]))
        return (Self.text(fromToolResult: result), result["isError"]?.boolValue ?? false)
    }

    /// 結果の content（text / resource / image など）を文章にまとめる
    static func text(fromToolResult result: JSONValue) -> String {
        var parts: [String] = []
        for item in result["content"]?.arrayValue ?? [] {
            switch item["type"]?.stringValue {
            case "text":
                parts.append(item["text"]?.stringValue ?? "")
            case "resource":
                if let text = item["resource"]?["text"]?.stringValue {
                    parts.append(text)
                } else if let uri = item["resource"]?["uri"]?.stringValue {
                    parts.append("[resource] \(uri)")
                }
            case "resource_link":
                parts.append("[link] \(item["name"]?.stringValue ?? "") \(item["uri"]?.stringValue ?? "")")
            case "image":
                parts.append(L("[画像]", "[image]"))
            case "audio":
                parts.append(L("[音声]", "[audio]"))
            default:
                parts.append(item.serialized())
            }
        }
        if parts.isEmpty, let structured = result["structuredContent"] { parts.append(structured.serialized()) }
        return parts.joined(separator: "\n")
    }

    // MARK: - JSON-RPC

    private func request(_ method: String, params: JSONValue) async throws -> JSONValue {
        let id = nextID
        nextID += 1
        let message = JSONValue.object([("jsonrpc", .string("2.0")), ("id", .int(id)), ("method", .string(method)), ("params", params)])
        let (data, response) = try await post(message)
        let type = response.value(forHTTPHeaderField: "Content-Type")?.lowercased() ?? ""
        let text = String(decoding: data, as: UTF8.self)
        let messages: [JSONValue]
        if type.contains("text/event-stream") {
            messages = Self.parseEventStream(text)
        } else {
            guard let value = try? JSONValue.parse(text) else { throw MCPError.invalidResponse(String(text.prefix(200))) }
            messages = value.arrayValue ?? [value]
        }
        guard let reply = messages.first(where: { $0["id"]?.intValue == id && ($0["result"] != nil || $0["error"] != nil) }) else {
            throw MCPError.invalidResponse(L("応答がありません", "no response"))
        }
        if let error = reply["error"] {
            throw MCPError.rpc(error["code"]?.intValue ?? 0, error["message"]?.stringValue ?? error.serialized())
        }
        return reply["result"] ?? .object([])
    }

    private func notify(_ method: String) async throws {
        let message = JSONValue.object([("jsonrpc", .string("2.0")), ("method", .string(method))])
        _ = try await post(message)
    }

    private func post(_ message: JSONValue) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.httpBody = Data(message.serialized().utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        if let negotiatedVersion { request.setValue(negotiatedVersion, forHTTPHeaderField: "MCP-Protocol-Version") }
        if let sessionID { request.setValue(sessionID, forHTTPHeaderField: "Mcp-Session-Id") }
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw MCPError.invalidResponse("not HTTP") }
        if let id = http.value(forHTTPHeaderField: "Mcp-Session-Id") { sessionID = id }
        if http.statusCode == 401 || http.statusCode == 403 { throw MCPError.unauthorized }
        guard (200..<300).contains(http.statusCode) else {
            throw MCPError.http(http.statusCode, String(String(decoding: data, as: UTF8.self).prefix(200)))
        }
        return (data, http)
    }

    /// SSE の data: 行を集めて JSON にする
    static func parseEventStream(_ text: String) -> [JSONValue] {
        var messages: [JSONValue] = []
        var dataLines: [String] = []
        func flush() {
            guard !dataLines.isEmpty else { return }
            if let value = try? JSONValue.parse(dataLines.joined(separator: "\n")) { messages.append(contentsOf: value.arrayValue ?? [value]) }
            dataLines = []
        }
        for line in text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n") {
            if line.isEmpty {
                flush()
            } else if line.hasPrefix("data:") {
                dataLines.append(String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces))
            }
        }
        flush()
        return messages
    }
}
