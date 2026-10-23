import Foundation
import Observation

/// 追加した MCP サーバー 1 つの設定。
struct MCPServerConfig: Codable, Identifiable, Hashable, Sendable {
    var id: UUID = UUID()
    var name: String
    var url: String
    var isEnabled: Bool = true
    /// 最後に取得したツール一覧（オフラインでも表示できるように保存）
    var tools: [MCPTool] = []
    /// 使わないツールの名前
    var disabledTools: Set<String> = []
    var lastError: String?
    var lastConnected: Date?

    /// トークンを保存するキーチェーンの名前
    var tokenKey: String { "mcp-token-\(id.uuidString)" }
}

/// ツールを呼んだ結果。
struct ToolOutcome: Sendable {
    var text: String
    var isError: Bool
}

/// コネクタ（モデルが使えるツール）の管理。組み込みの Web 検索・ページ取得と、追加した MCP サーバー。
@MainActor
@Observable
final class ConnectorStore {
    /// 組み込みツールの名前
    enum BuiltinTool: String, CaseIterable, Identifiable {
        case webSearch = "web_search"
        case fetchPage = "fetch_page"
        case currentTime = "current_time"

        var id: String { rawValue }

        var label: String {
            switch self {
            case .webSearch: return L("Web 検索", "Web search")
            case .fetchPage: return L("Web ページを読む", "Read web pages")
            case .currentTime: return L("現在の日時", "Current date and time")
            }
        }

        var definition: ToolDefinition {
            switch self {
            case .webSearch:
                return ToolDefinition(name: rawValue,
                                      description: L("Web を検索して、ページのタイトル・URL・要約を返す。最新の情報や知らないことを調べるときに使う。",
                                                     "Search the web and return titles, URLs and snippets. Use it for recent or unknown facts."),
                                      arguments: [("query", L("検索する言葉", "search words"), true)])
            case .fetchPage:
                return ToolDefinition(name: rawValue,
                                      description: L("URL のページを取得して本文のテキストを返す。", "Fetch a web page and return its text."),
                                      arguments: [("url", L("ページの URL", "page URL"), true)])
            case .currentTime:
                return ToolDefinition(name: rawValue,
                                      description: L("今の日付と時刻を返す。", "Return the current date and time."),
                                      arguments: [])
            }
        }
    }

    private(set) var enabledBuiltins: Set<String> {
        didSet { defaults.set(Array(enabledBuiltins), forKey: Keys.enabledBuiltins) }
    }
    var searchProvider: WebSearchProvider {
        didSet { defaults.set(searchProvider.rawValue, forKey: Keys.searchProvider) }
    }
    var braveAPIKey: String {
        didSet { Keychain.set(braveAPIKey, for: Keys.braveKey) }
    }
    var searxngURL: String {
        didSet { defaults.set(searxngURL, forKey: Keys.searxngURL) }
    }
    /// 検索結果の件数
    var resultCount: Int {
        didSet { defaults.set(resultCount, forKey: Keys.resultCount) }
    }
    private(set) var servers: [MCPServerConfig] {
        didSet { save() }
    }
    private(set) var refreshingServerIDs: Set<UUID> = []

    @ObservationIgnored private let defaults = UserDefaults.standard
    @ObservationIgnored private var clients: [UUID: MCPClient] = [:]

    init() {
        let defaults = UserDefaults.standard
        enabledBuiltins = Set(defaults.stringArray(forKey: Keys.enabledBuiltins)
                              ?? [BuiltinTool.webSearch.rawValue, BuiltinTool.fetchPage.rawValue, BuiltinTool.currentTime.rawValue])
        searchProvider = defaults.string(forKey: Keys.searchProvider).flatMap(WebSearchProvider.init(rawValue:)) ?? .duckDuckGo
        braveAPIKey = Keychain.get(Keys.braveKey) ?? ""
        searxngURL = defaults.string(forKey: Keys.searxngURL) ?? ""
        let count = defaults.integer(forKey: Keys.resultCount)
        resultCount = count > 0 ? count : 5
        if let data = defaults.data(forKey: Keys.servers),
           let saved = try? JSONDecoder().decode([MCPServerConfig].self, from: data) {
            servers = saved
        } else {
            servers = []
        }
    }

    // MARK: - 設定

    func isEnabled(_ tool: BuiltinTool) -> Bool { enabledBuiltins.contains(tool.rawValue) }

    func setEnabled(_ tool: BuiltinTool, _ enabled: Bool) {
        if enabled { enabledBuiltins.insert(tool.rawValue) } else { enabledBuiltins.remove(tool.rawValue) }
    }

    func addServer(name: String, url: String, token: String) -> MCPServerConfig {
        var config = MCPServerConfig(name: name.isEmpty ? (URL(string: url)?.host ?? "MCP") : name,
                                     url: url.trimmingCharacters(in: .whitespacesAndNewlines))
        config.isEnabled = true
        Keychain.set(token, for: config.tokenKey)
        servers.append(config)
        return config
    }

    func updateServer(_ config: MCPServerConfig, token: String?) {
        guard let index = servers.firstIndex(where: { $0.id == config.id }) else { return }
        servers[index] = config
        if let token { Keychain.set(token, for: config.tokenKey) }
        clients[config.id] = nil
    }

    func removeServer(_ id: UUID) {
        if let config = servers.first(where: { $0.id == id }) { Keychain.set("", for: config.tokenKey) }
        servers.removeAll { $0.id == id }
        clients[id] = nil
    }

    func token(for config: MCPServerConfig) -> String {
        Keychain.get(config.tokenKey) ?? ""
    }

    func setToolEnabled(_ name: String, in serverID: UUID, enabled: Bool) {
        guard let index = servers.firstIndex(where: { $0.id == serverID }) else { return }
        if enabled { servers[index].disabledTools.remove(name) } else { servers[index].disabledTools.insert(name) }
    }

    /// サーバーに接続してツール一覧を取り直す
    func refreshTools(_ serverID: UUID) async {
        guard let config = servers.first(where: { $0.id == serverID }) else { return }
        refreshingServerIDs.insert(serverID)
        defer { refreshingServerIDs.remove(serverID) }
        do {
            let tools = try await client(for: config).listTools()
            update(serverID) {
                $0.tools = tools
                $0.lastError = nil
                $0.lastConnected = Date()
            }
        } catch {
            clients[serverID] = nil
            update(serverID) { $0.lastError = error.localizedDescription }
        }
    }

    private func update(_ id: UUID, _ change: (inout MCPServerConfig) -> Void) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        change(&servers[index])
    }

    private func client(for config: MCPServerConfig) throws -> MCPClient {
        if let existing = clients[config.id] { return existing }
        guard let url = URL(string: config.url), url.scheme?.hasPrefix("http") == true else {
            throw WebToolError.invalidURL(config.url)
        }
        let client = MCPClient(endpoint: url, bearerToken: token(for: config))
        clients[config.id] = client
        return client
    }

    // MARK: - ツール

    /// 今使えるツールの定義（組み込み + 有効な MCP サーバーのツール）
    var availableTools: [ToolDefinition] {
        var tools = BuiltinTool.allCases.filter { isEnabled($0) }.map(\.definition)
        var names = Set(tools.map(\.name))
        for server in servers where server.isEnabled {
            for tool in server.tools where !server.disabledTools.contains(tool.name) && !names.contains(tool.name) {
                tools.append(tool.definition)
                names.insert(tool.name)
            }
        }
        return tools
    }

    var webSearchClient: WebSearchClient {
        WebSearchClient(provider: searchProvider, apiKey: braveAPIKey, searxngURL: searxngURL,
                        language: Locale.preferredLanguages.first?.hasPrefix("ja") == true ? "ja" : "en")
    }

    /// Web を検索する（「Web 検索して答える」ボタン用）
    func search(_ query: String) async throws -> [WebSearchResult] {
        try await webSearchClient.search(query, limit: resultCount)
    }

    /// ツールを実行する。結果は長すぎないように切る。
    func execute(_ call: ToolCall, maxCharacters: Int = 4000) async -> ToolOutcome {
        let outcome: ToolOutcome
        switch BuiltinTool(rawValue: call.name) {
        case .webSearch:
            let query = call.argument("query") ?? call.argument("q") ?? call.argument("input") ?? ""
            do {
                outcome = ToolOutcome(text: WebSearchClient.format(try await search(query)), isError: false)
            } catch {
                outcome = ToolOutcome(text: error.localizedDescription, isError: true)
            }
        case .fetchPage:
            let url = call.argument("url") ?? call.argument("input") ?? ""
            do {
                let page = try await webSearchClient.fetchPage(url, maxCharacters: maxCharacters)
                outcome = ToolOutcome(text: "# \(page.title)\n\(page.text)", isError: false)
            } catch {
                outcome = ToolOutcome(text: error.localizedDescription, isError: true)
            }
        case .currentTime:
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd (EEE) HH:mm zzz"
            outcome = ToolOutcome(text: formatter.string(from: Date()), isError: false)
        case nil:
            outcome = await executeMCP(call)
        }
        if outcome.text.count > maxCharacters {
            return ToolOutcome(text: String(outcome.text.prefix(maxCharacters)) + "…", isError: outcome.isError)
        }
        return outcome
    }

    private func executeMCP(_ call: ToolCall) async -> ToolOutcome {
        guard let server = servers.first(where: { $0.isEnabled && $0.tools.contains { $0.name == call.name } }) else {
            return ToolOutcome(text: L("ツール \(call.name) は見つかりません", "Tool \(call.name) was not found"), isError: true)
        }
        do {
            let result = try await client(for: server).callTool(call.name, arguments: call.arguments)
            return ToolOutcome(text: result.text, isError: result.isError)
        } catch {
            clients[server.id] = nil
            return ToolOutcome(text: error.localizedDescription, isError: true)
        }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(servers) { defaults.set(data, forKey: Keys.servers) }
    }

    private enum Keys {
        static let enabledBuiltins = "connectors.builtins"
        static let searchProvider = "connectors.searchProvider"
        static let braveKey = "connectors.braveKey"
        static let searxngURL = "connectors.searxngURL"
        static let resultCount = "connectors.resultCount"
        static let servers = "connectors.mcpServers"
    }
}
