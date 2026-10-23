import SwiftUI

/// コネクタの設定: 組み込みツール（Web 検索など）と MCP サーバー。
/// 型チェックを速くするため、セクションごとに別の View にしている。
struct ConnectorsView: View {
    @Environment(ConnectorStore.self) private var connectors
    @State private var addingServer = false

    var body: some View {
        Form {
            ToolUseSection()
            BuiltinToolsSection()
            WebSearchSettingsSection()
            MCPServersSection(onAdd: { addingServer = true })
        }
        .navigationTitle(L("コネクタ", "Connectors"))
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $addingServer) {
            NavigationStack {
                MCPServerEditor(server: nil)
            }
        }
    }
}

private struct ToolUseSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            Toggle(L("チャットでモデルにツールを使わせる", "Let the model use tools in chats"), isOn: $settings.toolsEnabled)
        } footer: {
            Text(L("オンにすると、モデルが必要に応じて Web 検索や MCP サーバーのツールを呼び出します。ツールの呼び出しに対応したモデル（Qwen2.5 / Qwen3 / Llama 3.1 以降など）で特によく動きます。",
                   "When on, the model can call web search and MCP tools when needed. Works best with models trained for tool calling (Qwen2.5 / Qwen3 / Llama 3.1+ and similar)."))
        }
    }
}

private struct BuiltinToolsSection: View {
    @Environment(ConnectorStore.self) private var connectors

    var body: some View {
        Section {
            ForEach(ConnectorStore.BuiltinTool.allCases) { tool in
                Toggle(isOn: Binding(get: { connectors.isEnabled(tool) },
                                     set: { connectors.setEnabled(tool, $0) })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Label(tool.label, systemImage: ToolResultView.icon(for: tool.rawValue))
                        Text(tool.rawValue)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        } header: {
            Text(L("組み込みのツール", "Built-in tools"))
        }
    }
}

private struct WebSearchSettingsSection: View {
    @Environment(ConnectorStore.self) private var connectors
    @State private var testQuery = ""
    @State private var testResult: String?
    @State private var isTesting = false

    var body: some View {
        @Bindable var connectors = connectors
        Section {
            Picker(L("検索サービス", "Search service"), selection: $connectors.searchProvider) {
                ForEach(WebSearchProvider.allCases) { provider in
                    Text(provider.label).tag(provider)
                }
            }
            if connectors.searchProvider == .brave {
                SecureField(L("Brave Search の API キー", "Brave Search API key"), text: $connectors.braveAPIKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }
            if connectors.searchProvider == .searxng {
                TextField("https://searx.example.com", text: $connectors.searxngURL)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }
            Stepper(value: $connectors.resultCount, in: 1...10) {
                LabeledContent(L("結果の件数", "Results"), value: "\(connectors.resultCount)")
            }
            HStack {
                TextField(L("試しに検索", "Try a search"), text: $testQuery)
                    .onSubmit(runTest)
                if isTesting {
                    ProgressView()
                } else {
                    Button(L("検索", "Search"), action: runTest)
                        .disabled(testQuery.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            if let testResult {
                Text(testResult)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
        } header: {
            Text(L("Web 検索", "Web search"))
        } footer: {
            Text(L("検索するときだけ、検索する言葉が選んだサービスに送られます。モデルや会話の中身は端末の外に出ません。",
                   "Only the search words are sent to the chosen service, and only when searching. Your model and chats stay on the device."))
        }
    }

    private func runTest() {
        let query = testQuery.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty, !isTesting else { return }
        isTesting = true
        Task {
            do {
                let results = try await connectors.search(query)
                testResult = results.isEmpty ? L("見つかりませんでした", "No results") : WebSearchClient.format(results)
            } catch {
                testResult = error.localizedDescription
            }
            isTesting = false
        }
    }
}

private struct MCPServersSection: View {
    @Environment(ConnectorStore.self) private var connectors
    var onAdd: () -> Void

    var body: some View {
        Section {
            ForEach(connectors.servers) { server in
                NavigationLink {
                    MCPServerEditor(server: server)
                } label: {
                    MCPServerRow(server: server)
                }
            }
            Button(action: onAdd) {
                Label(L("MCP サーバーを追加", "Add MCP server"), systemImage: "plus")
            }
        } header: {
            Text(L("MCP サーバー", "MCP servers"))
        } footer: {
            Text(L("Streamable HTTP に対応した MCP サーバーの URL を追加すると、そのツールをモデルが使えるようになります。トークンはキーチェーンに保存されます。",
                   "Add the URL of an MCP server that supports Streamable HTTP and the model can use its tools. Tokens are stored in the Keychain."))
        }
    }
}

private struct MCPServerRow: View {
    @Environment(ConnectorStore.self) private var connectors
    let server: MCPServerConfig

    var body: some View {
        HStack {
            Image(systemName: "server.rack")
                .foregroundStyle(server.isEnabled ? Color.accentColor : Color.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(server.name)
                Text(status)
                    .font(.caption)
                    .foregroundStyle(server.lastError == nil ? Color.secondary : Color.orange)
                    .lineLimit(1)
            }
            Spacer()
            if connectors.refreshingServerIDs.contains(server.id) {
                ProgressView()
            }
        }
    }

    private var status: String {
        if let error = server.lastError { return error }
        let active = server.tools.filter { !server.disabledTools.contains($0.name) }.count
        if server.tools.isEmpty { return server.url }
        return L("ツール \(active)/\(server.tools.count)", "\(active)/\(server.tools.count) tools")
    }
}

/// MCP サーバーの追加・編集。server が nil なら追加。
struct MCPServerEditor: View {
    @Environment(ConnectorStore.self) private var connectors
    @Environment(\.dismiss) private var dismiss

    let server: MCPServerConfig?

    @State private var name = ""
    @State private var url = ""
    @State private var token = ""
    @State private var isEnabled = true
    @State private var loaded = false

    private var current: MCPServerConfig? {
        guard let server else { return nil }
        return connectors.servers.first { $0.id == server.id }
    }

    var body: some View {
        Form {
            Section {
                TextField(L("名前", "Name"), text: $name)
                TextField("https://example.com/mcp", text: $url)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField(L("トークン（必要なら）", "Token (if needed)"), text: $token)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                if server != nil {
                    Toggle(L("使う", "Enabled"), isOn: $isEnabled)
                }
            }
            if let current {
                MCPToolsSection(server: current)
                Section {
                    Button(L("サーバーを削除", "Remove server"), role: .destructive) {
                        connectors.removeServer(current.id)
                        dismiss()
                    }
                }
            }
        }
        .navigationTitle(server == nil ? L("MCP サーバーを追加", "Add MCP Server") : (current?.name ?? ""))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if server == nil {
                ToolbarItem(placement: .cancellationAction) {
                    Button(L("キャンセル", "Cancel")) { dismiss() }
                }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button(server == nil ? L("追加", "Add") : L("保存", "Save"), action: save)
                    .disabled(URL(string: url.trimmingCharacters(in: .whitespaces))?.scheme?.hasPrefix("http") != true)
            }
        }
        .onAppear {
            guard !loaded else { return }
            loaded = true
            if let current {
                name = current.name
                url = current.url
                isEnabled = current.isEnabled
                token = connectors.token(for: current)
            }
        }
    }

    private func save() {
        if var current {
            current.name = name
            current.url = url.trimmingCharacters(in: .whitespacesAndNewlines)
            current.isEnabled = isEnabled
            connectors.updateServer(current, token: token)
            Task { await connectors.refreshTools(current.id) }
        } else {
            let added = connectors.addServer(name: name, url: url, token: token)
            Task { await connectors.refreshTools(added.id) }
            dismiss()
        }
    }
}

private struct MCPToolsSection: View {
    @Environment(ConnectorStore.self) private var connectors
    let server: MCPServerConfig

    var body: some View {
        Section {
            if let error = server.lastError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            ForEach(server.tools, id: \.name) { tool in
                Toggle(isOn: Binding(get: { !server.disabledTools.contains(tool.name) },
                                     set: { connectors.setToolEnabled(tool.name, in: server.id, enabled: $0) })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(tool.title ?? tool.name)
                        Text(tool.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                }
            }
            Button {
                Task { await connectors.refreshTools(server.id) }
            } label: {
                HStack {
                    Label(L("接続してツールを取得", "Connect and fetch tools"), systemImage: "arrow.clockwise")
                    if connectors.refreshingServerIDs.contains(server.id) {
                        Spacer()
                        ProgressView()
                    }
                }
            }
        } header: {
            Text(L("ツール", "Tools"))
        } footer: {
            if let date = server.lastConnected {
                Text(L("最後の接続: \(date.formatted(date: .abbreviated, time: .shortened))",
                       "Last connected: \(date.formatted(date: .abbreviated, time: .shortened))"))
            }
        }
    }
}
