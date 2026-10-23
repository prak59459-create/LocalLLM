import Foundation
import Observation

/// エージェントとのやり取り 1 件（.cdws/agent.json に保存する）。
struct AgentMessage: Codable, Identifiable, Hashable, Sendable {
    enum Role: String, Codable, Sendable {
        case user, assistant, tool
    }

    var id: UUID = UUID()
    var role: Role
    var content: String
    var toolCalls: [ToolCall] = []
    var toolName: String?
    var toolCallID: String?
    var isError = false
    var stats: GenerationStats?
    var date: Date = Date()
    /// この依頼でまとめて元に戻すための ID
    var batchID: UUID?

    var chatMessage: ChatMessage {
        switch role {
        case .user: return ChatMessage(role: .user, content: content)
        case .assistant: return ChatMessage(role: .assistant, content: content, toolCalls: toolCalls.isEmpty ? nil : toolCalls)
        case .tool: return ChatMessage(role: .tool, content: content, toolName: toolName, toolCallID: toolCallID)
        }
    }
}

/// 中央に表示するもの。
enum WorkspacePane: String, CaseIterable, Identifiable {
    case editor, preview, roadmap, changes

    var id: String { rawValue }

    var label: String {
        switch self {
        case .editor: return L("エディタ", "Editor")
        case .preview: return L("プレビュー", "Preview")
        case .roadmap: return L("ロードマップ", "Roadmap")
        case .changes: return L("変更履歴", "Changes")
        }
    }

    var symbolName: String {
        switch self {
        case .editor: return "chevron.left.forwardslash.chevron.right"
        case .preview: return "play.rectangle"
        case .roadmap: return "point.3.connected.trianglepath.dotted"
        case .changes: return "clock.arrow.circlepath"
        }
    }
}

/// エージェントがファイルを変える前の確認のしかた。
enum AgentApprovalMode: String, CaseIterable, Identifiable, Codable {
    /// 毎回たずねる
    case ask
    /// ファイルの作成・編集は自動で許可（削除だけたずねる）
    case autoEdits
    /// すべて自動
    case auto

    var id: String { rawValue }

    var label: String {
        switch self {
        case .ask: return L("毎回確認する", "Ask every time")
        case .autoEdits: return L("編集は自動で許可（削除は確認）", "Auto-approve edits (ask before deleting)")
        case .auto: return L("すべて自動で許可", "Approve everything")
        }
    }
}

/// 実行の許可を待っているツールの呼び出し。
struct PendingApproval: Identifiable {
    var id: String { call.id }
    var call: ToolCall
    var proposal: ProposedChange?
}

/// 開いているプロジェクト 1 つの状態: ファイルの木、エディタ、ロードマップ、変更履歴、エージェント。
@MainActor
@Observable
final class WorkspaceSession {
    let project: WorkspaceProject
    let fileSystem: WorkspaceFileSystem

    // ファイルの木
    private(set) var children: [String: [WorkspaceEntry]] = [:]
    var expanded: Set<String> = [""]
    private(set) var treeVersion = 0

    // エディタ
    private(set) var openTabs: [String] = []
    var selectedPath: String?
    private(set) var buffers: [String: String] = [:]
    private(set) var saved: [String: String] = [:]
    /// エージェントが変えたので、エディタの中身を入れ替える必要がある（パス → 版）
    private(set) var externalRevision: [String: Int] = [:]
    var pane: WorkspacePane = .editor

    // ロードマップと履歴
    var roadmap = Roadmap()
    var selectedNodeID: UUID?
    private(set) var history = ChangeHistory()

    // プレビューとコンソール
    private(set) var previewRevision = 0
    var previewFile: String?
    private(set) var consoleLines: [String] = []

    // エージェント
    private(set) var messages: [AgentMessage] = []
    private(set) var isAgentRunning = false
    private(set) var agentStatus: String?
    private(set) var streamingMessageID: UUID?
    private(set) var pendingApproval: PendingApproval?
    /// エージェントの欄を開いてほしいとき（ロードマップから依頼したときなど）に増える
    private(set) var agentPanelRequest = 0
    var notice: String?
    var approvalMode: AgentApprovalMode {
        didSet { UserDefaults.standard.set(approvalMode.rawValue, forKey: "cdws.approvalMode") }
    }
    var toolProfile: WorkspaceToolbox.Profile {
        didSet { UserDefaults.standard.set(toolProfile.rawValue, forKey: "cdws.toolProfile") }
    }
    var useWebTools: Bool {
        didSet { UserDefaults.standard.set(useWebTools, forKey: "cdws.useWebTools") }
    }

    @ObservationIgnored private var agentTask: Task<Void, Never>?
    @ObservationIgnored private var approvalContinuation: CheckedContinuation<Bool, Never>?
    @ObservationIgnored private var saveTasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored private let javaScript = JavaScriptRunner()

    init(project: WorkspaceProject, root: URL, initialRoadmap: Roadmap? = nil) {
        self.project = project
        fileSystem = WorkspaceFileSystem(root: root)
        let defaults = UserDefaults.standard
        approvalMode = defaults.string(forKey: "cdws.approvalMode").flatMap(AgentApprovalMode.init(rawValue:)) ?? .ask
        toolProfile = defaults.string(forKey: "cdws.toolProfile").flatMap(WorkspaceToolbox.Profile.init(rawValue:)) ?? .standard
        useWebTools = defaults.object(forKey: "cdws.useWebTools") as? Bool ?? false
        loadState()
        if let initialRoadmap, roadmap.nodes.isEmpty {
            roadmap = initialRoadmap
            saveRoadmap()
        }
        refreshTree()
        if let main = ProjectTemplates.template(id: project.templateID ?? "")?.mainFile, fileSystem.exists(main) {
            open(main)
        }
    }

    // MARK: - 保存（.cdws フォルダ）

    private var stateFolder: URL { fileSystem.root.appendingPathComponent(".cdws", isDirectory: true) }

    private func loadState() {
        let decoder = JSONDecoder()
        if let data = try? Data(contentsOf: stateFolder.appendingPathComponent("roadmap.json")),
           let value = try? decoder.decode(Roadmap.self, from: data) {
            roadmap = value
        }
        if let data = try? Data(contentsOf: stateFolder.appendingPathComponent("history.json")),
           let value = try? decoder.decode(ChangeHistory.self, from: data) {
            history = value
        }
        if let data = try? Data(contentsOf: stateFolder.appendingPathComponent("agent.json")),
           let value = try? decoder.decode([AgentMessage].self, from: data) {
            messages = value
        }
    }

    private func write<T: Encodable>(_ value: T, to name: String) {
        try? FileManager.default.createDirectory(at: stateFolder, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(value) {
            try? data.write(to: stateFolder.appendingPathComponent(name), options: .atomic)
        }
    }

    func saveRoadmap() { write(roadmap, to: "roadmap.json") }
    private func saveHistory() { write(history, to: "history.json") }
    private func saveMessages() { write(Array(messages.suffix(200)), to: "agent.json") }

    // MARK: - ファイルの木

    func refreshTree() {
        var next: [String: [WorkspaceEntry]] = [:]
        for folder in expanded where fileSystem.isDirectory(folder) {
            next[folder] = (try? fileSystem.children(of: folder)) ?? []
        }
        children = next
        expanded = expanded.filter { fileSystem.isDirectory($0) }
        treeVersion += 1
    }

    func toggle(_ folder: String) {
        if expanded.contains(folder) {
            expanded.remove(folder)
        } else {
            expanded.insert(folder)
        }
        refreshTree()
    }

    /// 木に並べる行（開いているフォルダの中身を順に）
    var visibleEntries: [WorkspaceEntry] {
        var result: [WorkspaceEntry] = []
        func add(_ folder: String) {
            for entry in children[folder] ?? [] {
                result.append(entry)
                if entry.isDirectory && expanded.contains(entry.path) { add(entry.path) }
            }
        }
        add("")
        return result
    }

    func createFile(named name: String, in folder: String) {
        let path = folder.isEmpty ? name : folder + "/" + name
        do {
            guard !fileSystem.exists(path) else { throw WorkspaceError.alreadyExists(path) }
            var change = try fileSystem.write(path, "")
            change.actor = "user"
            history.record(change)
            saveHistory()
            expanded.insert(folder)
            refreshTree()
            open(change.path)
        } catch {
            notice = error.localizedDescription
        }
    }

    func createFolder(named name: String, in folder: String) {
        let path = folder.isEmpty ? name : folder + "/" + name
        do {
            try fileSystem.makeDirectory(path)
            expanded.insert(folder)
            expanded.insert(try fileSystem.normalize(path))
            refreshTree()
        } catch {
            notice = error.localizedDescription
        }
    }

    func rename(_ path: String, to name: String) {
        let parent = (path as NSString).deletingLastPathComponent
        let destination = parent.isEmpty ? name : parent + "/" + name
        do {
            var change = try fileSystem.move(path, to: destination)
            change.actor = "user"
            history.record(change)
            saveHistory()
            retarget(from: path, to: change.path)
            refreshTree()
        } catch {
            notice = error.localizedDescription
        }
    }

    func delete(_ path: String) {
        do {
            let changes = try fileSystem.delete(path).map { change -> FileChange in
                var change = change
                change.actor = "user"
                return change
            }
            history.record(changes)
            saveHistory()
            for tab in openTabs where tab == path || tab.hasPrefix(path + "/") { close(tab) }
            refreshTree()
        } catch {
            notice = error.localizedDescription
        }
    }

    /// 移動・名前の変更に合わせて、開いているタブのパスを変える
    private func retarget(from old: String, to new: String) {
        func moved(_ path: String) -> String? {
            if path == old { return new }
            if path.hasPrefix(old + "/") { return new + String(path.dropFirst(old.count)) }
            return nil
        }
        for (index, tab) in openTabs.enumerated() {
            guard let target = moved(tab) else { continue }
            openTabs[index] = target
            buffers[target] = buffers.removeValue(forKey: tab)
            saved[target] = saved.removeValue(forKey: tab)
        }
        if let selected = selectedPath, let target = moved(selected) { selectedPath = target }
    }

    // MARK: - エディタ

    func open(_ path: String) {
        guard let normalized = try? fileSystem.normalize(path), !fileSystem.isDirectory(normalized) else { return }
        if buffers[normalized] == nil {
            do {
                let text = try fileSystem.read(normalized)
                buffers[normalized] = text
                saved[normalized] = text
            } catch {
                notice = error.localizedDescription
                return
            }
        }
        if !openTabs.contains(normalized) { openTabs.append(normalized) }
        selectedPath = normalized
        if normalized.lowercased().hasSuffix(".html") || normalized.lowercased().hasSuffix(".htm") { previewFile = normalized }
        if pane != .editor && pane != .preview { pane = .editor }
    }

    func close(_ path: String) {
        if isDirty(path) { save(path) }
        openTabs.removeAll { $0 == path }
        buffers[path] = nil
        saved[path] = nil
        if selectedPath == path { selectedPath = openTabs.last }
    }

    func isDirty(_ path: String) -> Bool {
        guard let buffer = buffers[path] else { return false }
        return buffer != saved[path]
    }

    func text(for path: String) -> String { buffers[path] ?? "" }

    func revision(for path: String) -> Int { externalRevision[path] ?? 0 }

    /// 入力のたびに呼ぶ。少し止まったら自動で保存する。
    func updateBuffer(_ path: String, text: String) {
        guard buffers[path] != text else { return }
        buffers[path] = text
        saveTasks[path]?.cancel()
        saveTasks[path] = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            guard !Task.isCancelled else { return }
            self?.save(path)
        }
    }

    func save(_ path: String) {
        saveTasks[path]?.cancel()
        saveTasks[path] = nil
        guard let buffer = buffers[path], buffer != saved[path] else { return }
        do {
            var change = try fileSystem.write(path, buffer)
            change.actor = "user"
            // 打つたびに履歴が増えすぎないよう、直前の自分の変更とまとめる
            if let last = history.changes.last, last.actor == "user", last.path == path, last.kind == .modified,
               Date().timeIntervalSince(last.date) < 120 {
                change.before = last.before
                history.changes.removeLast()
            }
            history.record(change)
            saveHistory()
            saved[path] = buffer
            previewRevision += 1
        } catch {
            notice = error.localizedDescription
        }
    }

    func saveAll() {
        for path in openTabs where isDirty(path) { save(path) }
    }

    /// エージェントや取り消しでディスクの中身が変わったファイルを読み直す
    private func reloadFromDisk(_ paths: [String]) {
        for path in paths {
            guard buffers[path] != nil else { continue }
            if let text = try? fileSystem.read(path) {
                buffers[path] = text
                saved[path] = text
                externalRevision[path, default: 0] += 1
            } else {
                close(path)
            }
        }
    }

    // MARK: - 変更履歴

    var changes: [FileChange] { history.changes }

    func undo(_ change: FileChange) {
        do {
            try history.undo(change.id, in: fileSystem)
            saveHistory()
            reloadFromDisk([change.path] + (change.movedFrom.map { [$0] } ?? []))
            refreshTree()
            previewRevision += 1
        } catch {
            notice = error.localizedDescription
        }
    }

    func undoBatch(_ batchID: UUID) {
        let paths = history.changes.filter { $0.batchID == batchID }.map(\.path)
        do {
            try history.undoBatch(batchID, in: fileSystem)
            saveHistory()
            reloadFromDisk(paths)
            refreshTree()
            previewRevision += 1
        } catch {
            notice = error.localizedDescription
        }
    }

    func hasChanges(in batchID: UUID) -> Bool {
        history.changes.contains { $0.batchID == batchID }
    }

    // MARK: - ロードマップ

    var selectedNode: RoadmapNode? { selectedNodeID.flatMap { roadmap.node($0) } }

    func addNode(title: String, parent: UUID?, at point: (x: Double, y: Double)? = nil) {
        let node = roadmap.add(title: title, kind: parent == nil ? .feature : .idea, status: .idea, parent: parent, at: point)
        selectedNodeID = node.id
        saveRoadmap()
    }

    func updateNode(_ id: UUID, _ change: (inout RoadmapNode) -> Void) {
        roadmap.update(id, change)
        saveRoadmap()
    }

    func moveNode(_ id: UUID, to point: (x: Double, y: Double)) {
        roadmap.move(id, to: point)
        saveRoadmap()
    }

    func removeNode(_ id: UUID) {
        roadmap.remove(id)
        if selectedNodeID == id { selectedNodeID = nil }
        saveRoadmap()
    }

    func connect(_ from: UUID, _ to: UUID) {
        roadmap.connect(from, to)
        saveRoadmap()
    }

    func autoLayout() {
        roadmap.autoLayout()
        saveRoadmap()
    }

    // MARK: - プレビューと JavaScript

    func reloadPreview() { previewRevision += 1 }

    func appendConsole(_ line: String) {
        consoleLines.append(line)
        if consoleLines.count > 500 { consoleLines.removeFirst(consoleLines.count - 500) }
    }

    func clearConsole() { consoleLines = [] }

    func runJavaScript(path: String) {
        saveAll()
        guard let code = try? fileSystem.read(path) else { return }
        appendConsole("▶︎ " + path)
        Task {
            let result = await javaScript.run(code)
            appendConsole(result.output)
        }
    }

    /// プレビュー用に、プロジェクトの Web 用ファイルをアプリの一時フォルダに写す（外のフォルダは Web ページから直接読めないため）
    func preparePreviewFolder() -> URL? {
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("cdws-preview", isDirectory: true)
            .appendingPathComponent(project.id.uuidString, isDirectory: true)
        try? FileManager.default.removeItem(at: destination)
        try? FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        var total: Int64 = 0
        for entry in (try? fileSystem.walk(limit: 3000)) ?? [] where !entry.isDirectory {
            total += entry.size
            if total > 80_000_000 { break }
            guard let source = try? fileSystem.url(for: entry.path) else { continue }
            let target = destination.appendingPathComponent(entry.path)
            try? FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.copyItem(at: source, to: target)
        }
        return destination
    }

    /// プレビューで最初に開く HTML
    var previewEntry: String? {
        if let previewFile, fileSystem.exists(previewFile) { return previewFile }
        if fileSystem.exists("index.html") { return "index.html" }
        return (try? fileSystem.find("*.html", limit: 1))?.first
    }

    // MARK: - エージェント

    /// 依頼を送って、エージェントに作業させる。
    func sendToAgent(_ text: String, chat: ChatController, settings: AppSettings, connectors: ConnectorStore) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isAgentRunning else { return }
        guard let info = chat.loadedInfo else {
            notice = L("先にエージェントの欄の上にある「モデルを選ぶ」から、モデルを読み込んでください（ツールの呼び出しに対応したモデルがおすすめです）。",
                       "Load a model first with \"Choose a model\" at the top of the agent panel (one that supports tool calling works best).")
            return
        }
        chat.stopGenerating()
        saveAll()
        let batchID = UUID()
        messages.append(AgentMessage(role: .user, content: trimmed, batchID: batchID))
        saveMessages()

        var parameters = settings.generation
        parameters.temperature = min(parameters.temperature, 0.4)
        parameters.repetitionPenalty = 1.0
        parameters.maxNewTokens = max(parameters.maxNewTokens, 2048)
        let javaScriptEnabled = toolProfile != .minimal
        var tools = WorkspaceToolbox.definitions(profile: toolProfile, javaScript: javaScriptEnabled)
        if useWebTools {
            tools += [ConnectorStore.BuiltinTool.webSearch, .fetchPage].filter { connectors.isEnabled($0) }.map(\.definition)
        }
        let notes = try? fileSystem.read("CDWS.md")
        let system = CodingAgentPrompt.system(projectName: project.name,
                                              tree: fileSystem.tree(maxDepth: 3, maxEntries: info.maxContext > 4096 ? 150 : 60),
                                              roadmap: roadmap.summary(maxNodes: info.maxContext > 4096 ? 40 : 15),
                                              notes: notes, openFile: selectedPath, javaScript: javaScriptEnabled)
        let configuration = AgentLoop.Configuration(systemPrompt: system, parameters: parameters,
                                                    enableThinking: settings.thinkingMode.enableThinking,
                                                    tools: tools, maxRounds: 12)
        let history = messages.suffix(40).map(\.chatMessage)
        isAgentRunning = true
        agentStatus = L("考えています…", "Thinking…")
        notice = nil

        let loop = AgentLoop(engine: chat.engine)
        agentTask = Task {
            do {
                try await loop.run(history: Array(history), configuration: configuration,
                                   execute: { call in await self.executeTool(call, batchID: batchID, connectors: connectors) },
                                   approve: { call in await self.approve(call) },
                                   onEvent: { event in self.handle(event, batchID: batchID) })
            } catch is CancellationError {
            } catch {
                notice = error.localizedDescription
            }
            if let id = streamingMessageID, let index = messages.firstIndex(where: { $0.id == id }),
               messages[index].content.isEmpty, messages[index].toolCalls.isEmpty {
                messages.remove(at: index)
            }
            streamingMessageID = nil
            isAgentRunning = false
            agentStatus = nil
            agentTask = nil
            saveMessages()
            refreshTree()
        }
    }

    func requestAgentPanel() { agentPanelRequest += 1 }

    func stopAgent() {
        agentTask?.cancel()
        resolveApproval(false)
    }

    func clearConversation() {
        guard !isAgentRunning else { return }
        messages = []
        saveMessages()
    }

    private func handle(_ event: AgentLoop.Event, batchID: UUID) {
        switch event {
        case .assistantStarted:
            let message = AgentMessage(role: .assistant, content: "", batchID: batchID)
            messages.append(message)
            streamingMessageID = message.id
            agentStatus = L("考えています…", "Thinking…")
        case .promptReady(_, let dropped):
            if dropped > 0 {
                notice = L("コンテキストに収まらないため、古いやり取り \(dropped) 件を省きました。",
                           "\(dropped) older messages were left out to fit the context.")
            }
        case .prefill(let done, let total):
            agentStatus = L("読み込み \(done)/\(total)", "Reading \(done)/\(total)")
        case .text(let visible):
            guard let id = streamingMessageID, let index = messages.firstIndex(where: { $0.id == id }) else { return }
            // 画面の更新が重くならないよう、少しずつ伸びたときだけ反映する
            if visible.count - messages[index].content.count >= 6 || visible.hasSuffix("\n") {
                messages[index].content = visible
            }
            agentStatus = L("書いています…", "Writing…")
        case .assistantFinished(let content, let calls, let stats):
            guard let id = streamingMessageID, let index = messages.firstIndex(where: { $0.id == id }) else { return }
            messages[index].content = content
            messages[index].toolCalls = calls
            messages[index].stats = stats
            if content.isEmpty && calls.isEmpty { messages.remove(at: index) }
            streamingMessageID = nil
            saveMessages()
        case .toolStarted(let call):
            agentStatus = L("実行中: ", "Running: ") + call.summary
        case .toolFinished(let call, let outcome):
            messages.append(AgentMessage(role: .tool, content: outcome.text, toolName: call.name, toolCallID: call.id,
                                         isError: outcome.isError, batchID: batchID))
            saveMessages()
        }
    }

    private func executeTool(_ call: ToolCall, batchID: UUID, connectors: ConnectorStore) async -> ToolOutcome {
        if WorkspaceToolbox.isWorkspaceTool(call.name) {
            let executor = WorkspaceToolExecutor(fileSystem: fileSystem)
            let result = executor.execute(call, roadmap: &roadmap)
            if !result.changes.isEmpty {
                history.record(result.changes.map { change -> FileChange in
                    var change = change
                    change.batchID = batchID
                    return change
                })
                saveHistory()
                reloadFromDisk(result.changes.map(\.path) + result.changes.compactMap(\.movedFrom))
                refreshTree()
                previewRevision += 1
                if let path = result.touchedFile, fileSystem.exists(path) {
                    if buffers[path] == nil { open(path) } else { selectedPath = path }
                }
            }
            if result.roadmapChanged { saveRoadmap() }
            return ToolOutcome(text: result.text, isError: result.isError)
        }
        if call.name == WorkspaceToolbox.Name.runJavaScript {
            var code = WorkspaceToolExecutor.argument(call, ["code", "script", "source", "js"]) ?? ""
            if code.isEmpty, let path = WorkspaceToolExecutor.path(call) {
                code = (try? fileSystem.read(path)) ?? ""
            }
            guard !code.isEmpty else { return ToolOutcome(text: L("code がありません", "Missing code"), isError: true) }
            let result = await javaScript.run(code)
            appendConsole("▶︎ run_javascript\n" + result.output)
            return ToolOutcome(text: result.output, isError: result.isError)
        }
        return await connectors.execute(call)
    }

    private func approve(_ call: ToolCall) async -> Bool {
        guard WorkspaceToolbox.mutatingTools.contains(call.name) else { return true }
        switch approvalMode {
        case .auto: return true
        case .autoEdits where !WorkspaceToolbox.destructiveTools.contains(call.name): return true
        default: break
        }
        let proposal = WorkspaceToolExecutor(fileSystem: fileSystem).preview(call)
        agentStatus = L("許可を待っています", "Waiting for approval")
        return await withCheckedContinuation { continuation in
            approvalContinuation = continuation
            pendingApproval = PendingApproval(call: call, proposal: proposal)
        }
    }

    /// 確認カードのボタンから呼ぶ。always なら、以後の編集は自動で許可する。
    func resolveApproval(_ allowed: Bool, always: Bool = false) {
        if always { approvalMode = .autoEdits }
        pendingApproval = nil
        approvalContinuation?.resume(returning: allowed)
        approvalContinuation = nil
    }
}
