import Foundation
import Observation
import SwiftData
import UIKit

/// モデルの読み込み状態。
enum ModelLoadState: Equatable {
    case idle
    case compiling(String)      // .mlpackage をコンパイル中（モデル名）
    case loading(String)        // 読み込み中（モデル名）
    case ready(LoadedModelInfo)
    case failed(String)
}

/// 生成中の状態（UI 表示用）。
struct GenerationProgress: Equatable {
    var conversationID: UUID
    var messageID: UUID
    var phase: Phase
    var promptTokens: Int = 0
    var prefillDone: Int = 0
    var prefillTotal: Int = 0
    var generatedCharacters: Int = 0

    enum Phase: Equatable {
        case preparing
        case searching
        case prefilling
        case generating
        case usingTool(String)
    }
}

/// 送信のときの追加の指定
struct SendOptions {
    /// 送る前に Web を検索して、結果を一緒にモデルへ渡す
    var searchWeb = false
}

/// モデルの読み込みと、会話への返答生成を受け持つ。
@MainActor
@Observable
final class ChatController {
    private(set) var loadState: ModelLoadState = .idle
    private(set) var progress: GenerationProgress?
    /// 直近の生成の統計
    private(set) var lastStats: GenerationStats?
    /// 画面上部に出すお知らせ（古いメッセージを省いた、など）
    var notice: String?
    /// 統計表示用のメモリ情報（定期更新）
    private(set) var memoryFootprint: Int64?
    private(set) var memoryAvailable: Int64 = 0

    @ObservationIgnored let engine = InferenceEngine()
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var loadTask: Task<Void, Never>?
    @ObservationIgnored private var memoryTimer: Timer?
    @ObservationIgnored private var lastSaveDate = Date.distantPast

    @ObservationIgnored private var memoryWarningObserver: NSObjectProtocol?

    init() {
        refreshMemory()
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.refreshMemory() }
        }
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                // 生成は控えめに続け、危なくなったら自分で止める（KV キャッシュも空ける）
                await self.engine.noteMemoryWarning()
                self.notice = L("メモリが少なくなっています。落ちないよう計算を小さく区切り、足りなくなったら途中で止めます。",
                                "Memory is getting low. Work is split into smaller steps and will stop early instead of crashing.")
                self.refreshMemory()
            }
        }
    }

    var loadedInfo: LoadedModelInfo? {
        if case .ready(let info) = loadState { return info }
        return nil
    }

    var isGenerating: Bool { progress != nil }

    var isBusyLoading: Bool {
        switch loadState {
        case .compiling, .loading: return true
        default: return false
        }
    }

    func refreshMemory() {
        memoryFootprint = DeviceMemory.footprintBytes
        memoryAvailable = DeviceMemory.availableBytes
    }

    // MARK: - モデルの読み込み

    /// モデルを読み込む（読み込み中・生成中なら止めてから）。
    func loadModel(id: String, library: ModelLibrary, settings: AppSettings) {
        stopGenerating()
        loadTask?.cancel()
        guard let model = library.model(withID: id) else {
            loadState = .failed(ModelLibraryError.missingModelFolder.localizedDescription)
            return
        }
        settings.activeModelID = id
        loadTask = Task {
            do {
                if library.needsCompilation(model) {
                    loadState = .compiling(model.displayName)
                } else {
                    loadState = .loading(model.displayName)
                }
                let files = try await library.prepareFiles(for: id, client: settings.makeHFClient())
                loadState = .loading(model.displayName)
                await engine.setContextLimit(settings.contextLimit)
                let info = try await engine.load(files: files,
                                                 computeUnits: settings.computeUnits,
                                                 prefillChunkSize: settings.prefillChunkSize)
                if Task.isCancelled { return }
                loadState = .ready(info)
            } catch is CancellationError {
                loadState = .idle
            } catch {
                loadState = .failed(error.localizedDescription)
            }
            refreshMemory()
        }
    }

    /// モデルを解放して、終わるまで待つ（モデルのファイルを書き換える前など）。
    func unloadModelAndWait() async {
        stopGenerating()
        loadTask?.cancel()
        await engine.unload()
        loadState = .idle
        refreshMemory()
    }

    /// 実際に使えるコンテキスト長（設定の上限を考えたもの）
    func effectiveContext(settings: AppSettings) -> Int? {
        guard let info = loadedInfo else { return nil }
        return settings.contextLimit > 0 ? min(info.maxContext, max(settings.contextLimit, 256)) : info.maxContext
    }

    /// モデルを解放する。
    func unloadModel() {
        stopGenerating()
        loadTask?.cancel()
        Task {
            await engine.unload()
            loadState = .idle
            refreshMemory()
        }
    }

    // MARK: - 生成

    /// ユーザーの発言を追加して、返答を生成する。
    /// アシスタントに参考資料があれば、発言に関係する部分を検索して一緒にモデルへ渡す。
    func send(_ text: String, in conversation: Conversation, context: ModelContext, settings: AppSettings,
              assistant: Assistant? = nil, knowledge: KnowledgeStore? = nil, connectors: ConnectorStore? = nil,
              options: SendOptions = SendOptions()) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }
        guard loadedInfo != nil else {
            notice = L("先にモデルを選んで読み込んでください。", "Please choose and load a model first.")
            return
        }
        let nextOrder = (conversation.orderedMessages.last?.order ?? -1) + 1
        let userMessage = StoredMessage(role: .user, content: trimmed, order: nextOrder)
        if let assistant, assistant.useKnowledge, let knowledge {
            attachKnowledge(to: userMessage, query: trimmed, assistant: assistant, knowledge: knowledge)
        }
        context.insert(userMessage)
        conversation.messages.append(userMessage)
        if conversation.title.isEmpty {
            conversation.title = Conversation.makeTitle(from: trimmed)
        }
        conversation.updatedAt = Date()
        try? context.save()

        guard options.searchWeb, let connectors else {
            generateReply(in: conversation, context: context, settings: settings, assistant: assistant, connectors: connectors)
            return
        }
        // 先に Web を検索して、結果を発言に付けてから生成する
        progress = GenerationProgress(conversationID: conversation.id, messageID: userMessage.id, phase: .searching)
        generationTask = Task {
            do {
                let results = try await connectors.search(trimmed)
                attachWebResults(results, to: userMessage)
                try? context.save()
            } catch {
                notice = error.localizedDescription
            }
            progress = nil
            generationTask = nil
            if !Task.isCancelled {
                generateReply(in: conversation, context: context, settings: settings, assistant: assistant, connectors: connectors)
            }
        }
    }

    /// 最後のアシスタントの返答を作り直す（ツールのやり取りも含めて消す）。
    func regenerate(in conversation: Conversation, context: ModelContext, settings: AppSettings,
                    assistant: Assistant? = nil, connectors: ConnectorStore? = nil) {
        guard !isGenerating else { return }
        let ordered = conversation.orderedMessages
        guard let lastUser = ordered.lastIndex(where: { $0.role == .user }) else { return }
        for message in ordered[(lastUser + 1)...] {
            conversation.messages.removeAll { $0.id == message.id }
            context.delete(message)
        }
        try? context.save()
        generateReply(in: conversation, context: context, settings: settings, assistant: assistant, connectors: connectors)
    }

    /// 参考資料を検索して、ユーザーの発言に付ける。
    private func attachKnowledge(to message: StoredMessage, query: String, assistant: Assistant, knowledge: KnowledgeStore) {
        let results = knowledge.search(query, in: assistant.id, limit: max(1, assistant.knowledgeResultCount))
        guard !results.isEmpty else { return }
        // コンテキスト長に合わせて入れる量を決める（おおよそ 1 トークン = 1〜3 文字）
        let contextLength = loadedInfo?.maxContext ?? 2048
        let maxCharacters = min(6000, max(600, contextLength))
        let block = KnowledgePromptBuilder.contextBlock(results: results, maxCharacters: maxCharacters)
        guard !block.isEmpty else { return }
        message.attachedContext = block
        message.sources = results.map { "\($0.chunk.source) L\($0.chunk.startLine)-\($0.chunk.endLine)" }
    }

    private func attachWebResults(_ results: [WebSearchResult], to message: StoredMessage) {
        guard !results.isEmpty else { return }
        let contextLength = loadedInfo?.maxContext ?? 2048
        let maxCharacters = min(4000, max(500, contextLength))
        var block = L("以下は Web 検索の結果です。質問に関係する場合は、これをもとに答え、使った情報の番号を [1] のように示してください。",
                      "Here are web search results. Use them if relevant and cite them like [1].") + "\n\n"
        block += WebSearchClient.format(results)
        if block.count > maxCharacters { block = String(block.prefix(maxCharacters)) + "…" }
        message.attachedContext = [message.attachedContext, block].compactMap { $0 }.joined(separator: "\n\n")
        message.sources = message.sources + results.map(\.url)
    }

    /// 生成を止める。
    func stopGenerating() {
        generationTask?.cancel()
        generationTask = nil
    }

    private func generateReply(in conversation: Conversation, context: ModelContext, settings: AppSettings,
                               assistant: Assistant?, connectors: ConnectorStore?) {
        guard let info = loadedInfo else {
            notice = L("先にモデルを選んで読み込んでください。", "Please choose and load a model first.")
            return
        }
        let history = conversation.orderedMessages
            .filter { !$0.isDeleted && $0.role != .system }
            .filter { !($0.role == .assistant && $0.content.isEmpty && $0.toolCalls.isEmpty) }
            .map(\.chatMessage)
        guard let last = history.last, last.role == .user || last.role == .tool else { return }

        // アシスタントがあれば、そのシステムプロンプトと生成設定を使う
        let rawPrompt = assistant?.systemPrompt ?? settings.systemPrompt
        let systemPrompt = rawPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        var parameters = settings.generation
        var thinking = settings.thinkingMode
        if let assistant {
            if assistant.useCustomGeneration { parameters = assistant.generation }
            if assistant.thinkingMode != .modelDefault { thinking = assistant.thinkingMode }
        }
        let tools = settings.toolsEnabled ? (connectors?.availableTools ?? []) : []
        let configuration = AgentLoop.Configuration(systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt,
                                                    parameters: parameters, enableThinking: thinking.enableThinking,
                                                    tools: tools, maxRounds: 4)
        conversation.modelID = info.modelID
        progress = GenerationProgress(conversationID: conversation.id, messageID: UUID(), phase: .preparing)
        notice = nil
        UIApplication.shared.isIdleTimerDisabled = true

        let loop = AgentLoop(engine: engine)
        generationTask = Task {
            var reply: StoredMessage?
            var lastDisplay = Date.distantPast
            do {
                try await loop.run(history: history, configuration: configuration, execute: { call in
                    guard let connectors else { return ToolOutcome(text: "no connectors", isError: true) }
                    return await connectors.execute(call)
                }, onEvent: { event in
                    switch event {
                    case .assistantStarted:
                        let message = self.appendMessage(role: .assistant, to: conversation, context: context, modelName: info.displayName)
                        reply = message
                        self.progress?.messageID = message.id
                        self.progress?.phase = .preparing
                    case .promptReady(let promptTokens, let dropped):
                        self.progress?.promptTokens = promptTokens
                        self.progress?.phase = .prefilling
                        if dropped > 0 {
                            self.notice = L("コンテキストに収まらないため、古いメッセージ \(dropped) 件を省いて送りました。",
                                            "\(dropped) older messages were left out to fit the context window.")
                        }
                    case .prefill(let done, let total):
                        self.progress?.prefillDone = done
                        self.progress?.prefillTotal = total
                    case .text(let visible):
                        // 画面の更新は 1 秒に 12 回までにまとめる（Markdown の再描画が重いので、トークンごとには描かない）
                        let now = Date()
                        if now.timeIntervalSince(lastDisplay) >= 0.08 {
                            lastDisplay = now
                            if self.progress?.phase != .generating { self.progress?.phase = .generating }
                            self.progress?.generatedCharacters = visible.count
                            reply?.content = visible
                            self.saveIfNeeded(context)
                        }
                    case .assistantFinished(let content, let calls, let stats):
                        reply?.content = content
                        reply?.toolCalls = calls
                        reply?.stats = stats
                        self.lastStats = stats
                        if stats.stopReason == .lowMemory {
                            self.notice = L("メモリが足りなくなりそうだったので、アプリが落ちないよう途中で止めました。続けるときは「続けて」と送ってください。",
                                            "Stopped early to keep the app from running out of memory. Send \"continue\" to go on.")
                        }
                        if calls.isEmpty, content.isEmpty, let reply {
                            // 何も書かなかった返答は残さない
                            conversation.messages.removeAll { $0.id == reply.id }
                            context.delete(reply)
                        }
                        try? context.save()
                    case .status(let text):
                        self.notice = text
                    case .toolStarted(let call):
                        self.progress?.phase = .usingTool(call.name)
                    case .toolFinished(let call, let outcome):
                        let message = self.appendMessage(role: .tool, to: conversation, context: context, modelName: nil)
                        message.content = outcome.text
                        message.toolName = call.name
                        message.toolCallID = call.id
                        if outcome.isError { message.errorText = outcome.text }
                        try? context.save()
                    }
                })
            } catch is CancellationError {
                // 止めたときは、そこまでの文章を残す
            } catch {
                if let reply {
                    reply.errorText = error.localizedDescription
                } else {
                    self.notice = error.localizedDescription
                }
            }
            conversation.updatedAt = Date()
            try? context.save()
            progress = nil
            generationTask = nil
            UIApplication.shared.isIdleTimerDisabled = false
            refreshMemory()
        }
    }

    private func appendMessage(role: ChatMessage.Role, to conversation: Conversation, context: ModelContext,
                               modelName: String?) -> StoredMessage {
        let order = (conversation.orderedMessages.last?.order ?? -1) + 1
        let message = StoredMessage(role: role, content: "", order: order, modelName: modelName)
        context.insert(message)
        conversation.messages.append(message)
        return message
    }

    /// 生成中は SwiftData への保存を 2 秒ごとにまとめる。
    private func saveIfNeeded(_ context: ModelContext) {
        let now = Date()
        if now.timeIntervalSince(lastSaveDate) > 2 {
            lastSaveDate = now
            try? context.save()
        }
    }
}
