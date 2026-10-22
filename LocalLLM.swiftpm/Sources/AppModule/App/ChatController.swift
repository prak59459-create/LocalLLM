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
        case prefilling
        case generating
    }
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
                self?.notice = L("メモリが不足しています。アプリが終了されるおそれがあるので、小さいモデルに切り替えるか、他のアプリを閉じてください。",
                                 "Memory is running low. The app may be terminated; switch to a smaller model or close other apps.")
                self?.refreshMemory()
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
                if model.kind == .package && model.compiledModelPath == nil {
                    loadState = .compiling(model.displayName)
                } else {
                    loadState = .loading(model.displayName)
                }
                let files = try await library.prepareFiles(for: id)
                loadState = .loading(model.displayName)
                let info = try await engine.load(files: files,
                                                 computeUnits: settings.computeUnits.computeUnits,
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
              assistant: Assistant? = nil, knowledge: KnowledgeStore? = nil) {
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
        generateReply(in: conversation, context: context, settings: settings, assistant: assistant)
    }

    /// 最後のアシスタントの返答を作り直す。
    func regenerate(in conversation: Conversation, context: ModelContext, settings: AppSettings,
                    assistant: Assistant? = nil) {
        guard !isGenerating else { return }
        var excluded: UUID?
        if let last = conversation.orderedMessages.last, last.role == .assistant {
            excluded = last.id
            conversation.messages.removeAll { $0.id == last.id }
            context.delete(last)
            try? context.save()
        }
        generateReply(in: conversation, context: context, settings: settings, assistant: assistant, excluding: excluded)
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

    /// 生成を止める。
    func stopGenerating() {
        generationTask?.cancel()
        generationTask = nil
    }

    private func generateReply(in conversation: Conversation, context: ModelContext, settings: AppSettings,
                               assistant: Assistant?, excluding excludedID: UUID? = nil) {
        guard let info = loadedInfo else {
            notice = L("先にモデルを選んで読み込んでください。", "Please choose and load a model first.")
            return
        }
        let messages = conversation.orderedMessages.filter { $0.id != excludedID && !$0.isDeleted }
        let history = messages
            .filter { $0.role != .system && !($0.role == .assistant && $0.content.isEmpty) }
            .map(\.chatMessage)
        guard history.last?.role == .user else { return }

        let nextOrder = (messages.last?.order ?? -1) + 1
        let reply = StoredMessage(role: .assistant, content: "", order: nextOrder, modelName: info.displayName)
        context.insert(reply)
        conversation.messages.append(reply)
        conversation.modelID = info.modelID
        conversation.updatedAt = Date()
        try? context.save()

        // アシスタントがあれば、そのシステムプロンプトと生成設定を使う
        let rawPrompt = assistant?.systemPrompt ?? settings.systemPrompt
        let systemPrompt = rawPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        var parameters = settings.generation
        var thinking = settings.thinkingMode
        if let assistant {
            if assistant.useCustomGeneration { parameters = assistant.generation }
            if assistant.thinkingMode != .modelDefault { thinking = assistant.thinkingMode }
        }
        let request = GenerationRequest(
            systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt,
            history: history,
            parameters: parameters,
            enableThinking: thinking.enableThinking
        )
        progress = GenerationProgress(conversationID: conversation.id, messageID: reply.id, phase: .preparing)
        notice = nil
        UIApplication.shared.isIdleTimerDisabled = true

        let stream = engine.generate(request)
        generationTask = Task {
            var text = ""
            do {
                for try await event in stream {
                    switch event {
                    case .promptReady(let promptTokens, let dropped):
                        progress?.promptTokens = promptTokens
                        progress?.phase = .prefilling
                        if dropped > 0 {
                            notice = L("コンテキストに収まらないため、古いメッセージ \(dropped) 件を省いて送りました。",
                                       "\(dropped) older messages were left out to fit the context window.")
                        }
                    case .prefill(let done, let total):
                        progress?.prefillDone = done
                        progress?.prefillTotal = total
                    case .text(let piece):
                        text += piece
                        progress?.phase = .generating
                        progress?.generatedCharacters += piece.count
                        reply.content = text
                        saveIfNeeded(context)
                    case .finished(let stats):
                        reply.stats = stats
                        lastStats = stats
                    }
                }
            } catch {
                reply.errorText = error.localizedDescription
            }
            reply.content = text
            conversation.updatedAt = Date()
            try? context.save()
            progress = nil
            generationTask = nil
            UIApplication.shared.isIdleTimerDisabled = false
            refreshMemory()
        }
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
