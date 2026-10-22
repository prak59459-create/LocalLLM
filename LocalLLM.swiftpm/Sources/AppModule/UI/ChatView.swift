import SwiftData
import SwiftUI

/// チャット画面（メッセージ一覧と入力欄）。
struct ChatView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    @Environment(KnowledgeStore.self) private var knowledge
    @Environment(\.modelContext) private var context

    @Bindable var conversation: Conversation
    var onShowModels: () -> Void

    @State private var draft = ""
    @FocusState private var composerFocused: Bool

    private var isGeneratingHere: Bool {
        chat.progress?.conversationID == conversation.id
    }

    var body: some View {
        VStack(spacing: 0) {
            if let notice = chat.notice {
                NoticeBar(text: notice) { chat.notice = nil }
            }
            if let assistant {
                AssistantBanner(assistant: assistant)
            }
            messageList
            if settings.showStats {
                StatsBar()
            }
            Divider()
            ComposerView(
                text: $draft,
                isGenerating: chat.isGenerating,
                canSend: chat.loadedInfo != nil && !chat.isGenerating,
                focused: $composerFocused,
                onSend: send,
                onStop: { chat.stopGenerating() }
            )
        }
        .navigationTitle(conversation.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                ShareLink(item: conversation.markdownTranscript) {
                    Label(L("会話を書き出す", "Export Chat"), systemImage: "square.and.arrow.up")
                }
                .disabled(conversation.messages.isEmpty)
            }
            ToolbarItem(placement: .topBarTrailing) {
                ModelPickerMenu(onShowModels: onShowModels)
            }
        }
    }

    // 型チェックを速くするため、中身は MessageListContent に分けている
    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                MessageListContent(messages: conversation.orderedMessages,
                                   streamingID: chat.progress?.messageID,
                                   bottomID: bottomID,
                                   onShowModels: onShowModels,
                                   onRegenerate: regenerate,
                                   onDelete: delete)
            }
            .scrollDismissesKeyboard(.interactively)
            .onAppear { scrollToBottom(proxy, animated: false) }
            .onChange(of: conversation.messages.count) { _, _ in scrollToBottom(proxy, animated: true) }
            .onChange(of: chat.progress?.generatedCharacters) { _, _ in scrollToBottom(proxy, animated: false) }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool) {
        if animated {
            withAnimation { proxy.scrollTo(bottomID, anchor: .bottom) }
        } else {
            proxy.scrollTo(bottomID, anchor: .bottom)
        }
    }

    private func regenerate() {
        chat.regenerate(in: conversation, context: context, settings: settings, assistant: assistant)
    }

    private let bottomID = "bottom"

    /// この会話のアシスタント（あれば）
    private var assistant: Assistant? {
        guard let id = conversation.assistantID else { return nil }
        let descriptor = FetchDescriptor<Assistant>(predicate: #Predicate { $0.id == id })
        return try? context.fetch(descriptor).first
    }

    private func send() {
        let text = draft
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        draft = ""
        chat.send(text, in: conversation, context: context, settings: settings,
                  assistant: assistant, knowledge: knowledge)
    }

    private func delete(_ message: StoredMessage) {
        if chat.progress?.messageID == message.id { chat.stopGenerating() }
        conversation.messages.removeAll { $0.id == message.id }
        context.delete(message)
        try? context.save()
    }
}

/// メッセージの並び。
private struct MessageListContent: View {
    let messages: [StoredMessage]
    let streamingID: UUID?
    let bottomID: String
    var onShowModels: () -> Void
    var onRegenerate: () -> Void
    var onDelete: (StoredMessage) -> Void

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 16) {
            if messages.isEmpty {
                EmptyChatHint(onShowModels: onShowModels)
                    .padding(.top, 60)
            }
            ForEach(messages) { message in
                row(for: message)
            }
            Color.clear.frame(height: 1).id(bottomID)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity)
    }

    private func row(for message: StoredMessage) -> some View {
        MessageRow(message: message,
                   isStreaming: streamingID == message.id,
                   isLast: message.id == messages.last?.id,
                   onRegenerate: onRegenerate,
                   onDelete: { onDelete(message) })
            .id(message.id)
    }
}

/// アシスタントを使っている会話の上部に出す帯。使うモデルが読み込まれていなければ読み込みボタンを出す。
struct AssistantBanner: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    @Environment(KnowledgeStore.self) private var knowledge
    let assistant: Assistant

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: assistant.symbolName)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(assistant.name).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let preferred = preferredModel, chat.loadedInfo?.modelID != preferred.id, !chat.isBusyLoading {
                Button {
                    chat.loadModel(id: preferred.id, library: library, settings: settings)
                } label: {
                    Label(L("\(preferred.displayName) を読み込む", "Load \(preferred.displayName)"), systemImage: "arrow.down.to.line")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.accentColor.opacity(0.08))
    }

    private var preferredModel: InstalledModel? {
        guard let id = assistant.preferredModelID, let model = library.model(withID: id), model.isComplete else { return nil }
        return model
    }

    private var detail: String {
        let count = knowledge.index(for: assistant.id).sources.count
        if assistant.useKnowledge && count > 0 {
            return L("参考資料 \(count) ファイル", "\(count) reference files")
        }
        return L("参考資料なし", "No reference files")
    }
}

/// 入力欄。
struct ComposerView: View {
    @Binding var text: String
    var isGenerating: Bool
    var canSend: Bool
    var focused: FocusState<Bool>.Binding
    var onSend: () -> Void
    var onStop: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField(L("メッセージを入力", "Message"), text: $text, axis: .vertical)
                .lineLimit(1...8)
                .textFieldStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20))
                .focused(focused)
                .onSubmit {
                    if canSend { onSend() }
                }

            if isGenerating {
                Button(action: onStop) {
                    Image(systemName: "stop.circle.fill")
                        .font(.system(size: 34))
                }
                .accessibilityLabel(L("停止", "Stop"))
            } else {
                Button(action: onSend) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 34))
                }
                .disabled(!canSend || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
                .accessibilityLabel(L("送信", "Send"))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

/// 右上のモデル切り替えメニュー。
struct ModelPickerMenu: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    var onShowModels: () -> Void

    var body: some View {
        Menu {
            if library.completedModels.isEmpty {
                Text(L("インストール済みのモデルがありません", "No installed models"))
            }
            ForEach(library.completedModels) { model in
                Button {
                    chat.loadModel(id: model.id, library: library, settings: settings)
                } label: {
                    if chat.loadedInfo?.modelID == model.id {
                        Label(model.displayName, systemImage: "checkmark")
                    } else {
                        Text(model.displayName)
                    }
                }
            }
            Divider()
            if chat.loadedInfo != nil {
                Button(role: .destructive) {
                    chat.unloadModel()
                } label: {
                    Label(L("モデルを解放", "Unload Model"), systemImage: "eject")
                }
            }
            Button {
                onShowModels()
            } label: {
                Label(L("モデルを管理…", "Manage Models…"), systemImage: "cpu")
            }
        } label: {
            HStack(spacing: 6) {
                if chat.isBusyLoading {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "cpu")
                }
                Text(menuTitle)
                    .lineLimit(1)
                    .frame(maxWidth: 220)
            }
        }
    }

    private var menuTitle: String {
        switch chat.loadState {
        case .ready(let info): return info.displayName
        case .loading(let name), .compiling(let name): return name
        case .failed: return L("読み込み失敗", "Load failed")
        case .idle: return L("モデルを選択", "Choose Model")
        }
    }
}

/// 速度とメモリの表示。
struct StatsBar: View {
    @Environment(ChatController.self) private var chat

    var body: some View {
        HStack(spacing: 14) {
            if let progress = chat.progress {
                switch progress.phase {
                case .preparing:
                    Label(L("準備中", "Preparing"), systemImage: "hourglass")
                case .prefilling:
                    Label(L("読み込み \(progress.prefillDone)/\(progress.prefillTotal)",
                            "Reading \(progress.prefillDone)/\(progress.prefillTotal)"),
                          systemImage: "text.viewfinder")
                case .generating:
                    Label(L("生成中", "Generating"), systemImage: "ellipsis.bubble")
                }
            } else if let stats = chat.lastStats {
                Label(String(format: "%.1f tok/s", stats.decodeTokensPerSecond), systemImage: "speedometer")
                Text(L("入力 \(stats.promptTokens) (再利用 \(stats.reusedPromptTokens))・出力 \(stats.generatedTokens)",
                       "In \(stats.promptTokens) (reused \(stats.reusedPromptTokens)) · Out \(stats.generatedTokens)"))
            }
            Spacer(minLength: 8)
            if let footprint = chat.memoryFootprint {
                Label(L("使用 \(ByteFormat.string(footprint))・残り \(ByteFormat.string(chat.memoryAvailable))",
                        "Used \(ByteFormat.string(footprint)) · Free \(ByteFormat.string(chat.memoryAvailable))"),
                      systemImage: "memorychip")
            }
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(.bar)
    }
}

struct NoticeBar: View {
    let text: String
    var onClose: () -> Void

    var body: some View {
        HStack(alignment: .top) {
            Image(systemName: "info.circle")
            Text(text)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(action: onClose) {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
        }
        .font(.footnote)
        .padding(10)
        .background(Color.yellow.opacity(0.18))
    }
}

private struct EmptyChatHint: View {
    @Environment(ChatController.self) private var chat
    var onShowModels: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "text.bubble")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            if chat.loadedInfo == nil {
                Text(L("右上のメニューからモデルを選んでください。", "Choose a model from the menu at the top right."))
                    .foregroundStyle(.secondary)
                Button(L("モデルを入手・管理", "Get or Manage Models"), action: onShowModels)
                    .buttonStyle(.bordered)
            } else {
                Text(L("何でも聞いてみてください。すべて iPad の中で処理されます。",
                       "Ask anything. Everything runs on your iPad."))
                    .foregroundStyle(.secondary)
            }
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
    }
}
