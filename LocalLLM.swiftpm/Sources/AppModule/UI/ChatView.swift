import SwiftData
import SwiftUI

/// チャット画面（メッセージ一覧と入力欄）。
struct ChatView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    @Environment(KnowledgeStore.self) private var knowledge
    @Environment(ConnectorStore.self) private var connectors
    @Environment(\.modelContext) private var context

    @Bindable var conversation: Conversation
    var onShowModels: () -> Void

    @State private var draft = ""
    /// 次の送信の前に Web 検索する
    @State private var searchWeb = false
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
                onStop: { chat.stopGenerating() },
                accessory: { ComposerOptions(searchWeb: $searchWeb) }
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
        chat.regenerate(in: conversation, context: context, settings: settings, assistant: assistant,
                        connectors: connectors)
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
                  assistant: assistant, knowledge: knowledge, connectors: connectors,
                  options: SendOptions(searchWeb: searchWeb))
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

/// 入力欄。左に送信オプション（accessory）を置ける。
struct ComposerView<Accessory: View>: View {
    @Binding var text: String
    var isGenerating: Bool
    var canSend: Bool
    var focused: FocusState<Bool>.Binding
    var onSend: () -> Void
    var onStop: () -> Void
    @ViewBuilder var accessory: () -> Accessory

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            accessory()
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

extension ComposerView where Accessory == EmptyView {
    init(text: Binding<String>, isGenerating: Bool, canSend: Bool, focused: FocusState<Bool>.Binding,
         onSend: @escaping () -> Void, onStop: @escaping () -> Void) {
        self.init(text: text, isGenerating: isGenerating, canSend: canSend, focused: focused,
                  onSend: onSend, onStop: onStop, accessory: { EmptyView() })
    }
}

/// 入力欄の左の、Web 検索とツールの切り替え。
struct ComposerOptions: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ConnectorStore.self) private var connectors
    @Binding var searchWeb: Bool

    var body: some View {
        @Bindable var settings = settings
        HStack(spacing: 4) {
            Button {
                searchWeb.toggle()
            } label: {
                Image(systemName: "globe")
                    .font(.system(size: 20))
                    .foregroundStyle(searchWeb ? Color.accentColor : Color.secondary)
                    .frame(width: 34, height: 40)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(searchWeb ? L("Web 検索してから答える: オン", "Search the web first: on")
                                          : L("Web 検索してから答える: オフ", "Search the web first: off"))
            Menu {
                Toggle(L("モデルにツールを使わせる", "Let the model use tools"), isOn: $settings.toolsEnabled)
                if settings.toolsEnabled {
                    Section(L("使えるツール", "Available tools")) {
                        ForEach(connectors.availableTools, id: \.name) { tool in
                            Text(tool.name)
                        }
                    }
                }
            } label: {
                Image(systemName: settings.toolsEnabled ? "wrench.and.screwdriver.fill" : "wrench.and.screwdriver")
                    .font(.system(size: 18))
                    .foregroundStyle(settings.toolsEnabled ? Color.accentColor : Color.secondary)
                    .frame(width: 34, height: 40)
            }
            .accessibilityLabel(L("ツール", "Tools"))
        }
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
    @Environment(AppSettings.self) private var settings

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
                case .searching:
                    Label(L("Web を検索中", "Searching the web"), systemImage: "globe")
                case .usingTool(let name):
                    Label(L("ツールを実行中: \(name)", "Running tool: \(name)"), systemImage: "wrench.and.screwdriver")
                }
            } else if let stats = chat.lastStats {
                Label(String(format: "%.1f tok/s", stats.decodeTokensPerSecond), systemImage: "speedometer")
                Text(L("入力 \(stats.promptTokens) (再利用 \(stats.reusedPromptTokens))・出力 \(stats.generatedTokens)",
                       "In \(stats.promptTokens) (reused \(stats.reusedPromptTokens)) · Out \(stats.generatedTokens)"))
            }
            Spacer(minLength: 8)
            ContextUsageLabel(used: contextUsed, total: chat.effectiveContext(settings: settings))
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

extension StatsBar {
    /// いまの会話で使っているコンテキスト（入力 + 出力）
    fileprivate var contextUsed: Int? {
        if let progress = chat.progress, progress.promptTokens > 0 { return progress.promptTokens }
        if let stats = chat.lastStats { return stats.promptTokens + stats.generatedTokens }
        return nil
    }
}

/// 「文脈 1,234 / 4,096」の表示（8 割を超えたら色を変える）。
private struct ContextUsageLabel: View {
    let used: Int?
    let total: Int?

    var body: some View {
        if let used, let total, total > 0 {
            Label("\(used.formatted())/\(total.formatted())", systemImage: "text.alignleft")
                .foregroundStyle(Double(used) / Double(total) > 0.8 ? Color.orange : Color.secondary)
                .help(L("使っているコンテキスト / 最大", "Context used / maximum"))
        }
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
