import SwiftData
import SwiftUI

/// 画面全体: 左に会話一覧、右にチャット。
struct RootView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    @Environment(AppUpdateController.self) private var updater
    @Environment(\.modelContext) private var context
    @Environment(\.scenePhase) private var scenePhase

    @State private var selectedConversationID: UUID?
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic
    @State private var showingModels = false
    @State private var showingSettings = false
    @State private var showingAssistants = false
    @State private var showingStudio = false
    @State private var showingWorkspace = false
    @State private var didAutoLoad = false

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ConversationListView(selection: $selectedConversationID,
                                 onNewChat: { newConversation(assistant: $0) },
                                 onShowModels: { showingModels = true },
                                 onShowAssistants: { showingAssistants = true },
                                 onShowStudio: { showingStudio = true },
                                 onShowWorkspace: { showingWorkspace = true },
                                 onShowSettings: { showingSettings = true })
        } detail: {
            if let conversation = selectedConversation {
                ChatView(conversation: conversation, onShowModels: { showingModels = true })
                    .id(conversation.id)
            } else {
                WelcomeView(onNewChat: { newConversation(assistant: nil) },
                            onShowModels: { showingModels = true },
                            onShowAssistants: { showingAssistants = true },
                            onShowStudio: { showingStudio = true },
                            onShowWorkspace: { showingWorkspace = true })
            }
        }
        // 言語を切り替えたら画面全体を作り直して文字列を更新する
        .id(settings.language)
        .sheet(isPresented: $showingModels) {
            ModelsView()
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
        .sheet(isPresented: $showingAssistants) {
            AssistantsView(onStartChat: { newConversation(assistant: $0) })
        }
        .fullScreenCover(isPresented: $showingStudio) {
            MiniLLMStudioView()
        }
        .fullScreenCover(isPresented: $showingWorkspace) {
            WorkspaceHomeView()
        }
        .task {
            removeEmptyConversations(except: nil)
            autoLoadIfNeeded()
            // GitHub に新しい版があるかだけを軽く確かめる（設定の「アプリの更新」に印が付く）
            await updater.checkInBackgroundIfNeeded()
        }
        // 何も話さなかった会話は、ほかの会話に移ったときに消す（履歴に空の会話を残さない）
        .onChange(of: selectedConversationID) { _, newID in
            removeEmptyConversations(except: newID)
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active:
                library.reload()
            case .background:
                // iOS はバックグラウンドで GPU / Neural Engine を使わせないので、生成は止めておく
                if chat.isGenerating { chat.stopGenerating() }
            default:
                break
            }
        }
    }

    private var selectedConversation: Conversation? {
        guard let id = selectedConversationID else { return nil }
        let descriptor = FetchDescriptor<Conversation>(predicate: #Predicate { $0.id == id })
        return try? context.fetch(descriptor).first
    }

    private func newConversation(assistant: Assistant?) {
        // いま開いている会話がまだ空で、同じアシスタントなら、それをそのまま使う
        if let current = selectedConversation, current.messages.isEmpty, current.assistantID == assistant?.id {
            return
        }
        let conversation = Conversation(modelID: settings.activeModelID, assistantID: assistant?.id)
        context.insert(conversation)
        try? context.save()
        selectedConversationID = conversation.id
        // アシスタントに使うモデルが決まっていて、まだ読み込まれていなければ読み込む
        if let id = assistant?.preferredModelID, chat.loadedInfo?.modelID != id,
           library.model(withID: id)?.isComplete == true, !chat.isGenerating {
            chat.loadModel(id: id, library: library, settings: settings)
        }
    }

    /// メッセージが 1 つもない会話を消す（いま開いているものは残す）。
    private func removeEmptyConversations(except keepID: UUID?) {
        guard let all = try? context.fetch(FetchDescriptor<Conversation>()) else { return }
        var changed = false
        for conversation in all where conversation.messages.isEmpty && conversation.id != keepID {
            if chat.progress?.conversationID == conversation.id { continue }
            context.delete(conversation)
            changed = true
        }
        if changed { try? context.save() }
    }

    /// 起動時に前回使ったモデルを読み込む。
    private func autoLoadIfNeeded() {
        guard !didAutoLoad else { return }
        didAutoLoad = true
        guard settings.autoLoadLastModel,
              chat.loadState == .idle,
              let id = settings.activeModelID,
              library.model(withID: id)?.isComplete == true else { return }
        chat.loadModel(id: id, library: library, settings: settings)
    }
}

/// 会話を選んでいないときの画面。
struct WelcomeView: View {
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    var onNewChat: () -> Void
    var onShowModels: () -> Void
    var onShowAssistants: () -> Void
    var onShowStudio: () -> Void
    var onShowWorkspace: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
            Text(L("iPad の中だけで動く AI チャット", "AI chat that runs entirely on your iPad"))
                .font(.title2.bold())
                .multilineTextAlignment(.center)
            Text(L("Hugging Face から Core ML のモデルをダウンロードして、オフラインで会話できます。",
                   "Download Core ML models from Hugging Face and chat offline."))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            ModelStatusBadge()

            HStack(spacing: 12) {
                Button {
                    onShowModels()
                } label: {
                    Label(library.completedModels.isEmpty ? L("モデルを入手", "Get a Model") : L("モデル", "Models"),
                          systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)

                Button {
                    onNewChat()
                } label: {
                    Label(L("新しいチャット", "New Chat"), systemImage: "square.and.pencil")
                }
                .buttonStyle(.borderedProminent)
            }

            HStack(spacing: 12) {
                Button(action: onShowAssistants) {
                    Label(L("自分専用の AI を作る", "Build your own assistant"), systemImage: "person.crop.rectangle.stack")
                }
                Button(action: onShowStudio) {
                    Label(L("LLM をゼロから学習", "Train an LLM from scratch"), systemImage: "hammer")
                }
            }
            Button(action: onShowWorkspace) {
                Label(L("AI と一緒にコードを書く（CDWS）", "Code together with the AI (CDWS)"),
                      systemImage: "chevron.left.forwardslash.chevron.right")
            }
            .buttonStyle(.bordered)
            .font(.callout)
            .buttonStyle(.bordered)
            .font(.callout)
        }
        .padding(32)
        .frame(maxWidth: 520)
    }
}
