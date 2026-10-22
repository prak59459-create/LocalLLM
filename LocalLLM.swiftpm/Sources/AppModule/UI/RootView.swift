import SwiftData
import SwiftUI

/// 画面全体: 左に会話一覧、右にチャット。
struct RootView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    @Environment(\.modelContext) private var context
    @Environment(\.scenePhase) private var scenePhase

    @State private var selectedConversationID: UUID?
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic
    @State private var showingModels = false
    @State private var showingSettings = false
    @State private var didAutoLoad = false

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ConversationListView(selection: $selectedConversationID,
                                 onNewChat: newConversation,
                                 onShowModels: { showingModels = true },
                                 onShowSettings: { showingSettings = true })
        } detail: {
            if let conversation = selectedConversation {
                ChatView(conversation: conversation, onShowModels: { showingModels = true })
                    .id(conversation.id)
            } else {
                WelcomeView(onNewChat: newConversation, onShowModels: { showingModels = true })
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
        .task {
            autoLoadIfNeeded()
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

    private func newConversation() {
        let conversation = Conversation(modelID: settings.activeModelID)
        context.insert(conversation)
        try? context.save()
        selectedConversationID = conversation.id
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
        }
        .padding(32)
        .frame(maxWidth: 520)
    }
}
