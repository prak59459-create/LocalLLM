import SwiftData
import SwiftUI

/// サイドバー: 会話の一覧。
struct ConversationListView: View {
    @Environment(\.modelContext) private var context
    @Environment(ChatController.self) private var chat
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    @Query(sort: \Assistant.createdAt) private var assistants: [Assistant]

    @Binding var selection: UUID?
    var onNewChat: (Assistant?) -> Void
    var onShowModels: () -> Void
    var onShowAssistants: () -> Void
    var onShowStudio: () -> Void
    var onShowSettings: () -> Void

    @State private var renaming: Conversation?
    @State private var renameText = ""
    @State private var deleting: Conversation?

    // 型チェックを速くするため、行・ツールバー・確認ダイアログを分けている
    var body: some View {
        List(selection: $selection) {
            Section {
                ModelStatusBadge()
                    .listRowBackground(Color.clear)
            }
            Section(L("チャット", "Chats")) {
                if conversations.isEmpty {
                    Text(L("まだ会話がありません", "No chats yet"))
                        .foregroundStyle(.secondary)
                }
                ForEach(conversations) { conversation in
                    row(for: conversation)
                }
            }
        }
        .navigationTitle("LocalLLM")
        .toolbar { toolbarContent }
        .alert(L("名前を変更", "Rename"), isPresented: isRenaming) {
            TextField(L("タイトル", "Title"), text: $renameText)
            Button(L("保存", "Save"), action: saveRename)
            Button(L("キャンセル", "Cancel"), role: .cancel) { renaming = nil }
        }
        .confirmationDialog(L("この会話を削除しますか？", "Delete this chat?"), isPresented: isDeleting,
                            titleVisibility: .visible) {
            Button(L("削除", "Delete"), role: .destructive, action: confirmDelete)
        }
    }

    private func row(for conversation: Conversation) -> some View {
        ConversationRow(conversation: conversation,
                        assistant: assistants.first { $0.id == conversation.assistantID })
            .tag(conversation.id)
            .contextMenu {
                Button {
                    renameText = conversation.displayTitle
                    renaming = conversation
                } label: {
                    Label(L("名前を変更", "Rename"), systemImage: "pencil")
                }
                Button(role: .destructive) {
                    deleting = conversation
                } label: {
                    Label(L("削除", "Delete"), systemImage: "trash")
                }
            }
            .swipeActions {
                Button(role: .destructive) {
                    deleting = conversation
                } label: {
                    Label(L("削除", "Delete"), systemImage: "trash")
                }
            }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            NewChatMenu(assistants: assistants, onNewChat: onNewChat, onShowAssistants: onShowAssistants)
        }
        ToolbarItemGroup(placement: .bottomBar) {
            Button(action: onShowModels) {
                Label(L("モデル", "Models"), systemImage: "cpu")
            }
            Spacer()
            Button(action: onShowAssistants) {
                Label(L("アシスタント", "Assistants"), systemImage: "person.crop.rectangle.stack")
            }
            Spacer()
            Button(action: onShowStudio) {
                Label(L("ミニLLM工房", "Mini LLM Studio"), systemImage: "hammer")
            }
            Spacer()
            Button(action: onShowSettings) {
                Label(L("設定", "Settings"), systemImage: "gearshape")
            }
        }
    }

    private var isRenaming: Binding<Bool> {
        Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })
    }

    private var isDeleting: Binding<Bool> {
        Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })
    }

    private func saveRename() {
        renaming?.title = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        try? context.save()
        renaming = nil
    }

    private func confirmDelete() {
        if let conversation = deleting { delete(conversation) }
        deleting = nil
    }

    private func delete(_ conversation: Conversation) {
        if chat.progress?.conversationID == conversation.id {
            chat.stopGenerating()
        }
        if selection == conversation.id { selection = nil }
        context.delete(conversation)
        try? context.save()
    }
}

/// 新しいチャットのボタン（長押し・タップでアシスタントも選べる）。
private struct NewChatMenu: View {
    let assistants: [Assistant]
    var onNewChat: (Assistant?) -> Void
    var onShowAssistants: () -> Void

    var body: some View {
        Menu {
            Button {
                onNewChat(nil)
            } label: {
                Label(L("通常のチャット", "Plain chat"), systemImage: "bubble.left")
            }
            if !assistants.isEmpty {
                Section(L("マイアシスタント", "My Assistants")) {
                    ForEach(assistants) { assistant in
                        Button {
                            onNewChat(assistant)
                        } label: {
                            Label(assistant.name, systemImage: assistant.symbolName)
                        }
                    }
                }
            }
            Button(action: onShowAssistants) {
                Label(L("アシスタントを作る・編集…", "Create or edit assistants…"), systemImage: "person.crop.rectangle.stack")
            }
        } label: {
            Label(L("新しいチャット", "New Chat"), systemImage: "square.and.pencil")
        } primaryAction: {
            onNewChat(nil)
        }
    }
}

private struct ConversationRow: View {
    let conversation: Conversation
    let assistant: Assistant?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                if let assistant {
                    Image(systemName: assistant.symbolName)
                        .font(.caption)
                        .foregroundStyle(.tint)
                }
                Text(conversation.displayTitle)
                    .lineLimit(1)
            }
            Text(conversation.updatedAt, format: .relative(presentation: .named))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

/// 読み込み中のモデルの状態を小さく表示する。
struct ModelStatusBadge: View {
    @Environment(ChatController.self) private var chat

    var body: some View {
        HStack(spacing: 8) {
            switch chat.loadState {
            case .idle:
                Image(systemName: "circle.dashed")
                    .foregroundStyle(.secondary)
                Text(L("モデル未読み込み", "No model loaded"))
                    .foregroundStyle(.secondary)
            case .compiling(let name):
                ProgressView()
                Text(L("\(name) を準備中（初回のみ）…", "Preparing \(name) (first time only)…"))
            case .loading(let name):
                ProgressView()
                Text(L("\(name) を読み込み中…", "Loading \(name)…"))
            case .ready(let info):
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text(info.displayName)
                    .lineLimit(1)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(message)
                    .lineLimit(3)
            }
        }
        .font(.footnote)
    }
}
