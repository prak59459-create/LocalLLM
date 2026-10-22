import SwiftData
import SwiftUI

/// サイドバー: 会話の一覧。
struct ConversationListView: View {
    @Environment(\.modelContext) private var context
    @Environment(ChatController.self) private var chat
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]

    @Binding var selection: UUID?
    var onNewChat: () -> Void
    var onShowModels: () -> Void
    var onShowSettings: () -> Void

    @State private var renaming: Conversation?
    @State private var renameText = ""
    @State private var deleting: Conversation?

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
                    ConversationRow(conversation: conversation)
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
            }
        }
        .navigationTitle("LocalLLM")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(action: onNewChat) {
                    Label(L("新しいチャット", "New Chat"), systemImage: "square.and.pencil")
                }
            }
            ToolbarItemGroup(placement: .bottomBar) {
                Button(action: onShowModels) {
                    Label(L("モデル", "Models"), systemImage: "cpu")
                }
                Spacer()
                Button(action: onShowSettings) {
                    Label(L("設定", "Settings"), systemImage: "gearshape")
                }
            }
        }
        .alert(L("名前を変更", "Rename"), isPresented: Binding(
            get: { renaming != nil },
            set: { if !$0 { renaming = nil } }
        )) {
            TextField(L("タイトル", "Title"), text: $renameText)
            Button(L("保存", "Save")) {
                renaming?.title = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                try? context.save()
                renaming = nil
            }
            Button(L("キャンセル", "Cancel"), role: .cancel) { renaming = nil }
        }
        .confirmationDialog(L("この会話を削除しますか？", "Delete this chat?"), isPresented: Binding(
            get: { deleting != nil },
            set: { if !$0 { deleting = nil } }
        ), titleVisibility: .visible) {
            Button(L("削除", "Delete"), role: .destructive) {
                if let conversation = deleting { delete(conversation) }
                deleting = nil
            }
        }
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

private struct ConversationRow: View {
    let conversation: Conversation

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(conversation.displayTitle)
                .lineLimit(1)
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
