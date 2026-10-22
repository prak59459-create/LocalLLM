import Foundation
import SwiftData

/// 会話 1 つ分（SwiftData に保存）。
@Model
final class Conversation {
    var id: UUID = UUID()
    var title: String = ""
    var createdAt: Date = Date()
    var updatedAt: Date = Date()
    /// 最後に使ったモデルの ID（InstalledModel.id）
    var modelID: String?
    @Relationship(deleteRule: .cascade, inverse: \StoredMessage.conversation)
    var messages: [StoredMessage] = []

    init(title: String = "", modelID: String? = nil) {
        self.id = UUID()
        self.title = title
        self.createdAt = Date()
        self.updatedAt = Date()
        self.modelID = modelID
        self.messages = []
    }

    /// 並び順でソートしたメッセージ。
    var orderedMessages: [StoredMessage] {
        messages.sorted { lhs, rhs in
            lhs.order != rhs.order ? lhs.order < rhs.order : lhs.createdAt < rhs.createdAt
        }
    }

    /// 表示用のタイトル（未設定なら最初の発言から作る）。
    var displayTitle: String {
        if !title.isEmpty { return title }
        if let first = orderedMessages.first(where: { $0.role == .user }) {
            return Conversation.makeTitle(from: first.content)
        }
        return L("新しいチャット", "New Chat")
    }

    /// 会話全体を Markdown のテキストにする（共有・書き出し用）。思考部分は省く。
    var markdownTranscript: String {
        var lines = ["# \(displayTitle)", ""]
        for message in orderedMessages {
            let name: String
            switch message.role {
            case .user: name = L("あなた", "You")
            case .assistant: name = message.modelName ?? L("アシスタント", "Assistant")
            case .system: name = L("システム", "System")
            }
            lines.append("**\(name)**")
            lines.append("")
            lines.append(ThinkingSplit.split(message.content).answer)
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    static func makeTitle(from text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.count > 30 ? String(trimmed.prefix(30)) + "…" : trimmed
    }
}

/// メッセージ 1 つ分。
@Model
final class StoredMessage {
    var id: UUID = UUID()
    var roleRaw: String = ChatMessage.Role.user.rawValue
    var content: String = ""
    var order: Int = 0
    var createdAt: Date = Date()
    /// 生成に使ったモデル名
    var modelName: String?
    /// 生成の統計（GenerationStats を JSON で保存）
    var statsData: Data?
    /// 生成エラーの説明（失敗したとき）
    var errorText: String?
    var conversation: Conversation?

    init(role: ChatMessage.Role, content: String, order: Int, modelName: String? = nil) {
        self.id = UUID()
        self.roleRaw = role.rawValue
        self.content = content
        self.order = order
        self.createdAt = Date()
        self.modelName = modelName
    }

    var role: ChatMessage.Role {
        get { ChatMessage.Role(rawValue: roleRaw) ?? .user }
        set { roleRaw = newValue.rawValue }
    }

    var stats: GenerationStats? {
        get { statsData.flatMap { try? JSONDecoder().decode(GenerationStats.self, from: $0) } }
        set { statsData = newValue.flatMap { try? JSONEncoder().encode($0) } }
    }

    var chatMessage: ChatMessage {
        ChatMessage(role: role, content: content)
    }
}
