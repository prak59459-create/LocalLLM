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
    /// この会話で使うマイアシスタント（Assistant.id）。nil なら通常のチャット
    var assistantID: UUID?
    @Relationship(deleteRule: .cascade, inverse: \StoredMessage.conversation)
    var messages: [StoredMessage] = []

    init(title: String = "", modelID: String? = nil, assistantID: UUID? = nil) {
        self.id = UUID()
        self.title = title
        self.createdAt = Date()
        self.updatedAt = Date()
        self.modelID = modelID
        self.assistantID = assistantID
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
    /// 参考資料の検索結果（user メッセージに付けてモデルへ渡す。画面には出さない）
    var attachedContext: String?
    /// 参考にした資料の一覧（"ファイル名 L10-40" の配列を JSON で保存）
    var sourcesData: Data?
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

    var sources: [String] {
        get { sourcesData.flatMap { try? JSONDecoder().decode([String].self, from: $0) } ?? [] }
        set { sourcesData = newValue.isEmpty ? nil : (try? JSONEncoder().encode(newValue)) }
    }

    /// モデルに渡す形。参考資料があれば本文の前に付ける（毎回同じ文字列になるので KV キャッシュも再利用できる）。
    var chatMessage: ChatMessage {
        if let attachedContext, !attachedContext.isEmpty, role == .user {
            return ChatMessage(role: role, content: attachedContext + "\n\n" + L("質問: ", "Question: ") + content)
        }
        return ChatMessage(role: role, content: content)
    }
}
