import Foundation

/// ファイルへの変更 1 つ。元に戻せるように、前後の中身を持つ。
struct FileChange: Codable, Hashable, Sendable, Identifiable {
    enum Kind: String, Codable, Sendable {
        case created
        case modified
        case deleted
        case moved
    }

    var id: UUID = UUID()
    var date: Date = Date()
    var path: String
    var kind: Kind
    /// 変更前の中身（新しく作ったときと、テキストでないときは nil）
    var before: String?
    /// 変更後の中身（消したときは nil）
    var after: String?
    var movedFrom: String?
    /// だれが変えたか（"agent" / "user"）
    var actor: String = "agent"
    /// どの依頼の中の変更か（エージェントの 1 回の作業ごと）
    var batchID: UUID?

    init(path: String, kind: Kind, before: String?, after: String?, movedFrom: String? = nil) {
        self.path = path
        self.kind = kind
        self.before = before
        self.after = after
        self.movedFrom = movedFrom
    }

    var stats: (added: Int, removed: Int) {
        TextDiff.stats(old: before ?? "", new: after ?? "")
    }

    var summary: String {
        switch kind {
        case .created:
            return L("作成 \(path)（+\(TextDiff.lines(after ?? "").count)）", "Created \(path) (+\(TextDiff.lines(after ?? "").count))")
        case .modified:
            let stats = stats
            return L("変更 \(path)（+\(stats.added) −\(stats.removed)）", "Edited \(path) (+\(stats.added) −\(stats.removed))")
        case .deleted:
            return L("削除 \(path)", "Deleted \(path)")
        case .moved:
            return L("移動 \(movedFrom ?? "") → \(path)", "Moved \(movedFrom ?? "") → \(path)")
        }
    }

    /// 元に戻せるか
    var canUndo: Bool {
        switch kind {
        case .created: return true
        case .modified, .deleted: return before != nil
        case .moved: return movedFrom != nil
        }
    }
}

/// 変更の履歴。新しいものが後ろ。プロジェクトの .cdws/history.json に保存する。
struct ChangeHistory: Codable, Sendable {
    var changes: [FileChange] = []

    /// 保存する件数と、1 つの中身の大きさの上限
    static let maxChanges = 300
    static let maxStoredCharacters = 400_000

    mutating func record(_ change: FileChange) {
        var stored = change
        if (stored.before?.count ?? 0) > Self.maxStoredCharacters { stored.before = nil }
        if (stored.after?.count ?? 0) > Self.maxStoredCharacters { stored.after = nil }
        changes.append(stored)
        if changes.count > Self.maxChanges { changes.removeFirst(changes.count - Self.maxChanges) }
    }

    mutating func record(_ newChanges: [FileChange]) {
        for change in newChanges { record(change) }
    }

    /// 変更を取り消して、履歴からも消す。
    mutating func undo(_ id: UUID, in fileSystem: WorkspaceFileSystem) throws {
        guard let index = changes.firstIndex(where: { $0.id == id }) else { return }
        try Self.revert(changes[index], in: fileSystem)
        changes.remove(at: index)
    }

    /// ひとまとまり（エージェントの 1 回の作業）の変更を、新しい順に取り消す。
    mutating func undoBatch(_ batchID: UUID, in fileSystem: WorkspaceFileSystem) throws {
        for change in changes.reversed() where change.batchID == batchID {
            try undo(change.id, in: fileSystem)
        }
    }

    static func revert(_ change: FileChange, in fileSystem: WorkspaceFileSystem) throws {
        switch change.kind {
        case .created:
            if fileSystem.exists(change.path) { try fileSystem.delete(change.path) }
        case .modified, .deleted:
            guard let before = change.before else {
                throw WorkspaceError.invalidArgument(L("この変更は元に戻せません（中身が大きすぎるか、テキストではありません）",
                                                       "This change cannot be undone (too large or not text)"))
            }
            try fileSystem.write(change.path, before)
        case .moved:
            guard let from = change.movedFrom else { return }
            try fileSystem.move(change.path, to: from)
        }
    }
}
