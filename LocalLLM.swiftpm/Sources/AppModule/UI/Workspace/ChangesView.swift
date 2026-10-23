import SwiftUI

/// 変更履歴: エージェントと自分の変更の一覧、差分、元に戻す。
struct ChangesView: View {
    @Bindable var session: WorkspaceSession
    @State private var selected: FileChange.ID?

    private var changes: [FileChange] { session.changes.reversed() }

    var body: some View {
        HStack(spacing: 0) {
            List(selection: $selected) {
                if changes.isEmpty {
                    Text(L("まだ変更はありません", "No changes yet"))
                        .foregroundStyle(.secondary)
                }
                ForEach(changes) { change in
                    ChangeRow(change: change)
                        .tag(change.id)
                        .contextMenu {
                            Button { session.undo(change) } label: {
                                Label(L("この変更を元に戻す", "Undo this change"), systemImage: "arrow.uturn.backward")
                            }
                            .disabled(!change.canUndo)
                            if let batch = change.batchID {
                                Button { session.undoBatch(batch) } label: {
                                    Label(L("この依頼の変更をすべて戻す", "Undo the whole request"), systemImage: "arrow.uturn.backward.circle")
                                }
                            }
                        }
                }
            }
            .listStyle(.plain)
            .frame(width: 300)
            Divider()
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder
    private var detail: some View {
        if let id = selected, let change = session.changes.first(where: { $0.id == id }) {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text(change.summary).font(.headline)
                    Spacer()
                    Button { session.undo(change); selected = nil } label: {
                        Label(L("元に戻す", "Undo"), systemImage: "arrow.uturn.backward")
                    }
                    .buttonStyle(.bordered)
                    .disabled(!change.canUndo)
                }
                .padding(12)
                Divider()
                ScrollView {
                    DiffView(before: change.before ?? "", after: change.after ?? "", maxLines: 4000)
                        .padding(8)
                }
            }
        } else {
            Text(L("左の変更を選ぶと、差分が見られます", "Select a change to see the diff"))
                .foregroundStyle(.secondary)
        }
    }
}

private struct ChangeRow: View {
    let change: FileChange

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(change.summary)
                    .font(.callout)
                    .lineLimit(2)
                HStack(spacing: 6) {
                    Image(systemName: change.actor == "agent" ? "sparkles" : "person.fill")
                    Text(change.date.formatted(date: .omitted, time: .shortened))
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var icon: String {
        switch change.kind {
        case .created: return "plus.circle.fill"
        case .modified: return "pencil.circle.fill"
        case .deleted: return "minus.circle.fill"
        case .moved: return "arrow.right.circle.fill"
        }
    }

    private var color: Color {
        switch change.kind {
        case .created: return .green
        case .modified: return .blue
        case .deleted: return .red
        case .moved: return .orange
        }
    }
}

/// 行ごとの差分の表示（足した行は緑、消した行は赤）。
struct DiffView: View {
    let before: String
    let after: String
    var maxLines = 400

    var body: some View {
        let hunks = TextDiff.hunks(old: before, new: after, context: 3)
        LazyVStack(alignment: .leading, spacing: 0) {
            if hunks.isEmpty {
                Text(L("中身の違いはありません", "No content changes"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(8)
            }
            ForEach(Array(hunks.enumerated()), id: \.offset) { _, hunk in
                Text(hunk.header)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
                    .padding(.horizontal, 6)
                ForEach(Array(hunk.lines.prefix(maxLines).enumerated()), id: \.offset) { _, line in
                    DiffLineRow(line: line)
                }
            }
        }
        .textSelection(.enabled)
    }
}

private struct DiffLineRow: View {
    let line: DiffLine

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            Text(line.oldNumber.map(String.init) ?? "")
                .frame(width: 34, alignment: .trailing)
                .foregroundStyle(.tertiary)
            Text(line.newNumber.map(String.init) ?? "")
                .frame(width: 34, alignment: .trailing)
                .foregroundStyle(.tertiary)
            Text(marker + line.text)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.caption.monospaced())
        .padding(.vertical, 1)
        .padding(.horizontal, 6)
        .background(background)
    }

    private var marker: String {
        switch line.kind {
        case .context: return "  "
        case .added: return "+ "
        case .removed: return "- "
        }
    }

    private var background: Color {
        switch line.kind {
        case .context: return .clear
        case .added: return .green.opacity(0.16)
        case .removed: return .red.opacity(0.16)
        }
    }
}
