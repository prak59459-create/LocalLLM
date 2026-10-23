import SwiftUI
import UIKit

/// プロジェクトのファイルの木。
struct FileExplorerView: View {
    @Bindable var session: WorkspaceSession

    @State private var prompt: NamePrompt?
    @State private var promptText = ""
    @State private var deleting: WorkspaceEntry?

    /// 名前を入れてもらう操作
    private struct NamePrompt: Identifiable {
        enum Action { case newFile, newFolder, rename }
        var action: Action
        /// 新しく作るときは親のフォルダ、名前の変更は対象のパス
        var path: String
        var id: String { "\(action)-\(path)" }

        var title: String {
            switch action {
            case .newFile: return L("新しいファイル", "New File")
            case .newFolder: return L("新しいフォルダ", "New Folder")
            case .rename: return L("名前を変更", "Rename")
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            tree
        }
        .background(Color(uiColor: .secondarySystemBackground).opacity(0.5))
        .alert(prompt?.title ?? "", isPresented: isPrompting) {
            TextField(L("名前", "Name"), text: $promptText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button(L("キャンセル", "Cancel"), role: .cancel) { prompt = nil }
            Button("OK", action: commitPrompt)
        }
        .confirmationDialog(deleteTitle, isPresented: isDeleting, titleVisibility: .visible) {
            Button(L("削除", "Delete"), role: .destructive, action: confirmDelete)
        } message: {
            Text(L("テキストのファイルは変更履歴から元に戻せます。", "Text files can be restored from the change history."))
        }
    }

    private var tree: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(session.visibleEntries) { entry in
                    row(entry)
                }
                if session.visibleEntries.isEmpty {
                    Text(L("ファイルがありません", "No files"))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding()
                }
            }
            .padding(.vertical, 4)
        }
        .refreshable { session.refreshTree() }
    }

    private func row(_ entry: WorkspaceEntry) -> some View {
        FileRow(entry: entry,
                isExpanded: session.expanded.contains(entry.path),
                isSelected: session.selectedPath == entry.path,
                isDirty: session.isDirty(entry.path))
            .onTapGesture { tap(entry) }
            .contextMenu { menu(for: entry) }
    }

    private var isPrompting: Binding<Bool> {
        Binding(get: { prompt != nil }, set: { if !$0 { prompt = nil } })
    }

    private var isDeleting: Binding<Bool> {
        Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })
    }

    private var deleteTitle: String {
        let name = deleting?.name ?? ""
        return L("「\(name)」を削除しますか？", "Delete \"\(name)\"?")
    }

    private func confirmDelete() {
        if let deleting { session.delete(deleting.path) }
        deleting = nil
    }

    private var header: some View {
        HStack {
            Text(session.project.name)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
            Spacer()
            Menu {
                Button { ask(.newFile, "") } label: { Label(L("新しいファイル", "New File"), systemImage: "doc.badge.plus") }
                Button { ask(.newFolder, "") } label: { Label(L("新しいフォルダ", "New Folder"), systemImage: "folder.badge.plus") }
                Button { session.refreshTree() } label: { Label(L("再読み込み", "Reload"), systemImage: "arrow.clockwise") }
            } label: {
                Image(systemName: "plus.circle")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func menu(for entry: WorkspaceEntry) -> some View {
        if entry.isDirectory {
            Button { ask(.newFile, entry.path) } label: { Label(L("ここに新しいファイル", "New File Here"), systemImage: "doc.badge.plus") }
            Button { ask(.newFolder, entry.path) } label: { Label(L("ここに新しいフォルダ", "New Folder Here"), systemImage: "folder.badge.plus") }
        } else {
            Button { session.open(entry.path) } label: { Label(L("開く", "Open"), systemImage: "doc.text") }
            if entry.path.lowercased().hasSuffix(".js") {
                Button {
                    session.runJavaScript(path: entry.path)
                    session.pane = .preview
                } label: { Label(L("実行", "Run"), systemImage: "play") }
            }
            if entry.path.lowercased().hasSuffix(".html") {
                Button {
                    session.previewFile = entry.path
                    session.pane = .preview
                    session.reloadPreview()
                } label: { Label(L("プレビュー", "Preview"), systemImage: "play.rectangle") }
            }
        }
        Button { ask(.rename, entry.path, name: entry.name) } label: { Label(L("名前を変更", "Rename"), systemImage: "pencil") }
        Button { UIPasteboard.general.string = entry.path } label: { Label(L("パスをコピー", "Copy Path"), systemImage: "doc.on.doc") }
        Button(role: .destructive) { deleting = entry } label: { Label(L("削除", "Delete"), systemImage: "trash") }
    }

    private func tap(_ entry: WorkspaceEntry) {
        if entry.isDirectory {
            withAnimation(.easeOut(duration: 0.15)) { session.toggle(entry.path) }
        } else {
            session.open(entry.path)
        }
    }

    private func ask(_ action: NamePrompt.Action, _ path: String, name: String = "") {
        promptText = name
        prompt = NamePrompt(action: action, path: path)
    }

    private func commitPrompt() {
        guard let prompt else { return }
        let name = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        self.prompt = nil
        guard !name.isEmpty else { return }
        switch prompt.action {
        case .newFile: session.createFile(named: name, in: prompt.path)
        case .newFolder: session.createFolder(named: name, in: prompt.path)
        case .rename: session.rename(prompt.path, to: name)
        }
    }
}

private struct FileRow: View {
    let entry: WorkspaceEntry
    let isExpanded: Bool
    let isSelected: Bool
    let isDirty: Bool

    var body: some View {
        HStack(spacing: 6) {
            if entry.isDirectory {
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.semibold))
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    .foregroundStyle(.secondary)
                    .frame(width: 12)
            } else {
                Color.clear.frame(width: 12)
            }
            Image(systemName: FileIcon.symbol(for: entry))
                .foregroundStyle(FileIcon.color(for: entry))
                .frame(width: 18)
            Text(entry.name)
                .font(.callout)
                .lineLimit(1)
            if isDirty {
                Circle().fill(Color.secondary).frame(width: 6, height: 6)
            }
            Spacer(minLength: 0)
        }
        .padding(.leading, CGFloat(entry.depth) * 14 + 8)
        .padding(.trailing, 8)
        .padding(.vertical, 5)
        .background(isSelected ? Color.accentColor.opacity(0.18) : Color.clear)
        .contentShape(Rectangle())
    }
}

enum FileIcon {
    static func symbol(for entry: WorkspaceEntry) -> String {
        if entry.isDirectory {
            return entry.name.hasSuffix(".swiftpm") ? "shippingbox" : "folder"
        }
        switch (entry.name as NSString).pathExtension.lowercased() {
        case "swift": return "swift"
        case "html", "htm": return "globe"
        case "css": return "paintbrush"
        case "js", "ts", "jsx", "tsx", "mjs": return "curlybraces"
        case "json": return "curlybraces.square"
        case "md", "txt": return "doc.text"
        case "png", "jpg", "jpeg", "gif", "svg", "heic", "webp": return "photo"
        case "py": return "chevron.left.forwardslash.chevron.right"
        default: return "doc"
        }
    }

    static func color(for entry: WorkspaceEntry) -> Color {
        if entry.isDirectory { return .accentColor }
        switch (entry.name as NSString).pathExtension.lowercased() {
        case "swift": return .orange
        case "html", "htm": return .blue
        case "css": return .purple
        case "js", "ts", "jsx", "tsx", "mjs": return .yellow
        case "json": return .green
        default: return .secondary
        }
    }
}
