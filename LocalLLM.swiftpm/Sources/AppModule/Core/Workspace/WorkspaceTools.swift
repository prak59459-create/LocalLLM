import Foundation

/// コーディングワークスペースでエージェントが使うツール。
enum WorkspaceToolbox {
    /// 渡すツールの数（小さいモデルはコンテキストが短いので、少なくもできる）
    enum Profile: String, CaseIterable, Codable, Identifiable, Sendable {
        case minimal
        case standard
        case full

        var id: String { rawValue }

        var label: String {
            switch self {
            case .minimal: return L("最小（読む・書く・直す）", "Minimal (read, write, edit)")
            case .standard: return L("標準（＋探す・ロードマップ）", "Standard (+ search, roadmap)")
            case .full: return L("すべて（＋消す・移動・JS）", "All (+ delete, move, JS)")
            }
        }
    }

    enum Name {
        static let listFiles = "list_files"
        static let readFile = "read_file"
        static let writeFile = "write_file"
        static let editFile = "edit_file"
        static let searchFiles = "search_files"
        static let findFiles = "find_files"
        static let makeDirectory = "make_directory"
        static let deleteFile = "delete_file"
        static let moveFile = "move_file"
        static let roadmapList = "roadmap_list"
        static let roadmapAdd = "roadmap_add"
        static let roadmapUpdate = "roadmap_update"
        /// 追加と更新をまとめたもの（id があれば更新）
        static let roadmap = "roadmap"
        static let runJavaScript = "run_javascript"
    }

    /// ファイルを変えるツール（実行の前に確認する）
    static let mutatingTools: Set<String> = [Name.writeFile, Name.editFile, Name.makeDirectory, Name.deleteFile, Name.moveFile]

    /// 中身を消すなど、元に戻しにくいツール（「編集は自動で許可」でも確認する）
    static let destructiveTools: Set<String> = [Name.deleteFile]

    /// モデルに渡すツール。小さいモデルでも迷わず、コンテキストも食わないよう、数も説明も最小限にしてある。
    /// - 最小: 読む・書く・直す
    /// - 標準: ＋ 探す（空ならファイル一覧）・ロードマップ
    /// - すべて: ＋ 消す・移動・JavaScript の実行
    static func definitions(profile: Profile, javaScript: Bool) -> [ToolDefinition] {
        let path = (L("パス", "path"), true)
        var tools: [ToolDefinition] = [
            ToolDefinition(name: Name.readFile, description: L("ファイルを読む（行番号付き）", "Read a file (numbered lines)"),
                           arguments: [("path", path.0, true), ("offset", L("始める行", "first line"), false)]),
            ToolDefinition(name: Name.writeFile, description: L("ファイルを作る・全体を書き換える", "Create or overwrite a file"),
                           arguments: [("path", path.0, true), ("content", L("中身全体", "full content"), true)]),
            ToolDefinition(name: Name.editFile, description: L("old_string（1 か所だけ）を new_string に置き換える", "Replace old_string (unique) with new_string"),
                           arguments: [("path", path.0, true), ("old_string", "old", true), ("new_string", "new", true)]),
        ]
        if profile != .minimal {
            tools.append(ToolDefinition(name: Name.searchFiles, description: L("文字列を探す（空ならファイル一覧）", "Search text (empty: list files)"),
                                        arguments: [("pattern", L("探す文字列", "text"), false)]))
            tools.append(ToolDefinition(name: Name.roadmap,
                                        description: L("ロードマップ: id 無しで追加、id 有りで更新（status: idea/planned/in_progress/done/blocked）",
                                                       "Roadmap: add (no id) or update (id; status: idea/planned/in_progress/done/blocked)"),
                                        arguments: [("id", L("番号（例 #2）", "number like #2"), false), ("title", L("名前", "title"), false),
                                                    ("status", "status", false), ("parent", L("親の番号", "parent number"), false)]))
        }
        if profile == .full {
            tools.append(ToolDefinition(name: Name.deleteFile, description: L("ファイル・フォルダを消す", "Delete a file or folder"),
                                        arguments: [("path", path.0, true)]))
            tools.append(ToolDefinition(name: Name.moveFile, description: L("移動・名前の変更", "Move or rename"),
                                        arguments: [("from", "from", true), ("to", "to", true)]))
            if javaScript {
                tools.append(ToolDefinition(name: Name.runJavaScript, description: L("JavaScript を実行して出力を返す", "Run JavaScript and return the output"),
                                            arguments: [("code", "code", false), ("path", L(".js ファイル", ".js file"), false)]))
            }
        }
        return tools
    }

    static func isWorkspaceTool(_ name: String) -> Bool {
        [Name.listFiles, Name.readFile, Name.writeFile, Name.editFile, Name.searchFiles, Name.findFiles, Name.makeDirectory,
         Name.deleteFile, Name.moveFile, Name.roadmapList, Name.roadmapAdd, Name.roadmapUpdate, Name.roadmap].contains(name)
    }
}

/// ツールを実行した結果。
struct WorkspaceToolResult: Sendable {
    var text: String
    var isError: Bool
    var changes: [FileChange] = []
    var roadmapChanged = false
    /// 画面で開くとよいファイル
    var touchedFile: String?
}

/// 確認画面に出す、これから行う変更。
struct ProposedChange: Sendable {
    var path: String
    var before: String?
    var after: String?
    var description: String
}

/// ファイルとロードマップのツールを実行する。
struct WorkspaceToolExecutor: Sendable {
    let fileSystem: WorkspaceFileSystem

    /// 引数の名前の揺れ（file / file_path / filename など）を吸収して読む
    static func argument(_ call: ToolCall, _ keys: [String]) -> String? {
        for key in keys {
            if let value = call.argument(key) { return value }
        }
        return nil
    }

    static func path(_ call: ToolCall) -> String? {
        argument(call, ["path", "file_path", "file", "filename", "filepath", "folder", "directory", "dir"])
    }

    func execute(_ call: ToolCall, roadmap: inout Roadmap) -> WorkspaceToolResult {
        do {
            return try run(call, roadmap: &roadmap)
        } catch {
            return WorkspaceToolResult(text: error.localizedDescription, isError: true)
        }
    }

    private func required(_ value: String?, _ name: String) throws -> String {
        guard let value, !value.isEmpty else {
            throw WorkspaceError.invalidArgument(L("引数 \(name) がありません", "Missing argument: \(name)"))
        }
        return value
    }

    private func run(_ call: ToolCall, roadmap: inout Roadmap) throws -> WorkspaceToolResult {
        typealias Name = WorkspaceToolbox.Name
        switch call.name {
        case Name.listFiles:
            let path = Self.path(call) ?? ""
            let tree = fileSystem.tree(path, maxDepth: 3, maxEntries: 80)
            return WorkspaceToolResult(text: tree.isEmpty ? L("（空）", "(empty)") : tree, isError: false)

        case Name.readFile:
            let path = try required(Self.path(call), "path")
            let offset = Self.argument(call, ["offset", "start", "line", "start_line"]).flatMap { Int($0) } ?? 1
            // 1 回で読むのは少しずつ（コンテキストとメモリを食わないように）
            let limit = min(Self.argument(call, ["limit", "lines", "count"]).flatMap { Int($0) } ?? 160, 300)
            let read = try fileSystem.readNumbered(path, offset: offset, limit: limit, maxCharacters: 4500)
            var text = read.text.isEmpty ? L("（空）", "(empty)") : read.text
            if read.shownUntil < read.totalLines {
                text += L("\n…続きは offset: \(read.shownUntil + 1)（全 \(read.totalLines) 行）",
                          "\n… continue with offset: \(read.shownUntil + 1) (\(read.totalLines) lines)")
            }
            return WorkspaceToolResult(text: text, isError: false, touchedFile: try fileSystem.normalize(path))

        case Name.writeFile:
            let path = try required(Self.path(call), "path")
            let content = Self.argument(call, ["content", "text", "code", "contents", "data"]) ?? ""
            let change = try fileSystem.write(path, content)
            return WorkspaceToolResult(text: "OK \(change.path) (\(TextDiff.lines(content).count) " + L("行", "lines") + ")",
                                       isError: false, changes: [change], touchedFile: change.path)

        case Name.editFile:
            let path = try required(Self.path(call), "path")
            let old = Self.argument(call, ["old_string", "old", "search", "find", "old_text", "target"]) ?? ""
            let new = Self.argument(call, ["new_string", "new", "replace", "replacement", "new_text"]) ?? ""
            let replaceAll = ["true", "1", "yes"].contains((Self.argument(call, ["replace_all", "all"]) ?? "").lowercased())
            let change = try fileSystem.edit(path, old: old, new: new, replaceAll: replaceAll)
            let stats = change.stats
            return WorkspaceToolResult(text: "OK \(change.path) +\(stats.added) −\(stats.removed)",
                                       isError: false, changes: [change], touchedFile: change.path)

        case Name.searchFiles:
            let pattern = (Self.argument(call, ["pattern", "query", "text", "regex", "name"]) ?? "").trimmingCharacters(in: .whitespaces)
            // 空ならファイルの一覧（list_files の代わり）
            if pattern.isEmpty || pattern == "*" {
                let tree = fileSystem.tree(Self.path(call) ?? "", maxDepth: 3, maxEntries: 80)
                return WorkspaceToolResult(text: tree.isEmpty ? L("（空）", "(empty)") : tree, isError: false)
            }
            let isRegex = ["true", "1", "yes"].contains((Self.argument(call, ["regex", "is_regex"]) ?? "").lowercased())
            let matches = try fileSystem.search(pattern, in: Self.path(call) ?? "", isRegex: isRegex, limit: 25)
            var lines = matches.map { match -> String in
                let text = match.text.trimmingCharacters(in: .whitespaces)
                return "\(match.path):\(match.line): " + (text.count > 100 ? String(text.prefix(100)) + "…" : text)
            }
            // ファイル名に含まれるものも（find_files の代わり）
            let named = ((try? fileSystem.find("*\(pattern)*", limit: 10)) ?? []).filter { name in !lines.contains { $0.hasPrefix(name + ":") } }
            lines += named
            return WorkspaceToolResult(text: lines.isEmpty ? L("見つかりませんでした", "No matches") : lines.joined(separator: "\n"), isError: false)

        case Name.findFiles:
            let pattern = try required(Self.argument(call, ["pattern", "name", "glob", "query"]), "pattern")
            let found = try fileSystem.find(pattern)
            return WorkspaceToolResult(text: found.isEmpty ? L("見つかりませんでした", "No matches") : found.joined(separator: "\n"),
                                       isError: false)

        case Name.makeDirectory:
            let path = try required(Self.path(call), "path")
            try fileSystem.makeDirectory(path)
            return WorkspaceToolResult(text: L("フォルダを作りました: \(path)", "Created folder \(path)"), isError: false)

        case Name.deleteFile:
            let path = try required(Self.path(call), "path")
            let changes = try fileSystem.delete(path)
            return WorkspaceToolResult(text: L("消しました: \(path)", "Deleted \(path)"), isError: false, changes: changes)

        case Name.moveFile:
            let from = try required(Self.argument(call, ["from", "source", "path", "old_path"]), "from")
            let to = try required(Self.argument(call, ["to", "destination", "new_path", "target"]), "to")
            let change = try fileSystem.move(from, to: to)
            return WorkspaceToolResult(text: L("移動しました: \(from) → \(change.path)", "Moved \(from) → \(change.path)"),
                                       isError: false, changes: [change])

        case Name.roadmapList:
            return WorkspaceToolResult(text: roadmap.summary(maxNodes: 80), isError: false)

        case Name.roadmap:
            // id があれば更新、無ければ追加（title も無ければ一覧）
            if Self.argument(call, ["id", "number", "node", "item"]) != nil {
                var update = call
                update.name = Name.roadmapUpdate
                return try run(update, roadmap: &roadmap)
            }
            if Self.argument(call, ["title", "name", "feature", "idea"]) == nil {
                return WorkspaceToolResult(text: roadmap.summary(maxNodes: 30), isError: false)
            }
            var add = call
            add.name = Name.roadmapAdd
            return try run(add, roadmap: &roadmap)

        case Name.roadmapAdd:
            let title = try required(Self.argument(call, ["title", "name", "feature", "idea"]), "title")
            let detail = Self.argument(call, ["detail", "description", "details", "body"]) ?? ""
            let parent = Self.argument(call, ["parent", "from", "parent_id", "after"]).flatMap { roadmap.node(reference: $0) }
            let kind = Self.argument(call, ["kind", "type"]).flatMap(RoadmapNode.Kind.parse) ?? (parent == nil ? .feature : .idea)
            let status = Self.argument(call, ["status", "state"]).flatMap(RoadmapNode.Status.parse) ?? .idea
            let node = roadmap.add(title: title, detail: detail, kind: kind, status: status, parent: parent?.id)
            return WorkspaceToolResult(text: "OK \(node.reference) \(node.title)",
                                       isError: false, roadmapChanged: true)

        case Name.roadmapUpdate:
            let reference = try required(Self.argument(call, ["id", "number", "node", "item", "title"]), "id")
            guard let node = roadmap.node(reference: reference) else {
                throw WorkspaceError.invalidArgument(L("ロードマップに \(reference) はありません", "No roadmap item \(reference)"))
            }
            let status = Self.argument(call, ["status", "state"]).flatMap(RoadmapNode.Status.parse)
            let detail = Self.argument(call, ["detail", "description", "details"])
            let title = Self.argument(call, ["new_title", "rename"])
            let files = Self.argument(call, ["files"])
            roadmap.update(node.id) { item in
                if let status { item.status = status }
                if let detail { item.detail = detail }
                if let title { item.title = title }
                if let files {
                    item.files = files.split(whereSeparator: { $0 == "," || $0 == "\n" })
                        .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                }
            }
            let updated = roadmap.node(node.id) ?? node
            let statusName = updated.status == .inProgress ? "in_progress" : updated.status.rawValue
            return WorkspaceToolResult(text: "OK \(updated.reference) \(statusName)",
                                       isError: false, roadmapChanged: true)

        default:
            throw WorkspaceError.invalidArgument(L("ツール \(call.name) はありません", "Unknown tool \(call.name)"))
        }
    }

    /// 確認画面用に、ファイルがどう変わるかを先に計算する。
    func preview(_ call: ToolCall) -> ProposedChange? {
        typealias Name = WorkspaceToolbox.Name
        switch call.name {
        case Name.writeFile:
            guard let path = Self.path(call), let normalized = try? fileSystem.normalize(path) else { return nil }
            let before = fileSystem.exists(normalized) ? (try? fileSystem.read(normalized)) : nil
            let content = Self.argument(call, ["content", "text", "code", "contents", "data"]) ?? ""
            return ProposedChange(path: normalized, before: before, after: content,
                                  description: before == nil ? L("新しいファイル", "New file") : L("全体を書き換え", "Replace whole file"))
        case Name.editFile:
            guard let path = Self.path(call), let normalized = try? fileSystem.normalize(path) else { return nil }
            let old = Self.argument(call, ["old_string", "old", "search", "find", "old_text", "target"]) ?? ""
            let new = Self.argument(call, ["new_string", "new", "replace", "replacement", "new_text"]) ?? ""
            let replaceAll = ["true", "1", "yes"].contains((Self.argument(call, ["replace_all", "all"]) ?? "").lowercased())
            guard let result = try? fileSystem.editedContent(normalized, old: old, new: new, replaceAll: replaceAll) else {
                return ProposedChange(path: normalized, before: nil, after: nil,
                                      description: L("置き換える場所が見つかりません（実行するとエラーになります）",
                                                     "The text to replace was not found (this will fail)"))
            }
            return ProposedChange(path: normalized, before: result.before, after: result.after, description: L("一部を置き換え", "Partial edit"))
        case Name.deleteFile:
            guard let path = Self.path(call), let normalized = try? fileSystem.normalize(path) else { return nil }
            let before = fileSystem.isDirectory(normalized) ? nil : (try? fileSystem.read(normalized))
            return ProposedChange(path: normalized, before: before, after: nil,
                                  description: fileSystem.isDirectory(normalized) ? L("フォルダごと削除", "Delete folder") : L("削除", "Delete"))
        case Name.moveFile:
            let from = Self.argument(call, ["from", "source", "path", "old_path"]) ?? ""
            let to = Self.argument(call, ["to", "destination", "new_path", "target"]) ?? ""
            return ProposedChange(path: to, before: nil, after: nil, description: "\(from) → \(to)")
        case Name.makeDirectory:
            return ProposedChange(path: Self.path(call) ?? "", before: nil, after: nil, description: L("フォルダを作る", "Create folder"))
        default:
            return nil
        }
    }
}

/// コーディングエージェントへの指示。
enum CodingAgentPrompt {
    static func system(projectName: String, tree: String, roadmap: String, notes: String?, openFile: String?,
                       javaScript: Bool, usesRoadmap: Bool = true, extra: String? = nil) -> String {
        // 小さいモデルでも守れるよう短く。毎回送るので、短いほど速くメモリも食わない。
        var parts: [String] = [L("""
        iPad のコーディングエージェント。プロジェクト「\(projectName)」をツールで編集して依頼を実装する。
        - 変える前に read_file で読む。小さな変更は edit_file、新規・全体は write_file。
        - ツールは前置きなしで呼ぶ。終わったら変えた内容を 1〜2 文で伝える。
        """, """
        Coding agent on an iPad. Edit the project "\(projectName)" with the tools to do what the user asks.
        - Read with read_file before changing. edit_file for small changes, write_file for new files or rewrites.
        - Call tools without preamble. When done, summarize the change in 1–2 sentences.
        """)]
        if usesRoadmap {
            parts[0] += L("\n- 機能ができたら roadmap で done にする。", "\n- Mark finished features done with roadmap.")
        }
        if javaScript {
            parts[0] += L("\n- run_javascript で動作を確かめられる。", "\n- run_javascript can check JavaScript.")
        }
        parts.append(L("ファイル:", "Files:") + "\n" + (tree.isEmpty ? L("（空）", "(empty)") : tree))
        if usesRoadmap, !roadmap.isEmpty { parts.append(L("ロードマップ:", "Roadmap:") + "\n" + roadmap) }
        if let openFile { parts.append(L("開いているファイル: ", "Open file: ") + openFile) }
        if let notes, !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parts.append(L("メモ（CDWS.md）:", "Notes (CDWS.md):") + "\n" + String(notes.prefix(600)))
        }
        if let extra, !extra.isEmpty { parts.append(extra) }
        return parts.joined(separator: "\n\n")
    }

    /// ロードマップの丸を実装してもらう依頼
    static func implement(_ node: RoadmapNode) -> String {
        let detail = node.detail.isEmpty ? "" : L("\n詳細: \(node.detail)", "\nDetails: \(node.detail)")
        return L("ロードマップの \(node.reference)「\(node.title)」を実装して。\(detail)\n終わったら roadmap（id: \(node.reference), status: done）。",
                 "Implement roadmap item \(node.reference) \"\(node.title)\".\(detail)\nWhen done, call roadmap (id: \(node.reference), status: done).")
    }

    /// 丸から新しい案を生やしてもらう依頼
    static func brainstorm(from node: RoadmapNode, count: Int = 3) -> String {
        L("ロードマップの \(node.reference)「\(node.title)」から発展させる新しいアイデアを \(count) つ、roadmap（title, parent: \(node.reference)）で足して。",
          "Add \(count) new ideas that grow out of roadmap item \(node.reference) \"\(node.title)\" with roadmap (title, parent: \(node.reference)).")
    }

    /// 最初のロードマップを作ってもらう依頼
    static func planRoadmap(goal: String) -> String {
        L("次の目標の実装計画を、機能ごとに roadmap（title、依存は parent）で足して。目標: \(goal)",
          "Plan this goal: add each feature with roadmap (title; link dependent steps with parent). Goal: \(goal)")
    }
}
