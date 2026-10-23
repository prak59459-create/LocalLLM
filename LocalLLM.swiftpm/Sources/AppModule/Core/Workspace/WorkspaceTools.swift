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
            case .standard: return L("標準", "Standard")
            case .full: return L("すべて", "All")
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
        static let runJavaScript = "run_javascript"
    }

    /// ファイルを変えるツール（実行の前に確認する）
    static let mutatingTools: Set<String> = [Name.writeFile, Name.editFile, Name.makeDirectory, Name.deleteFile, Name.moveFile]

    /// 中身を消すなど、元に戻しにくいツール（「編集は自動で許可」でも確認する）
    static let destructiveTools: Set<String> = [Name.deleteFile]

    static func definitions(profile: Profile, javaScript: Bool) -> [ToolDefinition] {
        var tools: [ToolDefinition] = [
            ToolDefinition(name: Name.readFile,
                           description: L("ファイルを行番号付きで読む。", "Read a file with line numbers."),
                           arguments: [("path", L("ファイルのパス", "file path"), true),
                                       ("offset", L("始める行（省略可）", "first line (optional)"), false)]),
            ToolDefinition(name: Name.writeFile,
                           description: L("ファイルを作る、または全体を書き換える。", "Create a file or replace its whole content."),
                           arguments: [("path", L("ファイルのパス", "file path"), true),
                                       ("content", L("ファイルの中身全体", "the full file content"), true)]),
            ToolDefinition(name: Name.editFile,
                           description: L("ファイルの一部を置き換える。old_string は今の中身と完全に同じで、1 か所だけにあること。",
                                          "Replace part of a file. old_string must match the current text exactly and occur once."),
                           arguments: [("path", L("ファイルのパス", "file path"), true),
                                       ("old_string", L("置き換える元の文字列", "text to replace"), true),
                                       ("new_string", L("新しい文字列", "replacement text"), true)]),
        ]
        if profile != .minimal {
            tools.insert(ToolDefinition(name: Name.listFiles,
                                        description: L("フォルダの中のファイルの一覧。", "List files in a folder."),
                                        arguments: [("path", L("フォルダ（省略でプロジェクト全体）", "folder (optional)"), false)]), at: 0)
            tools += [
                ToolDefinition(name: Name.searchFiles,
                               description: L("プロジェクトの中の文字列を探して、ファイル名と行を返す。", "Search text in the project; returns file and line."),
                               arguments: [("pattern", L("探す文字列", "text to find"), true)]),
                ToolDefinition(name: Name.roadmapAdd,
                               description: L("ロードマップに機能やアイデアの丸を足す。", "Add a feature or idea to the roadmap."),
                               arguments: [("title", L("短い名前", "short title"), true),
                                           ("detail", L("説明（省略可）", "description (optional)"), false),
                                           ("parent", L("つなぐ元の番号（例 #2、省略可）", "parent number like #2 (optional)"), false)]),
                ToolDefinition(name: Name.roadmapUpdate,
                               description: L("ロードマップの丸の状態などを変える。status は idea / planned / in_progress / done / blocked。",
                                              "Update a roadmap item. status: idea / planned / in_progress / done / blocked."),
                               arguments: [("id", L("番号（例 #3）", "number like #3"), true),
                                           ("status", L("新しい状態（省略可）", "new status (optional)"), false),
                                           ("detail", L("新しい説明（省略可）", "new description (optional)"), false)]),
            ]
        }
        if profile == .full {
            tools += [
                ToolDefinition(name: Name.findFiles,
                               description: L("名前でファイルを探す（* が使える）。", "Find files by name (supports *)."),
                               arguments: [("pattern", L("名前のパターン（例 *.js）", "name pattern like *.js"), true)]),
                ToolDefinition(name: Name.makeDirectory,
                               description: L("フォルダを作る。", "Create a folder."),
                               arguments: [("path", L("フォルダのパス", "folder path"), true)]),
                ToolDefinition(name: Name.moveFile,
                               description: L("ファイルやフォルダを移動・名前の変更をする。", "Move or rename a file or folder."),
                               arguments: [("from", L("今のパス", "current path"), true), ("to", L("新しいパス", "new path"), true)]),
                ToolDefinition(name: Name.deleteFile,
                               description: L("ファイルやフォルダを消す。", "Delete a file or folder."),
                               arguments: [("path", L("パス", "path"), true)]),
                ToolDefinition(name: Name.roadmapList,
                               description: L("ロードマップの丸の一覧。", "List the roadmap items."),
                               arguments: []),
            ]
        }
        if javaScript {
            tools.append(ToolDefinition(name: Name.runJavaScript,
                                        description: L("JavaScript を実行して console.log の出力と最後の値を返す。path でファイルを実行もできる。",
                                                       "Run JavaScript and return console.log output and the result. Can run a file via path."),
                                        arguments: [("code", L("実行するコード（path を使うなら省略）", "code to run (omit when using path)"), false),
                                                    ("path", L("実行する .js ファイル（省略可）", ".js file to run (optional)"), false)]))
        }
        return tools
    }

    static func isWorkspaceTool(_ name: String) -> Bool {
        [Name.listFiles, Name.readFile, Name.writeFile, Name.editFile, Name.searchFiles, Name.findFiles, Name.makeDirectory,
         Name.deleteFile, Name.moveFile, Name.roadmapList, Name.roadmapAdd, Name.roadmapUpdate].contains(name)
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
            let tree = fileSystem.tree(path, maxDepth: 3, maxEntries: 200)
            return WorkspaceToolResult(text: tree.isEmpty ? L("（空です）", "(empty)") : tree, isError: false)

        case Name.readFile:
            let path = try required(Self.path(call), "path")
            let offset = Self.argument(call, ["offset", "start", "line", "start_line"]).flatMap { Int($0) } ?? 1
            let limit = Self.argument(call, ["limit", "lines", "count"]).flatMap { Int($0) } ?? 400
            let read = try fileSystem.readNumbered(path, offset: offset, limit: limit, maxCharacters: 9000)
            var text = read.text.isEmpty ? L("（空のファイル）", "(empty file)") : read.text
            if read.shownUntil < read.totalLines {
                text += L("…（全 \(read.totalLines) 行。続きは offset: \(read.shownUntil + 1)）",
                          "… (\(read.totalLines) lines in total; continue with offset: \(read.shownUntil + 1))")
            }
            return WorkspaceToolResult(text: text, isError: false, touchedFile: try fileSystem.normalize(path))

        case Name.writeFile:
            let path = try required(Self.path(call), "path")
            let content = Self.argument(call, ["content", "text", "code", "contents", "data"]) ?? ""
            let change = try fileSystem.write(path, content)
            return WorkspaceToolResult(text: L("書きました: \(change.path)（\(TextDiff.lines(content).count) 行）",
                                               "Wrote \(change.path) (\(TextDiff.lines(content).count) lines)"),
                                       isError: false, changes: [change], touchedFile: change.path)

        case Name.editFile:
            let path = try required(Self.path(call), "path")
            let old = Self.argument(call, ["old_string", "old", "search", "find", "old_text", "target"]) ?? ""
            let new = Self.argument(call, ["new_string", "new", "replace", "replacement", "new_text"]) ?? ""
            let replaceAll = ["true", "1", "yes"].contains((Self.argument(call, ["replace_all", "all"]) ?? "").lowercased())
            let change = try fileSystem.edit(path, old: old, new: new, replaceAll: replaceAll)
            let stats = change.stats
            return WorkspaceToolResult(text: L("直しました: \(change.path)（+\(stats.added) −\(stats.removed)）",
                                               "Edited \(change.path) (+\(stats.added) −\(stats.removed))"),
                                       isError: false, changes: [change], touchedFile: change.path)

        case Name.searchFiles:
            let pattern = try required(Self.argument(call, ["pattern", "query", "text", "regex"]), "pattern")
            let isRegex = ["true", "1", "yes"].contains((Self.argument(call, ["regex", "is_regex"]) ?? "").lowercased())
            let matches = try fileSystem.search(pattern, in: Self.path(call) ?? "", isRegex: isRegex, limit: 50)
            let text = matches.isEmpty ? L("見つかりませんでした", "No matches")
                                       : matches.map { "\($0.path):\($0.line): \($0.text)" }.joined(separator: "\n")
            return WorkspaceToolResult(text: text, isError: false)

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

        case Name.roadmapAdd:
            let title = try required(Self.argument(call, ["title", "name", "feature", "idea"]), "title")
            let detail = Self.argument(call, ["detail", "description", "details", "body"]) ?? ""
            let parent = Self.argument(call, ["parent", "from", "parent_id", "after"]).flatMap { roadmap.node(reference: $0) }
            let kind = Self.argument(call, ["kind", "type"]).flatMap(RoadmapNode.Kind.parse) ?? (parent == nil ? .feature : .idea)
            let status = Self.argument(call, ["status", "state"]).flatMap(RoadmapNode.Status.parse) ?? .idea
            let node = roadmap.add(title: title, detail: detail, kind: kind, status: status, parent: parent?.id)
            return WorkspaceToolResult(text: L("ロードマップに \(node.reference)「\(node.title)」を足しました",
                                               "Added \(node.reference) \"\(node.title)\" to the roadmap"),
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
            return WorkspaceToolResult(text: L("\(updated.reference)「\(updated.title)」: \(updated.status.label)",
                                               "\(updated.reference) \"\(updated.title)\": \(updated.status.label)"),
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
                       javaScript: Bool, extra: String? = nil) -> String {
        var parts: [String] = []
        parts.append(L("""
        あなたは iPad の中で動くコーディングエージェントです。プロジェクト「\(projectName)」のファイルをツールで読み書きして、ユーザーの依頼を実装します。
        進め方:
        1. 変更する前に read_file で今の中身を読む（推測で書き換えない）。
        2. 小さな変更は edit_file、新しいファイルや全体の書き換えは write_file。
        3. 機能を作り終えたら roadmap_update で done にする。新しい案は roadmap_add で足す。
        4. 最後に、何を変えたかを短く日本語で伝える。
        ツールを使うときは説明を書かずにすぐ呼び出すこと。
        """, """
        You are a coding agent running on an iPad. Use the tools to read and write files in the project "\(projectName)" and implement what the user asks.
        How to work:
        1. Read files with read_file before changing them (never guess their content).
        2. Use edit_file for small changes and write_file for new files or full rewrites.
        3. When a feature is finished, mark it done with roadmap_update. Add new ideas with roadmap_add.
        4. Finish with a short summary of what you changed.
        When you use a tool, call it right away without extra explanation.
        """))
        if javaScript {
            parts.append(L("run_javascript で JavaScript を実行して動作を確かめられます。", "You can check JavaScript with run_javascript."))
        }
        parts.append(L("ファイル:", "Files:") + "\n" + (tree.isEmpty ? L("（空）", "(empty)") : tree))
        parts.append(L("ロードマップ:", "Roadmap:") + "\n" + roadmap)
        if let openFile { parts.append(L("ユーザーが開いているファイル: ", "File open in the editor: ") + openFile) }
        if let notes, !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parts.append(L("プロジェクトのメモ（CDWS.md）:", "Project notes (CDWS.md):") + "\n" + String(notes.prefix(1500)))
        }
        if let extra, !extra.isEmpty { parts.append(extra) }
        return parts.joined(separator: "\n\n")
    }

    /// ロードマップの丸を実装してもらう依頼
    static func implement(_ node: RoadmapNode) -> String {
        let detail = node.detail.isEmpty ? "" : L("\n詳細: \(node.detail)", "\nDetails: \(node.detail)")
        return L("ロードマップの \(node.reference)「\(node.title)」を実装してください。\(detail)\n必要なファイルを読んでから変更し、終わったら roadmap_update で \(node.reference) を done にしてください。",
                 "Please implement roadmap item \(node.reference) \"\(node.title)\".\(detail)\nRead the files you need before changing them, and mark \(node.reference) done with roadmap_update when finished.")
    }

    /// 丸から新しい案を生やしてもらう依頼
    static func brainstorm(from node: RoadmapNode, count: Int = 3) -> String {
        L("ロードマップの \(node.reference)「\(node.title)」から発展させられる新しいアイデアを \(count) つ考えて、それぞれ roadmap_add（parent: \(node.reference)）で足してください。最後に一言ずつ説明してください。",
          "Think of \(count) new ideas that grow out of roadmap item \(node.reference) \"\(node.title)\" and add each with roadmap_add (parent: \(node.reference)). Then explain each in one sentence.")
    }

    /// 最初のロードマップを作ってもらう依頼
    static func planRoadmap(goal: String) -> String {
        L("次の目標のための実装計画を立てて、機能ごとに roadmap_add でロードマップに足してください（順番に依存するものは parent でつなぐ）。目標: \(goal)",
          "Plan the implementation for this goal and add each feature to the roadmap with roadmap_add (link dependent steps with parent). Goal: \(goal)")
    }
}
