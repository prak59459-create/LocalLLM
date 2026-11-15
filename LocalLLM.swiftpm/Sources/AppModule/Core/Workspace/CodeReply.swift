import Foundation

/// 「かんたん」方式（小さいモデル向け）。
///
/// 小さいモデルは、ツールの呼び出し（JSON の中にファイル全体を文字列として入れる）を崩しやすい。
/// そこで書くときは、ふだんどおり「ファイル名の行 + コードブロック（ファイル全体）」で答えてもらい、アプリがそれを読み取ってファイルに書く。
/// ほかのツールは「READ index.html」のような 1 行のコマンド（十数トークン）で呼べる。今のファイルの中身はプロンプトに入れてある。
enum CodeReply {
    /// 返答から取り出した 1 つのファイル
    struct File: Equatable, Sendable {
        var path: String
        var content: String
    }

    /// 書かなかったコードブロック（理由つき）
    struct Skipped: Equatable, Sendable {
        var path: String?
        var reason: String
    }

    struct Result: Equatable, Sendable {
        var files: [File]
        var skipped: [Skipped]
        /// コードブロックの外に書かれた短いコマンド（READ index.html など）。書かれた順
        var commands: [Command] = []
    }

    /// 1 行で書ける短いツールの呼び出し（十数トークンで済む。JSON は要らない）
    struct Command: Equatable, Sendable {
        enum Kind: String, Sendable, CaseIterable {
            case read = "READ", list = "LIST", find = "FIND", delete = "DELETE", move = "MOVE", run = "RUN"
        }
        var kind: Kind
        var arguments: [String]

        /// 結果をモデルに返して、続きを書いてもらうコマンド
        var needsReply: Bool { kind == .read || kind == .list || kind == .find || kind == .run }

        /// 画面やモデルに見せる 1 行
        var line: String { ([kind.rawValue] + arguments).joined(separator: " ") }

        /// いまあるツールの呼び出しに直す（確認・変更履歴・元に戻すはそのまま使える）
        var toolCall: ToolCall? {
            typealias Name = WorkspaceToolbox.Name
            switch kind {
            case .read:
                guard let path = arguments.first else { return nil }
                return ToolCall(name: Name.readFile, arguments: .object([("path", .string(path))]))
            case .list:
                return ToolCall(name: Name.listFiles, arguments: .object([]))
            case .find:
                return ToolCall(name: Name.searchFiles, arguments: .object([("pattern", .string(arguments.joined(separator: " ")))]))
            case .delete:
                guard let path = arguments.first else { return nil }
                return ToolCall(name: Name.deleteFile, arguments: .object([("path", .string(path))]))
            case .move:
                guard arguments.count >= 2 else { return nil }
                return ToolCall(name: Name.moveFile, arguments: .object([("from", .string(arguments[0])), ("to", .string(arguments[1]))]))
            case .run:
                guard let path = arguments.first else { return nil }
                return ToolCall(name: Name.runJavaScript, arguments: .object([("path", .string(path))]))
            }
        }
    }

    /// 言いかえ（小さいモデルが書きがちなもの）
    private static let commandAliases: [String: Command.Kind] = [
        "READ": .read, "OPEN": .read, "SHOW": .read, "CAT": .read, "VIEW": .read,
        "LIST": .list, "LS": .list, "FILES": .list,
        "FIND": .find, "SEARCH": .find, "GREP": .find,
        "DELETE": .delete, "REMOVE": .delete, "RM": .delete,
        "MOVE": .move, "RENAME": .move, "MV": .move,
        "RUN": .run, "EXEC": .run,
    ]

    /// 1 行をコマンドとして読む。大文字の命令（READ index.html）か、/ で始まるもの（/read index.html）だけ。
    /// 文章の中の「Read the file …」などをコマンドと間違えないよう、引数の形も確かめる。
    static func command(in rawLine: String) -> Command? {
        var line = rawLine.trimmingCharacters(in: .whitespaces)
        line = line.trimmingCharacters(in: CharacterSet(charactersIn: "`*"))
        for bullet in ["- ", "* ", "> "] where line.hasPrefix(bullet) { line = String(line.dropFirst(bullet.count)) }
        var slashed = false
        if line.hasPrefix("/") || line.hasPrefix("!") || line.hasPrefix("@") {
            line = String(line.dropFirst())
            slashed = true
        }
        let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
        guard var head = parts.first else { return nil }
        if head.hasSuffix(":") { head = String(head.dropLast()) }
        guard slashed || head == head.uppercased(), let kind = commandAliases[head.uppercased()] else { return nil }
        let arguments = parts.dropFirst().map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "`\"'*,")) }.filter { !$0.isEmpty }
        switch kind {
        case .list:
            return arguments.count <= 1 ? Command(kind: kind, arguments: []) : nil
        case .find:
            let text = parts.dropFirst().joined(separator: " ").trimmingCharacters(in: CharacterSet(charactersIn: "`\"' "))
            return text.isEmpty || text.count > 80 ? nil : Command(kind: kind, arguments: [text])
        case .read, .delete, .run:
            guard arguments.count == 1, let path = clean(arguments[0]) else { return nil }
            if kind == .run, !["js", "mjs"].contains(fileExtension(path)) { return nil }
            return Command(kind: kind, arguments: [path])
        case .move:
            let paths = arguments.filter { $0 != "->" && $0 != "→" && $0.lowercased() != "to" }.compactMap(clean)
            return paths.count == 2 ? Command(kind: kind, arguments: paths) : nil
        }
    }

    /// 小さいモデルは「index.html を書き換えて」と言っても calculator.html のような新しいページを作りがち。
    /// プレビューに出るのは index.html なので、依頼でその名前が出ていなければ、新しいページ 1 つを index.html に書く。
    static func keepMainPage(_ result: Result, knownFiles: [String], request: String) -> Result {
        let main = "index.html"
        guard knownFiles.contains(main), !result.files.contains(where: { $0.path == main }) else { return result }
        let newPages = result.files.indices.filter { index in
            let path = result.files[index].path
            return ["html", "htm"].contains(fileExtension(path)) && !knownFiles.contains(path)
        }
        guard newPages.count == 1, let index = newPages.first else { return result }
        let name = result.files[index].path
        let stem = (name.split(separator: "/").last.map(String.init) ?? name).components(separatedBy: ".").first ?? name
        guard !request.lowercased().contains(stem.lowercased()) else { return result }
        var changed = result
        changed.files[index].path = main
        return changed
    }

    /// 次の往復でモデルに渡す返答（コードブロックは「書いた」という一言にする。ファイル全体をまた読ませないため）
    static func condensed(_ text: String, written: [String]) -> String {
        var kept: [String] = []
        var inFence = false
        for line in text.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                inFence.toggle()
                continue
            }
            if !inFence { kept.append(line) }
        }
        var result = kept.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        if !written.isEmpty { result += (result.isEmpty ? "" : "\n") + "(wrote " + written.joined(separator: ", ") + ")" }
        return result
    }

    /// コードブロックの言語 → 拡張子
    static let extensions: [String: [String]] = [
        "html": ["html", "htm"], "htm": ["html", "htm"], "css": ["css"], "javascript": ["js", "mjs"], "js": ["js", "mjs"],
        "typescript": ["ts"], "ts": ["ts"], "jsx": ["jsx"], "tsx": ["tsx"], "json": ["json"], "python": ["py"], "py": ["py"],
        "swift": ["swift"], "markdown": ["md"], "md": ["md"], "xml": ["xml"], "svg": ["svg"], "yaml": ["yml", "yaml"],
        "yml": ["yml", "yaml"], "c": ["c", "h"], "cpp": ["cpp", "cc", "hpp"], "c++": ["cpp", "cc", "hpp"], "java": ["java"],
        "kotlin": ["kt"], "go": ["go"], "rust": ["rs"], "rs": ["rs"], "ruby": ["rb"], "rb": ["rb"], "php": ["php"], "sql": ["sql"],
        "vue": ["vue"], "csv": ["csv"],
    ]

    /// 名前が無いときに使う、拡張子ごとの決まった名前
    static let defaultNames: [String: String] = [
        "html": "index.html", "css": "style.css", "js": "script.js", "py": "main.py", "swift": "main.swift", "json": "data.json",
        "md": "README.md", "ts": "main.ts",
    ]

    /// コマンドや説明の例のことが多いので、ファイル名が無ければ書かない言語
    static let commandLanguages: Set<String> = ["bash", "sh", "shell", "zsh", "console", "terminal", "text", "plaintext", "txt",
                                                "output", "diff"]

    /// 返答からファイルを取り出す。
    /// - Parameters:
    ///   - knownFiles: プロジェクトにあるファイル（名前が書かれていないブロックの行き先を決めるのに使う）
    ///   - openFile: 開いているファイル
    static func parse(_ text: String, knownFiles: [String], openFile: String?) -> Result {
        var files: [File] = []
        var skipped: [Skipped] = []
        var commands: [Command] = []
        let lines = text.components(separatedBy: "\n")
        var index = 0
        while index < lines.count {
            let line = lines[index].trimmingCharacters(in: .whitespaces)
            // プロンプトの「Current files:」などをそのまま書き写し始めたら、そこで終わり（古い中身で上書きしないように）
            if echoMarkers.contains(where: { line.hasPrefix($0) }) { break }
            guard line.hasPrefix("```") else {
                if let command = command(in: line), !commands.contains(command) { commands.append(command) }
                index += 1
                continue
            }
            let info = String(line.drop { $0 == "`" }).trimmingCharacters(in: .whitespaces)
            var body: [String] = []
            var end = index + 1
            while end < lines.count, !isClosingFence(lines[end]) {
                body.append(lines[end])
                end += 1
            }
            let before = Array(lines[max(0, index - 2)..<index])
            let block = Block(info: info, body: body, before: before)
            switch resolve(block, knownFiles: knownFiles, openFile: openFile) {
            case .file(let file):
                files.removeAll { $0.path == file.path }
                files.append(file)
            case .skip(let reason):
                skipped.append(reason)
            case .ignore:
                break
            }
            index = end + 1
        }
        return Result(files: files, skipped: skipped, commands: commands)
    }

    private static let echoMarkers = ["Current files:", "Other files:", "Earlier requests", "Notes from the user", "Results:"]

    private struct Block {
        var info: String
        var body: [String]
        var before: [String]
    }

    private enum Resolution {
        case file(File)
        case skip(Skipped)
        case ignore
    }

    private static func isClosingFence(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.hasPrefix("```") && trimmed.allSatisfy { $0 == "`" }
    }

    private static func resolve(_ block: Block, knownFiles: [String], openFile: String?) -> Resolution {
        var content = block.body.joined(separator: "\n")
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .ignore }
        let (language, infoPath) = parseInfo(block.info)
        let wanted = language.flatMap { extensions[$0] } ?? []
        var path = infoPath ?? pathInComment(block.body.first ?? "") ?? pathBefore(block.before, preferring: wanted)
        if path == nil {
            if let language, commandLanguages.contains(language) { return .ignore }
            path = inferPath(language: language, content: content, knownFiles: knownFiles, openFile: openFile)
        }
        guard var found = path.flatMap(clean) else { return .ignore }
        // Index.html のように大文字・小文字だけ違うときは、今あるファイルに合わせる
        if !knownFiles.contains(found), let same = knownFiles.first(where: { $0.lowercased() == found.lowercased() }) { found = same }
        if !content.hasSuffix("\n") { content += "\n" }
        if isElided(content) {
            let exists = knownFiles.contains(found)
            if exists {
                return .skip(Skipped(path: found, reason: L("途中が省略されていたので、\(found) は書き換えませんでした（ファイル全体を書いてもらってください）。",
                                                            "\(found) was not changed because parts were left out (ask for the whole file).")))
            }
        }
        return .file(File(path: found, content: content))
    }

    /// ```html index.html / ```html:index.html / ```index.html / ```html title="index.html"
    static func parseInfo(_ info: String) -> (language: String?, path: String?) {
        guard !info.isEmpty else { return (nil, nil) }
        var text = info
        var path: String?
        if let range = text.range(of: #"(?:title|file|filename|path)\s*=\s*"([^"]+)""#, options: .regularExpression) {
            let match = String(text[range])
            if let quote = match.firstIndex(of: "\"") {
                path = String(match[match.index(after: quote)...].dropLast())
            }
            text.removeSubrange(range)
        }
        let parts = text.split(whereSeparator: { $0 == " " || $0 == ":" || $0 == "\t" }).map(String.init)
        var language: String?
        for part in parts {
            if path == nil, looksLikePath(part) {
                path = part
            } else if language == nil {
                language = part.lowercased()
            }
        }
        // ```index.html のように、名前だけのとき
        if language == nil, let path, let ext = path.split(separator: ".").last { language = String(ext).lowercased() }
        return (language, path)
    }

    /// 1 行目のコメントがファイル名だけ（<!-- index.html --> / // script.js / /* style.css */ / # main.py）
    static func pathInComment(_ line: String) -> String? {
        var text = line.trimmingCharacters(in: .whitespaces)
        for (open, close) in [("<!--", "-->"), ("/*", "*/"), ("//", ""), ("#", "")] where text.hasPrefix(open) {
            text = String(text.dropFirst(open.count))
            if !close.isEmpty, text.hasSuffix(close) { text = String(text.dropLast(close.count)) }
            text = text.trimmingCharacters(in: .whitespaces)
            for prefix in ["File:", "file:", "ファイル:", "ファイル："] where text.hasPrefix(prefix) {
                text = String(text.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
            }
            return looksLikePath(text) ? text : nil
        }
        return nil
    }

    /// ブロックの直前の行にあるファイル名（**index.html** / ### `script.js` / ファイル: style.css / index.html:）。
    /// 文の中に名前がいくつかあるときは、ブロックの言語に合うものを選ぶ
    static func pathBefore(_ lines: [String], preferring extensions: [String]) -> String? {
        for line in lines.reversed() {
            let tokens = pathTokens(in: line)
            if tokens.isEmpty {
                if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
                return nil
            }
            if extensions.isEmpty { return tokens.last }
            if let match = tokens.last(where: { extensions.contains(fileExtension($0)) }) { return match }
            return nil
        }
        return nil
    }

    private static let pathPattern = try? NSRegularExpression(pattern: #"[A-Za-z0-9_\-./]*[A-Za-z0-9_\-]\.[A-Za-z][A-Za-z0-9]{0,7}"#)

    static func pathTokens(in line: String) -> [String] {
        guard let pathPattern else { return [] }
        let range = NSRange(line.startIndex..., in: line)
        return pathPattern.matches(in: line, range: range).compactMap { match in
            Range(match.range, in: line).map { String(line[$0]) }
        }.filter(looksLikePath)
    }

    static func looksLikePath(_ text: String) -> Bool {
        let value = text.trimmingCharacters(in: CharacterSet(charactersIn: "`*\"'"))
        guard value.count <= 120, !value.contains(" "), !value.hasPrefix("http"), !value.contains("..") else { return false }
        let ext = fileExtension(value)
        guard !ext.isEmpty, ext.count <= 8, ext.first?.isLetter == true else { return false }
        // 数字の「1.5」や「e.g」のようなものは除く
        let name = value.split(separator: "/").last.map(String.init) ?? value
        return name.count > ext.count + 1 && extensions.values.contains(where: { $0.contains(ext) }) || knownExtension(ext)
    }

    private static func knownExtension(_ ext: String) -> Bool {
        ["txt", "toml", "ini", "cfg", "env", "lock", "gradle", "plist", "storyboard", "m", "mm", "scss", "sass", "less"].contains(ext)
    }

    static func fileExtension(_ path: String) -> String {
        guard let dot = path.lastIndex(of: "."), dot != path.startIndex else { return "" }
        return String(path[path.index(after: dot)...]).lowercased()
    }

    /// 名前が書かれていないブロックの行き先（言語 → 開いているファイル → 同じ種類のファイル → 決まった名前）
    static func inferPath(language: String?, content: String, knownFiles: [String], openFile: String?) -> String? {
        var wanted = language.flatMap { extensions[$0] } ?? []
        if wanted.isEmpty {
            let head = content.trimmingCharacters(in: .whitespacesAndNewlines).prefix(200).lowercased()
            if head.hasPrefix("<!doctype html") || head.hasPrefix("<html") { wanted = ["html", "htm"] }
        }
        guard let primary = wanted.first else { return nil }
        if let openFile, wanted.contains(fileExtension(openFile)) { return openFile }
        let same = knownFiles.filter { wanted.contains(fileExtension($0)) }
        if let preferred = defaultNames[primary], same.contains(preferred) { return preferred }
        if same.count == 1 { return same[0] }
        if let first = same.sorted(by: { $0.count < $1.count }).first { return first }
        return defaultNames[primary] ?? "main.\(primary)"
    }

    /// 「…」や「（省略）」「existing code」など、ファイルの一部を省いた印があるか
    static func isElided(_ content: String) -> Bool {
        let markers = ["existing code", "rest of the", "rest unchanged", "unchanged code", "same as before", "remains the same",
                       "省略", "以下同様", "前と同じ", "変更なし", "そのまま"]
        let lower = content.lowercased()
        for line in lower.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isComment = trimmed.hasPrefix("//") || trimmed.hasPrefix("#") || trimmed.hasPrefix("<!--") || trimmed.hasPrefix("/*")
            if isComment, markers.contains(where: { trimmed.contains($0) }) { return true }
            let bare = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "/#*<>!- "))
            if bare == "..." || bare == "…" || bare == "...." { return true }
        }
        return false
    }

    /// パスを整える（先頭の ./ や / を取り、引用符などを外す）。使えないものは nil
    static func clean(_ path: String) -> String? {
        var value = path.trimmingCharacters(in: CharacterSet(charactersIn: "`*\"' :"))
        while value.hasPrefix("./") { value = String(value.dropFirst(2)) }
        while value.hasPrefix("/") { value = String(value.dropFirst()) }
        guard !value.isEmpty, !value.contains(".."), looksLikePath(value) else { return nil }
        return value
    }
}

/// 「コードを書くだけ」の方式のプロンプト。小さいモデルでも守れるよう、短く、形の見本つき。
enum SimpleCodingPrompt {
    /// プロンプトに入れるファイルの中身
    struct ContextFile: Equatable, Sendable {
        var path: String
        var content: String
        /// 長すぎて途中までしか入れていない
        var truncated: Bool
    }

    /// プロンプトに入れるファイルを選ぶ（開いているファイル → index.html などの中心のファイル → ほかの小さいファイル）
    /// - Parameter request: 依頼に名前が出てくるファイルは、いちばん先に入れる（小さいモデルは READ を使わずに想像で書きがちなので）
    static func contextFiles(all: [String], openFile: String?, read: (String) -> String?, budget: Int,
                             request: String = "") -> (files: [ContextFile], others: [String]) {
        let main = ["index.html", "style.css", "styles.css", "script.js", "main.js", "app.js", "main.py", "README.md"]
        var order: [String] = []
        for token in CodeReply.pathTokens(in: request) {
            let name = token.lowercased()
            if let path = all.first(where: { $0.lowercased() == name || $0.lowercased().hasSuffix("/" + name) }), !order.contains(path) {
                order.append(path)
            }
        }
        if let openFile, all.contains(openFile), !order.contains(openFile) { order.append(openFile) }
        for name in main where all.contains(name) && !order.contains(name) { order.append(name) }
        let text = Set(CodeReply.extensions.values.flatMap { $0 } + ["txt", "scss", "less", "toml"])
        for path in all.sorted() where !order.contains(path) && text.contains(CodeReply.fileExtension(path)) && !path.hasPrefix(".") {
            order.append(path)
        }
        var files: [ContextFile] = []
        var used = 0
        for path in order {
            guard used < budget, let content = read(path) else { continue }
            let room = budget - used
            if content.count <= room {
                files.append(ContextFile(path: path, content: content, truncated: false))
                used += content.count
            } else if files.isEmpty || room > 800 {
                files.append(ContextFile(path: path, content: String(content.prefix(room)), truncated: true))
                used = budget
            }
        }
        let included = Set(files.map(\.path))
        let others = all.filter { !included.contains($0) && !$0.hasPrefix(".") }
        return (files, others)
    }

    static func system(projectName: String, files: [ContextFile], others: [String], earlier: [String], notes: String?,
                       javaScript: Bool = true) -> String {
        var parts: [String] = ["""
        You are a coding assistant in a code editor on an iPad. Change the files of the project "\(projectName)" so that they do exactly what the user asks now.

        How to answer:
        1. For each file you create or change: write the file name on its own line, then the WHOLE new file in one code block.
        2. Write every line of the file. Never leave anything out (no "...", no "rest unchanged").
        3. Do what the new request says, even if it is different from the current files. Replace old features that the user did not ask to keep.
        4. Keep the main page in index.html (that is the page that is shown).
        5. After the code blocks, say in one short sentence what you changed, in the user's language.

        Short commands (one per line, outside code blocks) if you need them:
        READ path       show a file that is not shown below
        LIST            list all files
        FIND text       search the files for text
        DELETE path     delete a file
        MOVE old new    rename a file\(javaScript ? "\nRUN file.js     run JavaScript and see its output" : "")
        After READ, LIST, FIND or RUN, stop writing. The results come in the next message.

        Format example (only the format, not the task):
        index.html
        ```html
        <!DOCTYPE html>
        <html><body><h1>Hello</h1></body></html>
        ```
        Made a page with a heading.
        """]
        if files.isEmpty {
            parts.append("The project has no files yet. Create what is needed (for a web page: index.html, and style.css / script.js if useful).")
        } else {
            var current = ["Current files:"]
            for file in files {
                let fence = fenceLanguage(file.path)
                current.append("\(file.path)\n```\(fence)\n\(file.content)\(file.content.hasSuffix("\n") ? "" : "\n")```"
                               + (file.truncated ? "\n(This file is longer; only the beginning is shown.)" : ""))
            }
            parts.append(current.joined(separator: "\n"))
        }
        if !others.isEmpty {
            parts.append("Other files: " + others.prefix(30).joined(separator: ", "))
        }
        if let notes, !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parts.append("Notes from the user (CDWS.md):\n" + String(notes.prefix(500)))
        }
        if !earlier.isEmpty {
            parts.append("Earlier requests (already reflected in the files above):\n" + earlier.map { "- " + $0.prefix(200) }.joined(separator: "\n"))
        }
        return parts.joined(separator: "\n\n")
    }

    /// モデルに渡す依頼。小さいコード向けモデルは日本語を取り違えやすい（「電卓風」→ 時計 など）ので、
    /// よく使う言葉の英語を添える（実際のモデルで試して、これで電卓を作るようになった）。
    static func userMessage(_ request: String) -> String {
        var text = "Request: " + request
        let words = englishKeywords(request)
        if !words.isEmpty { text += "\n(English: " + words.joined(separator: ", ") + ")" }
        return text + "\n\nWrite the complete new files for this request."
    }

    /// 依頼に出てくる言葉の英語（出てきた順・重なりは長いほうだけ）
    static func englishKeywords(_ request: String) -> [String] {
        var found: [(position: Int, range: Range<String.Index>, english: String)] = []
        for (japanese, english) in glossary.sorted(by: { $0.key.count > $1.key.count }) {
            var searchStart = request.startIndex
            while let range = request.range(of: japanese, range: searchStart..<request.endIndex) {
                searchStart = range.upperBound
                guard !found.contains(where: { $0.range.overlaps(range) }) else { continue }
                found.append((request.distance(from: request.startIndex, to: range.lowerBound), range, english))
            }
        }
        var seen = Set<String>()
        return found.sorted { $0.position < $1.position }.map(\.english).filter { seen.insert($0).inserted }
    }

    /// 画面・ゲーム・道具づくりでよく使う言葉
    static let glossary: [String: String] = [
        "電卓": "calculator", "計算機": "calculator", "計算": "calculate", "足し算": "addition", "引き算": "subtraction",
        "掛け算": "multiplication", "割り算": "division", "時計": "clock", "タイマー": "timer", "ストップウォッチ": "stopwatch",
        "カウンター": "counter", "数える": "count", "回数": "number of times", "カレンダー": "calendar", "日付": "date",
        "時間": "time", "天気": "weather", "メモ帳": "notepad", "メモ": "memo", "やること": "to-do list", "ToDo": "to-do list",
        "TODO": "to-do list", "リスト": "list", "一覧": "list", "表を": "table", "グラフ": "chart", "ゲーム": "game",
        "じゃんけん": "rock-paper-scissors", "クイズ": "quiz", "問題": "question", "答え": "answer", "点数": "score",
        "スコア": "score", "迷路": "maze", "パズル": "puzzle", "テトリス": "falling blocks game", "ブロック": "block",
        "ボール": "ball", "キャラ": "character", "キャラクター": "character", "敵": "enemy", "ジャンプ": "jump",
        "ボタン": "button", "入力欄": "input field", "入力": "input", "テキスト": "text", "文字": "text", "見出し": "heading",
        "画像": "image", "写真": "photo", "アイコン": "icon", "メニュー": "menu", "ナビ": "navigation bar", "ヘッダー": "header",
        "フッター": "footer", "リンク": "link", "フォーム": "form", "送信": "submit", "検索": "search", "ログイン": "login",
        "パスワード": "password", "チェックボックス": "checkbox", "スライダー": "slider", "選択": "select", "画面": "screen",
        "ページ": "page", "サイト": "website", "ホームページ": "website", "アプリ": "app", "背景": "background", "色": "color",
        "文字色": "text color", "赤": "red", "青": "blue", "緑": "green", "黄色": "yellow", "黒": "black",
        "白": "white", "灰色": "gray", "ピンク": "pink", "紫": "purple", "オレンジ": "orange", "水色": "light blue",
        "大きく": "bigger", "大きい": "big", "小さく": "smaller", "小さい": "small", "太く": "bold", "丸く": "rounded",
        "角丸": "rounded corners", "影をつけ": "add a shadow", "枠線": "border", "中央": "center", "真ん中": "center",
        "左に": "on the left", "右に": "on the right", "上に": "at the top", "下に": "at the bottom", "横に": "side by side", "縦に": "vertically", "並べ": "arrange",
        "並べて": "arrange", "アニメーション": "animation", "動く": "moving", "動かす": "move", "回転": "rotate",
        "点滅": "blink", "光る": "glow", "音": "sound", "鳴らす": "play a sound", "クリック": "click", "タップ": "tap",
        "押す": "press", "押したら": "when pressed", "押すと": "when pressed", "表示": "show", "隠す": "hide",
        "消す": "delete", "削除": "delete", "追加": "add", "保存": "save", "読み込み": "load", "リセット": "reset",
        "スタート": "start", "開始": "start", "停止": "stop", "止める": "stop", "一時停止": "pause", "ランダム": "random",
        "数字": "number", "合計": "total", "平均": "average", "風": "-style", "みたい": "like", "っぽく": "-like",
        "かっこよく": "stylish", "おしゃれ": "stylish", "かわいく": "cute", "シンプル": "simple", "ダークモード": "dark mode",
        "直して": "fix", "修正": "fix", "バグ": "bug", "エラー": "error", "動かない": "does not work", "変えて": "change",
        "変更": "change", "作って": "make", "作る": "make", "新しく": "new", "全部": "all",
    ]

    static func fenceLanguage(_ path: String) -> String {
        switch CodeReply.fileExtension(path) {
        case "html", "htm": return "html"
        case "css": return "css"
        case "js", "mjs": return "javascript"
        case "py": return "python"
        case "md": return "markdown"
        case "json": return "json"
        case "swift": return "swift"
        case "ts": return "typescript"
        default: return ""
        }
    }
}
