import Foundation

enum WorkspaceError: Error, LocalizedError, Equatable {
    case outsideWorkspace(String)
    case notFound(String)
    case alreadyExists(String)
    case notText(String)
    case tooLarge(String, Int64)
    case editNotFound(String)
    case editNotUnique(String, Int)
    case invalidArgument(String)

    var errorDescription: String? {
        switch self {
        case .outsideWorkspace(let path):
            return L("プロジェクトの外は触れません: \(path)", "Paths outside the project are not allowed: \(path)")
        case .notFound(let path):
            return L("見つかりません: \(path)", "Not found: \(path)")
        case .alreadyExists(let path):
            return L("既にあります: \(path)", "Already exists: \(path)")
        case .notText(let path):
            return L("テキストのファイルではありません: \(path)", "Not a text file: \(path)")
        case .tooLarge(let path, let size):
            return L("大きすぎて開けません（\(ByteFormat.string(size))）: \(path)", "Too large to open (\(ByteFormat.string(size))): \(path)")
        case .editNotFound(let path):
            return L("\(path) に置き換える文字列が見つかりません。read_file で今の中身を確かめてください。",
                     "The text to replace was not found in \(path). Check the current content with read_file.")
        case .editNotUnique(let path, let count):
            return L("\(path) に置き換える文字列が \(count) か所あります。前後の行を含めて 1 か所に絞るか、replace_all を使ってください。",
                     "The text to replace occurs \(count) times in \(path). Include more surrounding lines or use replace_all.")
        case .invalidArgument(let message):
            return message
        }
    }
}

/// プロジェクトのフォルダの中のファイル・フォルダ。
struct WorkspaceEntry: Hashable, Sendable, Identifiable {
    /// プロジェクトのフォルダからの相対パス（"/" 区切り）
    var path: String
    var isDirectory: Bool
    var size: Int64
    var modified: Date?

    var id: String { path }
    var name: String { (path as NSString).lastPathComponent }
    var depth: Int { path.split(separator: "/").count - 1 }
}

/// ファイルの検索で見つかった行。
struct WorkspaceSearchMatch: Hashable, Sendable {
    var path: String
    var line: Int
    var text: String
}

/// プロジェクトのフォルダの中だけを読み書きする。パスはすべてフォルダからの相対パスで、外に出るパスは拒む。
struct WorkspaceFileSystem: Sendable {
    let root: URL

    /// 一覧や検索で飛ばすフォルダ・ファイル
    static let ignoredNames: Set<String> = [
        ".git", ".build", "node_modules", ".DS_Store", ".swiftpm", "DerivedData", "__pycache__", ".cdws", "xcuserdata",
    ]
    /// 開けるテキストの上限
    static let maxTextBytes: Int64 = 4_000_000

    init(root: URL) {
        self.root = root.standardizedFileURL
    }

    private var fileManager: FileManager { FileManager.default }

    // MARK: - パス

    /// 相対パスを整える（"./a/../b/" → "b"）。プロジェクトの外を指すならエラー。
    func normalize(_ path: String) throws -> String {
        var trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix(root.path + "/") { trimmed = String(trimmed.dropFirst(root.path.count + 1)) }
        if trimmed == root.path { trimmed = "" }
        var parts: [Substring] = []
        for component in trimmed.split(separator: "/", omittingEmptySubsequences: true) {
            if component == "." { continue }
            if component == ".." {
                guard !parts.isEmpty else { throw WorkspaceError.outsideWorkspace(path) }
                parts.removeLast()
                continue
            }
            if component == "~" { throw WorkspaceError.outsideWorkspace(path) }
            parts.append(component)
        }
        return parts.joined(separator: "/")
    }

    func url(for path: String) throws -> URL {
        let normalized = try normalize(path)
        return normalized.isEmpty ? root : root.appendingPathComponent(normalized)
    }

    func exists(_ path: String) -> Bool {
        guard let url = try? url(for: path) else { return false }
        return fileManager.fileExists(atPath: url.path)
    }

    func isDirectory(_ path: String) -> Bool {
        guard let url = try? url(for: path) else { return false }
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    // MARK: - 一覧

    /// フォルダの中身（フォルダが先、名前順）。
    func children(of path: String = "", includeHidden: Bool = false) throws -> [WorkspaceEntry] {
        let base = try normalize(path)
        let folder = try url(for: base)
        guard isDirectory(base) else { throw WorkspaceError.notFound(path) }
        let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey]
        let urls = try fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: keys)
        var entries: [WorkspaceEntry] = []
        for url in urls {
            let name = url.lastPathComponent
            if !includeHidden && (Self.ignoredNames.contains(name) || name.hasPrefix(".")) { continue }
            let values = try? url.resourceValues(forKeys: Set(keys))
            entries.append(WorkspaceEntry(path: base.isEmpty ? name : base + "/" + name,
                                          isDirectory: values?.isDirectory ?? false,
                                          size: Int64(values?.fileSize ?? 0),
                                          modified: values?.contentModificationDate))
        }
        return entries.sorted { lhs, rhs in
            if lhs.isDirectory != rhs.isDirectory { return lhs.isDirectory }
            return lhs.name.localizedStandardCompare(rhs.name) == .orderedAscending
        }
    }

    /// フォルダの中を深さ優先でたどる（上限まで）。
    func walk(_ path: String = "", maxDepth: Int = 12, limit: Int = 5000) throws -> [WorkspaceEntry] {
        var result: [WorkspaceEntry] = []
        func visit(_ folder: String, depth: Int) throws {
            for entry in try children(of: folder) {
                guard result.count < limit else { return }
                result.append(entry)
                if entry.isDirectory && depth < maxDepth {
                    try visit(entry.path, depth: depth + 1)
                }
            }
        }
        try visit(try normalize(path), depth: 0)
        return result
    }

    /// モデルに見せる、字下げしたファイルの木。
    func tree(_ path: String = "", maxDepth: Int = 4, maxEntries: Int = 120) -> String {
        guard let entries = try? walk(path, maxDepth: maxDepth, limit: maxEntries + 1) else { return "" }
        let baseDepth = ((try? normalize(path)) ?? "").split(separator: "/").count
        var lines: [String] = []
        for entry in entries.prefix(maxEntries) {
            let indent = String(repeating: "  ", count: max(0, entry.depth - baseDepth))
            lines.append(indent + entry.name + (entry.isDirectory ? "/" : ""))
        }
        if entries.count > maxEntries { lines.append("…") }
        return lines.joined(separator: "\n")
    }

    // MARK: - 読む

    func read(_ path: String) throws -> String {
        let normalized = try normalize(path)
        let url = try url(for: normalized)
        guard fileManager.fileExists(atPath: url.path), !isDirectory(normalized) else { throw WorkspaceError.notFound(normalized) }
        let size = (try? fileManager.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
        guard size <= Self.maxTextBytes else { throw WorkspaceError.tooLarge(normalized, size) }
        let data = try Data(contentsOf: url)
        guard let text = Self.decodeText(data) else { throw WorkspaceError.notText(normalized) }
        return text
    }

    /// 行番号付きで一部を読む（offset は 1 始まりの行番号）。
    func readNumbered(_ path: String, offset: Int = 1, limit: Int = 400, maxCharacters: Int = 12000) throws -> (text: String, totalLines: Int, shownUntil: Int) {
        let lines = TextLines.split(try read(path))
        let start = max(1, offset)
        guard start <= lines.count else { return ("", lines.count, lines.count) }
        var output = ""
        var last = start - 1
        for number in start...min(lines.count, start + max(1, limit) - 1) {
            let line = "\(number)| " + lines[number - 1] + "\n"
            if output.count + line.count > maxCharacters && number > start { break }
            output += line
            last = number
        }
        return (output, lines.count, last)
    }

    /// UTF-8（BOM 付きも）、または UTF-16 のテキストとして読めればその文字列
    static func decodeText(_ data: Data) -> String? {
        if data.isEmpty { return "" }
        if data.prefix(8192).contains(0) {
            if data.starts(with: [0xFF, 0xFE]) || data.starts(with: [0xFE, 0xFF]) {
                return String(data: data, encoding: .utf16)
            }
            return nil
        }
        if data.starts(with: [0xEF, 0xBB, 0xBF]) {
            return String(data: data.dropFirst(3), encoding: .utf8)
        }
        return String(data: data, encoding: .utf8)
    }

    // MARK: - 書く

    /// ファイルを書く（フォルダが無ければ作る）。変更の記録を返す。
    @discardableResult
    func write(_ path: String, _ content: String) throws -> FileChange {
        let normalized = try normalize(path)
        guard !normalized.isEmpty else { throw WorkspaceError.invalidArgument(L("ファイル名がありません", "Missing file name")) }
        let url = try url(for: normalized)
        if isDirectory(normalized) { throw WorkspaceError.alreadyExists(normalized) }
        let before = fileManager.fileExists(atPath: url.path) ? (try? read(normalized)) : nil
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(content.utf8).write(to: url, options: .atomic)
        return FileChange(path: normalized, kind: before == nil ? .created : .modified, before: before, after: content)
    }

    /// old の文字列を new に置き換える（Claude Code の Edit と同じ考え方。old は 1 か所だけにあること）。
    @discardableResult
    func edit(_ path: String, old: String, new: String, replaceAll: Bool = false) throws -> FileChange {
        let normalized = try normalize(path)
        let (before, after) = try editedContent(normalized, old: old, new: new, replaceAll: replaceAll)
        let url = try url(for: normalized)
        try Data(after.utf8).write(to: url, options: .atomic)
        return FileChange(path: normalized, kind: .modified, before: before, after: after)
    }

    /// 置き換えたあとの中身（書き込まない。確認画面の差分用）
    func editedContent(_ path: String, old: String, new: String, replaceAll: Bool = false) throws -> (before: String, after: String) {
        let normalized = try normalize(path)
        let before = try read(normalized)
        if old.isEmpty {
            // 空のファイルに書き込むときなど
            guard before.isEmpty else { throw WorkspaceError.invalidArgument(L("old_string が空です", "old_string is empty")) }
            return (before, new)
        }
        let count = Self.occurrences(of: old, in: before)
        if count == 0 {
            // 改行コードや行末の空白の違いを許して、もう一度探す
            guard let fuzzy = Self.fuzzyReplace(in: before, old: old, new: new) else {
                throw WorkspaceError.editNotFound(normalized)
            }
            return (before, fuzzy)
        }
        if count > 1 && !replaceAll { throw WorkspaceError.editNotUnique(normalized, count) }
        return (before, before.replacingOccurrences(of: old, with: new))
    }

    static func occurrences(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var searchRange = haystack.startIndex..<haystack.endIndex
        while let found = haystack.range(of: needle, range: searchRange) {
            count += 1
            searchRange = found.upperBound..<haystack.endIndex
        }
        return count
    }

    /// 行末の空白と CRLF を無視して、ちょうど 1 か所に一致すれば置き換える
    static func fuzzyReplace(in text: String, old: String, new: String) -> String? {
        func clean(_ line: String) -> String {
            var scalars = Array(line.unicodeScalars)
            while let last = scalars.last, last == " " || last == "\t" || last == "\r" { scalars.removeLast() }
            return String(String.UnicodeScalarView(scalars))
        }
        let lines = TextLines.split(text)
        let target = TextLines.split(old).map(clean)
        guard !target.isEmpty, target.count <= lines.count else { return nil }
        let cleaned = lines.map(clean)
        var matches: [Int] = []
        for start in 0...(lines.count - target.count) where Array(cleaned[start..<(start + target.count)]) == target {
            matches.append(start)
        }
        guard matches.count == 1, let start = matches.first else { return nil }
        var result = Array(lines[0..<start])
        result.append(new)
        result.append(contentsOf: lines[(start + target.count)...])
        return result.joined(separator: "\n")
    }

    func makeDirectory(_ path: String) throws {
        let url = try url(for: path)
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
    }

    /// 消す。テキストのファイルは元に戻せるように中身を記録する。
    @discardableResult
    func delete(_ path: String) throws -> [FileChange] {
        let normalized = try normalize(path)
        guard !normalized.isEmpty else { throw WorkspaceError.outsideWorkspace(path) }
        let url = try url(for: normalized)
        guard fileManager.fileExists(atPath: url.path) else { throw WorkspaceError.notFound(normalized) }
        var changes: [FileChange] = []
        if isDirectory(normalized) {
            for entry in (try? walk(normalized, limit: 400)) ?? [] where !entry.isDirectory {
                changes.append(FileChange(path: entry.path, kind: .deleted, before: try? read(entry.path), after: nil))
            }
            if changes.isEmpty { changes.append(FileChange(path: normalized, kind: .deleted, before: nil, after: nil)) }
        } else {
            changes.append(FileChange(path: normalized, kind: .deleted, before: try? read(normalized), after: nil))
        }
        try fileManager.removeItem(at: url)
        return changes
    }

    @discardableResult
    func move(_ from: String, to: String) throws -> FileChange {
        let source = try normalize(from)
        let destination = try normalize(to)
        guard !source.isEmpty, !destination.isEmpty else { throw WorkspaceError.outsideWorkspace(source.isEmpty ? from : to) }
        let sourceURL = try url(for: source)
        let destinationURL = try url(for: destination)
        guard fileManager.fileExists(atPath: sourceURL.path) else { throw WorkspaceError.notFound(source) }
        guard !fileManager.fileExists(atPath: destinationURL.path) else { throw WorkspaceError.alreadyExists(destination) }
        try fileManager.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fileManager.moveItem(at: sourceURL, to: destinationURL)
        return FileChange(path: destination, kind: .moved, before: nil, after: nil, movedFrom: source)
    }

    // MARK: - 検索

    /// ファイルの中を探す（正規表現か、そのままの文字列）。
    func search(_ pattern: String, in path: String = "", isRegex: Bool = false, caseInsensitive: Bool = true,
                limit: Int = 60) throws -> [WorkspaceSearchMatch] {
        guard !pattern.isEmpty else { return [] }
        var regex: NSRegularExpression?
        if isRegex {
            regex = try NSRegularExpression(pattern: pattern, options: caseInsensitive ? [.caseInsensitive] : [])
        }
        let options: String.CompareOptions = caseInsensitive ? [.caseInsensitive] : []
        var matches: [WorkspaceSearchMatch] = []
        let files = isDirectory(path) ? try walk(path).filter { !$0.isDirectory } : [WorkspaceEntry(path: try normalize(path), isDirectory: false, size: 0)]
        for file in files where file.size < 1_000_000 {
            guard let text = try? read(file.path) else { continue }
            var number = 0
            for line in TextLines.split(text) {
                number += 1
                let found: Bool
                if let regex {
                    found = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil
                } else {
                    found = line.range(of: pattern, options: options) != nil
                }
                if found {
                    let shown = line.count > 200 ? String(line.prefix(200)) + "…" : line
                    matches.append(WorkspaceSearchMatch(path: file.path, line: number, text: shown))
                    if matches.count >= limit { return matches }
                }
            }
        }
        return matches
    }

    /// 名前で探す（* と ? が使える。"/" を含むならパス全体と比べる）。
    func find(_ pattern: String, limit: Int = 200) throws -> [String] {
        let matchPath = pattern.contains("/")
        return try walk().filter { entry in
            Glob.matches(pattern, matchPath ? entry.path : entry.name)
        }.prefix(limit).map { $0.path + ($0.isDirectory ? "/" : "") }
    }
}

/// 行に分ける。"\r\n" は Swift では 1 文字（Character）になるので、Unicode のスカラーで "\n" を探して分ける。
/// 行末の "\r" は残る（書き戻すときに元の改行コードを保つため）。
enum TextLines {
    static func split(_ text: String) -> [String] {
        text.unicodeScalars.split(separator: "\n", omittingEmptySubsequences: false)
            .map { String(String.UnicodeScalarView($0)) }
    }
}

/// かんたんなワイルドカード（* ? と、"**/" はどの階層でも）。
enum Glob {
    static func matches(_ pattern: String, _ text: String) -> Bool {
        let p = Array(pattern.lowercased().replacingOccurrences(of: "**/", with: "*"))
        let t = Array(text.lowercased())
        var pi = 0, ti = 0
        var star = -1, mark = 0
        while ti < t.count {
            if pi < p.count && (p[pi] == "?" || p[pi] == t[ti]) {
                pi += 1
                ti += 1
            } else if pi < p.count && p[pi] == "*" {
                star = pi
                mark = ti
                pi += 1
            } else if star >= 0 {
                pi = star + 1
                mark += 1
                ti = mark
            } else {
                return false
            }
        }
        while pi < p.count && p[pi] == "*" { pi += 1 }
        return pi == p.count
    }
}
