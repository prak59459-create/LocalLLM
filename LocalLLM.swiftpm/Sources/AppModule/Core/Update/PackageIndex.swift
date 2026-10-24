import Foundation

/// パッケージの中のファイル 1 つ（git の識別子と大きさ）。
struct PackageFile: Codable, Hashable, Sendable {
    var sha: String
    var size: Int64
}

/// パッケージの中のファイルの一覧（相対パス → 識別子）。
struct PackageIndex: Sendable {
    var files: [String: PackageFile]
    /// iCloud にあって、まだこの端末に落ちていないファイル（本当の相対パス）
    var placeholders: [String] = []
    /// 読めなかったファイル
    var unreadable: [String] = []

    var manifest: [String: String] { files.mapValues(\.sha) }
}

/// .swiftpm の中身を読む。
enum PackageScanner {
    /// コピーしないもの（Playgrounds の設定や、ほかの環境のもの）
    static let ignoredNames: Set<String> = [".git", ".swiftpm", ".build", "xcuserdata", ".DS_Store", "__MACOSX", ".github",
                                            ".localllm-update"]

    static func isPackage(_ url: URL) -> Bool {
        let fileManager = FileManager.default
        var isFolder: ObjCBool = false
        guard fileManager.fileExists(atPath: url.appendingPathComponent("Sources").path, isDirectory: &isFolder), isFolder.boolValue else {
            return false
        }
        return fileManager.fileExists(atPath: url.appendingPathComponent("Package.swift").path)
            || fileManager.fileExists(atPath: url.appendingPathComponent(".Package.swift.icloud").path)
    }

    /// 無視する場所か（どこかの階層が ignoredNames に入っている）
    static func isIgnored(_ relativePath: String) -> Bool {
        relativePath.split(separator: "/").contains { ignoredNames.contains(String($0)) }
    }

    /// パッケージの外に出たり、おかしな名前になったりしない相対パスか
    static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\u{0}") else { return false }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    /// iCloud のまだ落ちていないファイル（".Name.swift.icloud"）の本当の名前
    static func iCloudPlaceholderName(_ name: String) -> String? {
        guard name.hasPrefix("."), name.hasSuffix(".icloud"), name.count > ".icloud".count + 1 else { return nil }
        let inner = String(name.dropFirst().dropLast(".icloud".count))
        return inner.isEmpty ? nil : inner
    }

    /// パッケージのファイルの相対パス（ignoredNames は除く）
    static func relativePaths(in root: URL) -> (files: [String], placeholders: [String]) {
        var files: [String] = []
        var placeholders: [String] = []
        let fileManager = FileManager.default
        func visit(_ folder: URL, prefix: String) {
            guard let items = try? fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey]) else { return }
            for item in items {
                let name = item.lastPathComponent
                if let real = iCloudPlaceholderName(name) {
                    if !ignoredNames.contains(real) { placeholders.append(prefix + real) }
                    continue
                }
                guard !ignoredNames.contains(name) else { continue }
                let values = try? item.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                if values?.isSymbolicLink == true { continue }
                if values?.isDirectory == true {
                    visit(item, prefix: prefix + name + "/")
                } else {
                    files.append(prefix + name)
                }
            }
        }
        visit(root, prefix: "")
        return (files.sorted(), placeholders.sorted())
    }

    /// 全部のファイルの git の識別子を計算する
    static func index(of root: URL) -> PackageIndex {
        let (paths, placeholders) = relativePaths(in: root)
        var files: [String: PackageFile] = [:]
        var unreadable: [String] = []
        for path in paths {
            guard let data = try? Data(contentsOf: root.appendingPathComponent(path), options: .mappedIfSafe) else {
                unreadable.append(path)
                continue
            }
            files[path] = PackageFile(sha: GitBlob.sha(of: data), size: Int64(data.count))
        }
        return PackageIndex(files: files, placeholders: placeholders.filter { files[$0] == nil }, unreadable: unreadable)
    }

    /// パスの最後の名前（"a/B.swiftpm/" → "B.swiftpm"、"" → ""）
    static func lastComponent(_ path: String) -> String {
        guard let last = path.split(separator: "/").last else { return "" }
        return String(last)
    }

    /// パッケージの候補から 1 つ選ぶ: preferredNames の名前 → いちばん浅い .swiftpm → いちばん浅いもの
    static func pickPreferred(_ candidates: [String], preferredNames: [String]) -> String? {
        for wanted in preferredNames {
            if let match = candidates.first(where: { lastComponent($0) == wanted }) { return match }
        }
        let swiftpm = candidates.filter { lastComponent($0).hasSuffix(".swiftpm") }
        return shortest(swiftpm) ?? shortest(candidates)
    }

    private static func shortest(_ items: [String]) -> String? {
        var best: String?
        for item in items {
            if let current = best, current.count <= item.count { continue }
            best = item
        }
        return best
    }

    /// 1 つのファイルの識別子（無ければ nil）
    static func sha(of path: String, in root: URL) -> String? {
        (try? Data(contentsOf: root.appendingPathComponent(path), options: .mappedIfSafe)).map(GitBlob.sha(of:))
    }
}

/// GitHub のファイル一覧の中の、アプリのパッケージ（.swiftpm）。
struct RemotePackage: Sendable {
    /// リポジトリの中での場所（"LocalLLM.swiftpm/"。リポジトリそのものがパッケージなら ""）
    var prefix: String
    var files: [String: PackageFile]

    var manifest: [String: String] { files.mapValues(\.sha) }
    var name: String { PackageScanner.lastComponent(prefix) }

    /// ファイル一覧からパッケージを探す。preferredNames の名前のものを優先する。
    static func locate(in tree: GitHubTree, preferredNames: [String]) -> RemotePackage? {
        var blobs: [GitHubTreeEntry] = []
        var folders = Set<String>()
        for entry in tree.entries {
            if entry.isFile { blobs.append(entry) } else { folders.insert(entry.path) }
        }
        var prefixes: [String] = []
        for entry in blobs where entry.path == "Package.swift" || entry.path.hasSuffix("/Package.swift") {
            let prefix = String(entry.path.dropLast("Package.swift".count))
            if PackageScanner.isIgnored(prefix) { continue }
            let sources = prefix + "Sources"
            let sourcesPrefix = sources + "/"
            if folders.contains(sources) || blobs.contains(where: { $0.path.hasPrefix(sourcesPrefix) }) {
                prefixes.append(prefix)
            }
        }
        guard let prefix = PackageScanner.pickPreferred(prefixes, preferredNames: preferredNames) else { return nil }
        return make(prefix: prefix, entries: blobs)
    }

    /// prefix の下のファイルだけを取り出す
    static func make(prefix: String, entries: [GitHubTreeEntry]) -> RemotePackage {
        var files: [String: PackageFile] = [:]
        for entry in entries where entry.isFile && entry.path.hasPrefix(prefix) {
            let relative = String(entry.path.dropFirst(prefix.count))
            guard PackageScanner.isSafeRelativePath(relative), !PackageScanner.isIgnored(relative) else { continue }
            files[relative] = PackageFile(sha: entry.sha, size: entry.size)
        }
        return RemotePackage(prefix: prefix, files: files)
    }
}

/// 「ファイル」で選んだものから、更新するパッケージを見つける。
enum UpdateTargetResolver {
    enum Resolution: Equatable {
        /// 選んだものがパッケージそのもの
        case package
        /// 選んだフォルダの中にパッケージがある（選んだフォルダからの相対パス）
        case candidates([String])
        /// パッケージの中のファイル・フォルダを選んだ（パッケージ全体を書き換える許可がない）
        case insidePackage(String)
        case notFound
    }

    static func resolve(_ url: URL) -> Resolution {
        if PackageScanner.isPackage(url) { return .package }
        var isFolder: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder)
        if exists, isFolder.boolValue {
            let found = packages(in: url, depth: 3)
            if !found.isEmpty { return .candidates(found) }
        }
        var parent = url.deletingLastPathComponent()
        for _ in 0..<8 {
            if PackageScanner.isPackage(parent) { return .insidePackage(parent.lastPathComponent) }
            let next = parent.deletingLastPathComponent()
            if next.path == parent.path { break }
            parent = next
        }
        return .notFound
    }

    /// フォルダの中のパッケージ（相対パス）。"LocalLLM" で始まるものを先に並べる。
    static func packages(in folder: URL, depth: Int) -> [String] {
        var found: [String] = []
        func visit(_ current: URL, prefix: String, level: Int) {
            guard level < depth,
                  let items = try? FileManager.default.contentsOfDirectory(at: current, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
            for item in items where !PackageScanner.ignoredNames.contains(item.lastPathComponent) && !item.lastPathComponent.hasPrefix(".") {
                guard (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
                let path = prefix + item.lastPathComponent
                if PackageScanner.isPackage(item) {
                    found.append(path)
                } else {
                    visit(item, prefix: path + "/", level: level + 1)
                }
            }
        }
        visit(folder, prefix: "", level: 0)
        func isThisApp(_ path: String) -> Bool { path.split(separator: "/").last?.hasPrefix("LocalLLM") ?? false }
        return found.sorted { lhs, rhs in
            let a = isThisApp(lhs), b = isThisApp(rhs)
            return a != b ? a : lhs.localizedStandardCompare(rhs) == .orderedAscending
        }
    }

    /// 展開した ZIP などの中から、アプリのパッケージを探す（preferredNames の名前を優先する）
    static func findPackage(in folder: URL, preferredNames: [String]) -> URL? {
        if PackageScanner.isPackage(folder) { return folder }
        let found = packages(in: folder, depth: 4)
        guard let chosen = PackageScanner.pickPreferred(found, preferredNames: preferredNames) else { return nil }
        return folder.appendingPathComponent(chosen, isDirectory: true)
    }
}

/// 選んだパッケージの情報（画面に出す）。
struct PackageSummary: Equatable, Sendable {
    var folderName: String
    var appName: String?
    var version: PackageVersion?
    var swiftFileCount: Int
    var location: String
    /// このアプリ（LocalLLM）のパッケージらしいか
    var looksLikeThisApp: Bool

    static func read(_ root: URL) -> PackageSummary {
        let text = (try? String(contentsOf: root.appendingPathComponent("Package.swift"), encoding: .utf8)) ?? ""
        let paths = PackageScanner.relativePaths(in: root)
        let all = paths.files + paths.placeholders
        let swiftCount = all.filter { $0.hasPrefix("Sources/") && $0.hasSuffix(".swift") }.count
        let appName = PackageManifest.stringValue(of: "name", in: text)
        let hasEntry = all.contains("Sources/AppModule/App/LocalLLMApp.swift")
        return PackageSummary(folderName: root.lastPathComponent, appName: appName,
                              version: PackageVersion.parse(packageSwift: text), swiftFileCount: swiftCount,
                              location: locationDescription(root), looksLikeThisApp: hasEntry || appName == "LocalLLM")
    }

    /// 「iCloud Drive › Playgrounds › LocalLLM.swiftpm」のような場所の説明
    static func locationDescription(_ url: URL) -> String {
        let parts = url.standardizedFileURL.pathComponents
        func tail(after index: Int, root: String) -> String {
            ([root] + parts[(index + 1)...]).joined(separator: " › ")
        }
        if let index = parts.firstIndex(where: { $0.hasPrefix("iCloud~") && $0.contains("Playgrounds") }) {
            let rest = parts.indices.contains(index + 1) && parts[index + 1] == "Documents" ? index + 1 : index
            return tail(after: rest, root: "iCloud Drive › Playgrounds")
        }
        if let index = parts.firstIndex(of: "com~apple~CloudDocs") {
            return tail(after: index, root: "iCloud Drive")
        }
        if let index = parts.firstIndex(of: "File Provider Storage") {
            return tail(after: index, root: L("このiPad内", "On My iPad"))
        }
        return parts.suffix(3).joined(separator: " › ")
    }
}
