import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Package.swift に書かれたアプリの版（displayVersion と bundleVersion）。
struct PackageVersion: Codable, Equatable, Sendable, CustomStringConvertible {
    var display: String
    var bundle: Int

    var description: String { "\(display) (\(bundle))" }

    /// Package.swift の文字列から読む
    static func parse(packageSwift text: String) -> PackageVersion? {
        guard let display = value(of: "displayVersion", in: text) else { return nil }
        let bundle = value(of: "bundleVersion", in: text).flatMap { Int($0) } ?? 1
        return PackageVersion(display: display, bundle: bundle)
    }

    private static func value(of key: String, in text: String) -> String? {
        guard let keyRange = text.range(of: key + ":") else { return nil }
        let rest = text[keyRange.upperBound...]
        guard let open = rest.firstIndex(of: "\"") else { return nil }
        let afterOpen = rest.index(after: open)
        guard let close = rest[afterOpen...].firstIndex(of: "\"") else { return nil }
        return String(rest[afterOpen..<close])
    }

    /// "1.2.10" と "1.2.9" のような版の比べ方（数字ごとに比べる）
    static func compare(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let a = lhs.split(separator: ".").map { Int($0.filter(\.isNumber)) ?? 0 }
        let b = rhs.split(separator: ".").map { Int($0.filter(\.isNumber)) ?? 0 }
        for index in 0..<max(a.count, b.count) {
            let x = index < a.count ? a[index] : 0
            let y = index < b.count ? b[index] : 0
            if x != y { return x < y ? .orderedAscending : .orderedDescending }
        }
        return .orderedSame
    }

    /// 最後の数字を 1 つ上げる（"0.2" → "0.2.1"、"0.2.1" → "0.2.2"）
    static func bumpPatch(_ version: String) -> String {
        var parts = version.split(separator: ".").map(String.init)
        if parts.count < 3 {
            while parts.count < 2 { parts.append("0") }
            parts.append("1")
        } else if let last = Int(parts[parts.count - 1].filter(\.isNumber)) {
            parts[parts.count - 1] = String(last + 1)
        } else {
            parts.append("1")
        }
        return parts.joined(separator: ".")
    }

    /// 更新したあとの版。bundleVersion は毎回 1 つ上げる。
    /// 新しいソースの displayVersion の方が大きければそれを使い、そうでなければ今の版の最後の数字を上げる。
    static func next(current: PackageVersion?, incoming: PackageVersion?) -> PackageVersion {
        let bundle = max(current?.bundle ?? 0, incoming?.bundle ?? 0) + 1
        let display: String
        switch (current, incoming) {
        case (let current?, let incoming?):
            display = compare(incoming.display, current.display) == .orderedDescending ? incoming.display : bumpPatch(current.display)
        case (let current?, nil):
            display = bumpPatch(current.display)
        case (nil, let incoming?):
            display = incoming.display
        case (nil, nil):
            display = "1.0"
        }
        return PackageVersion(display: display, bundle: bundle)
    }

    /// Package.swift の版を書き換えた文字列
    func apply(to text: String) -> String {
        var result = Self.replace("displayVersion", with: display, in: text)
        result = Self.replace("bundleVersion", with: String(bundle), in: result)
        return result
    }

    private static func replace(_ key: String, with value: String, in text: String) -> String {
        guard let keyRange = text.range(of: key + ":"),
              let open = text[keyRange.upperBound...].firstIndex(of: "\"") else { return text }
        let afterOpen = text.index(after: open)
        guard let close = text[afterOpen...].firstIndex(of: "\"") else { return text }
        return text.replacingCharacters(in: afterOpen..<close, with: value)
    }
}

/// 更新で行うこと。
struct UpdatePlan: Codable, Sendable {
    var added: [String]
    var changed: [String]
    var removed: [String]
    var unchangedCount: Int
    var previousVersion: PackageVersion?
    var newVersion: PackageVersion

    var isEmpty: Bool { added.isEmpty && changed.isEmpty && removed.isEmpty }
}

enum AppUpdateError: Error, LocalizedError {
    case packageNotFound
    case notAPackage(String)
    case download(Int)

    var errorDescription: String? {
        switch self {
        case .packageNotFound:
            return L("新しいソースの中に、アプリのパッケージ（Package.swift と Sources/ のあるフォルダ）が見つかりません。",
                     "No app package (a folder with Package.swift and Sources/) was found in the new source.")
        case .notAPackage(let name):
            return L("「\(name)」は Swift Playgrounds のアプリ（Package.swift のある .swiftpm）ではないようです。",
                     "\"\(name)\" does not look like a Swift Playgrounds app (.swiftpm with a Package.swift).")
        case .download(let status):
            return L("ダウンロードに失敗しました（HTTP \(status)）。リポジトリ名・ブランチ・トークンを確かめてください。",
                     "Download failed (HTTP \(status)). Check the repository, branch and token.")
        }
    }
}

/// Swift Playgrounds のアプリ（.swiftpm）を、新しいソースで上書きする。
enum AppUpdater {
    /// コピーしないもの（Playgrounds の設定や、ほかの環境のもの）
    static let ignoredNames: Set<String> = [".git", ".swiftpm", ".build", "xcuserdata", ".DS_Store", "__MACOSX", ".github"]
    /// 新しいソースに無ければ消す範囲（これ以外のファイルは消さない）
    static let managedFolders = ["Sources/"]

    /// フォルダの中から、アプリのパッケージのフォルダを探す（preferredName を優先）
    static func packageRoot(in folder: URL, preferredName: String = "LocalLLM.swiftpm") -> URL? {
        let fileManager = FileManager.default
        func isPackage(_ url: URL) -> Bool {
            fileManager.fileExists(atPath: url.appendingPathComponent("Package.swift").path)
                && fileManager.fileExists(atPath: url.appendingPathComponent("Sources").path)
        }
        if isPackage(folder) { return folder }
        var candidates: [URL] = []
        var queue: [(URL, Int)] = [(folder, 0)]
        while !queue.isEmpty {
            let (current, depth) = queue.removeFirst()
            guard depth < 4, let items = try? fileManager.contentsOfDirectory(at: current, includingPropertiesForKeys: [.isDirectoryKey]) else { continue }
            for item in items where !ignoredNames.contains(item.lastPathComponent) {
                guard (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
                if isPackage(item) { candidates.append(item) } else { queue.append((item, depth + 1)) }
            }
        }
        return candidates.first { $0.lastPathComponent == preferredName }
            ?? candidates.first { $0.pathExtension == "swiftpm" }
            ?? candidates.first
    }

    /// パッケージの中のファイル（相対パス → URL）
    static func files(in root: URL) -> [String: URL] {
        var result: [String: URL] = [:]
        let fileManager = FileManager.default
        func visit(_ folder: URL, prefix: String) {
            guard let items = try? fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
            for item in items where !ignoredNames.contains(item.lastPathComponent) {
                let path = prefix + item.lastPathComponent
                if (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                    visit(item, prefix: path + "/")
                } else {
                    result[path] = item
                }
            }
        }
        visit(root, prefix: "")
        return result
    }

    /// 何を足して、何を変えて、何を消すかを調べる（まだ書き込まない）
    static func plan(source: URL, target: URL) throws -> UpdatePlan {
        let targetPackage = target.appendingPathComponent("Package.swift")
        guard FileManager.default.fileExists(atPath: targetPackage.path) else {
            throw AppUpdateError.notAPackage(target.lastPathComponent)
        }
        let sourceFiles = files(in: source)
        let targetFiles = files(in: target)
        guard sourceFiles["Package.swift"] != nil else { throw AppUpdateError.packageNotFound }

        let current = (try? String(contentsOf: targetPackage, encoding: .utf8)).flatMap(PackageVersion.parse(packageSwift:))
        let incoming = sourceFiles["Package.swift"].flatMap { try? String(contentsOf: $0, encoding: .utf8) }
            .flatMap(PackageVersion.parse(packageSwift:))
        let next = PackageVersion.next(current: current, incoming: incoming)

        var added: [String] = [], changed: [String] = [], unchanged = 0
        for (path, url) in sourceFiles {
            guard let existing = targetFiles[path] else {
                added.append(path)
                continue
            }
            if path == "Package.swift" {
                // 版は必ず変わるので、いつも書き直す
                changed.append(path)
                continue
            }
            if (try? Data(contentsOf: url)) == (try? Data(contentsOf: existing)) {
                unchanged += 1
            } else {
                changed.append(path)
            }
        }
        let removed = targetFiles.keys.filter { path in
            sourceFiles[path] == nil && managedFolders.contains { path.hasPrefix($0) }
        }
        return UpdatePlan(added: added.sorted(), changed: changed.sorted(), removed: removed.sorted(),
                          unchangedCount: unchanged, previousVersion: current, newVersion: next)
    }

    /// 更新を書き込む。backup を渡すと、変える前のファイルをそこに残す（restore で戻せる）。
    static func apply(_ plan: UpdatePlan, source: URL, target: URL, backup: URL?) throws {
        let fileManager = FileManager.default
        if let backup {
            try? fileManager.removeItem(at: backup)
            try fileManager.createDirectory(at: backup.appendingPathComponent("files"), withIntermediateDirectories: true)
            for path in plan.changed + plan.removed {
                let original = target.appendingPathComponent(path)
                let copy = backup.appendingPathComponent("files").appendingPathComponent(path)
                try fileManager.createDirectory(at: copy.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fileManager.copyItem(at: original, to: copy)
            }
            try JSONEncoder().encode(plan).write(to: backup.appendingPathComponent("plan.json"))
        }
        for path in plan.added + plan.changed {
            let from = source.appendingPathComponent(path)
            let to = target.appendingPathComponent(path)
            try fileManager.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            var data = try Data(contentsOf: from)
            if path == "Package.swift", let text = String(data: data, encoding: .utf8) {
                data = Data(plan.newVersion.apply(to: text).utf8)
            }
            try data.write(to: to, options: .atomic)
        }
        for path in plan.removed {
            try? fileManager.removeItem(at: target.appendingPathComponent(path))
        }
        removeEmptyFolders(in: target.appendingPathComponent("Sources"))
    }

    /// 前回の更新を取り消す
    static func restore(from backup: URL, target: URL) throws {
        let fileManager = FileManager.default
        let plan = try JSONDecoder().decode(UpdatePlan.self, from: Data(contentsOf: backup.appendingPathComponent("plan.json")))
        for path in plan.added {
            try? fileManager.removeItem(at: target.appendingPathComponent(path))
        }
        for path in plan.changed + plan.removed {
            let copy = backup.appendingPathComponent("files").appendingPathComponent(path)
            guard fileManager.fileExists(atPath: copy.path) else { continue }
            let to = target.appendingPathComponent(path)
            try fileManager.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fileManager.removeItem(at: to)
            try fileManager.copyItem(at: copy, to: to)
        }
        removeEmptyFolders(in: target.appendingPathComponent("Sources"))
    }

    private static func removeEmptyFolders(in folder: URL) {
        let fileManager = FileManager.default
        guard let items = try? fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        for item in items where (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
            removeEmptyFolders(in: item)
            if (try? fileManager.contentsOfDirectory(atPath: item.path))?.isEmpty == true {
                try? fileManager.removeItem(at: item)
            }
        }
    }

    /// GitHub のリポジトリを ZIP で取ってくる URL（トークンがあれば API 経由で、非公開のリポジトリも取れる）
    static func githubArchiveRequest(repository: String, ref: String, token: String?) -> URLRequest? {
        let trimmed = repository.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "https://github.com/", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let parts = trimmed.split(separator: "/")
        guard parts.count >= 2 else { return nil }
        let owner = parts[0], repo = parts[1].replacingOccurrences(of: ".git", with: "")
        let branch = ref.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "main" : ref.trimmingCharacters(in: .whitespacesAndNewlines)
        var request: URLRequest
        if let token, !token.isEmpty {
            guard let url = URL(string: "https://api.github.com/repos/\(owner)/\(repo)/zipball/\(branch)") else { return nil }
            request = URLRequest(url: url)
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        } else {
            let encoded = branch.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? branch
            guard let url = URL(string: "https://codeload.github.com/\(owner)/\(repo)/zip/refs/heads/\(encoded)") else { return nil }
            request = URLRequest(url: url)
        }
        request.setValue("LocalLLM-Updater", forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 120
        return request
    }
}
