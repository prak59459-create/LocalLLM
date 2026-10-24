import Foundation

/// 確認の途中経過（画面に出す）。
enum UpdateStage: Equatable, Sendable {
    case contacting
    case listing
    case scanning
    case comparing
    case downloading(done: Int, total: Int, bytes: Int64)

    var message: String {
        switch self {
        case .contacting: return L("GitHub に最新のコミットを問い合わせています…", "Asking GitHub for the latest commit…")
        case .listing: return L("GitHub のファイル一覧を取っています…", "Fetching the file list from GitHub…")
        case .scanning: return L("この iPad のファイルを調べています…", "Scanning the files on this iPad…")
        case .comparing: return L("違うファイルを探しています…", "Looking for changed files…")
        case .downloading(let done, let total, _):
            return L("変わったファイルをダウンロードしています（\(done)/\(total)）…", "Downloading changed files (\(done)/\(total))…")
        }
    }

    var fraction: Double? {
        guard case .downloading(let done, let total, _) = self, total > 0 else { return nil }
        return Double(done) / Double(total)
    }
}

/// 確認の結果。
struct UpdateCheckResult: Sendable {
    var sourceLabel: String
    var repository: String?
    var branch: String?
    var commit: GitHubCommit?
    /// 前回の更新から今回のコミットまでのコミット（新しい順。分からなければ空）
    var commitsSinceBase: [GitHubCommit] = []
    var remote: RemotePackage
    var local: PackageIndex
    var diff: UpdateDiff
    var base: UpdateRecord?
    var localPackageSwift: String
    var remotePackageSwift: String?

    var currentVersion: PackageVersion? { PackageVersion.parse(packageSwift: localPackageSwift) }
    var incomingVersion: PackageVersion? { remotePackageSwift.flatMap(PackageVersion.parse(packageSwift:)) }

    /// 選んだファイルで更新したときの版
    func newVersion() -> PackageVersion {
        PackageVersion.next(current: currentVersion, incoming: incomingVersion)
    }

    /// 選んだファイルで更新する内容
    func request(selection: Set<String>) -> UpdateRequest {
        let chosen = diff.changes.filter { selection.contains($0.path) }
        var writes: [String: String] = [:]
        var deletes: [String] = []
        var expected: [String: String?] = ["Package.swift": local.files["Package.swift"]?.sha]
        for change in chosen {
            expected[change.path] = change.localSHA
            if change.kind == .removed {
                deletes.append(change.path)
            } else if let sha = change.remoteSHA, !change.isPackageManifest {
                writes[change.path] = sha
            }
        }
        let takeManifest = chosen.contains { $0.isPackageManifest }
        return UpdateRequest(writes: writes, deletes: deletes.sorted(), expectedLocal: expected,
                             remotePackageSwift: remotePackageSwift, takeRemotePackageSwift: takeManifest,
                             basePackageSwift: base?.remotePackageSwift, newVersion: newVersion(),
                             manifest: UpdateDiffer.manifest(after: diff, remote: remote.files, base: base?.manifest,
                                                             applied: Set(chosen.map(\.path))),
                             sourceLabel: sourceLabel, repository: repository, branch: branch, commit: commit)
    }

    /// ダウンロードするファイル（GitHub にあって、この iPad と違うもの）
    var downloadItems: [UpdateDownloadItem] {
        diff.changes.compactMap { change in
            guard let sha = change.remoteSHA, change.kind.writesRemoteFile else { return nil }
            return UpdateDownloadItem(path: change.path, sha: sha, size: change.remoteSize ?? 0)
        }
    }
}

enum UpdateCheckError: Error, LocalizedError, Equatable {
    /// iCloud からまだ落ちていない・読めないファイル
    case notAvailableLocally([String])

    var errorDescription: String? {
        switch self {
        case .notAvailableLocally(let paths):
            return L("この iPad にまだダウンロードされていない（または読めない）ファイルがあります: \(paths.prefix(3).joined(separator: ", "))\(paths.count > 3 ? " ほか \(paths.count - 3) 個" : "")。ファイル App で .swiftpm を開いて iCloud からダウンロードしてから、もう一度試してください。",
                     "Some files are not downloaded to this iPad (or cannot be read): \(paths.prefix(3).joined(separator: ", "))\(paths.count > 3 ? " and \(paths.count - 3) more" : ""). Open the .swiftpm in the Files app to download it from iCloud, then try again.")
        }
    }
}

/// 更新の確認（GitHub やファイルと、この iPad のパッケージを比べる）。
enum UpdateEngine {
    /// GitHub のブランチと比べて、変わったファイルだけをダウンロードしておく
    static func checkGitHub(client: GitHubClient, ref: String, target: URL, objects: UpdateObjectStore, base: UpdateRecord?,
                            progress: @escaping @Sendable (UpdateStage) -> Void) async throws -> UpdateCheckResult {
        progress(.contacting)
        let commit = try await client.commit(ref: ref)
        progress(.listing)
        let remote = try await client.package(commit: commit, preferredNames: [target.lastPathComponent, "LocalLLM.swiftpm"])
        progress(.scanning)
        let local = try scanLocal(target)
        let localPackageSwift = try String(contentsOf: target.appendingPathComponent("Package.swift"), encoding: .utf8)

        progress(.comparing)
        var remotePackageSwift: String?
        if let manifest = remote.files["Package.swift"] {
            if manifest.sha == local.files["Package.swift"]?.sha {
                remotePackageSwift = localPackageSwift
            } else {
                let data = try await client.file(path: remote.prefix + "Package.swift", commit: commit.sha, expectedSHA: manifest.sha)
                try objects.store(data, sha: manifest.sha)
                remotePackageSwift = String(decoding: data, as: UTF8.self)
            }
        }
        var result = compare(local: local, remote: remote, localPackageSwift: localPackageSwift,
                             remotePackageSwift: remotePackageSwift, base: base,
                             sourceLabel: "GitHub \(client.repository) (\(ref))")
        result.repository = client.repository.description
        result.branch = ref
        result.commit = commit
        if let baseSHA = base?.commit?.sha, baseSHA != commit.sha,
           let comparison = try? await client.compare(base: baseSHA, head: commit.sha) {
            result.commitsSinceBase = comparison.commits
        }

        // 変わったファイルだけを取ってくる（中身は GitHub の一覧の識別子で確かめる）
        let prefix = remote.prefix
        let items = result.downloadItems.map { UpdateDownloadItem(path: prefix + $0.path, sha: $0.sha, size: $0.size) }
        try await UpdateDownloader.download(items, commit: commit.sha, client: client, objects: objects) { done, total, bytes in
            progress(.downloading(done: done, total: total, bytes: bytes))
        }
        return result
    }

    /// 選んだフォルダ（ZIP を展開したもの・.swiftpm）と比べて、変わったファイルを objects に置いておく
    static func checkFolder(_ source: URL, label: String, target: URL, objects: UpdateObjectStore,
                            base: UpdateRecord?) throws -> UpdateCheckResult {
        let incoming = PackageScanner.index(of: source)
        let remote = RemotePackage(prefix: "", files: incoming.files)
        let local = try scanLocal(target)
        let localPackageSwift = try String(contentsOf: target.appendingPathComponent("Package.swift"), encoding: .utf8)
        let remotePackageSwift = try? String(contentsOf: source.appendingPathComponent("Package.swift"), encoding: .utf8)
        let result = compare(local: local, remote: remote, localPackageSwift: localPackageSwift,
                             remotePackageSwift: remotePackageSwift, base: base, sourceLabel: label)
        for item in result.downloadItems where !objects.contains(item.sha) {
            try objects.store(try Data(contentsOf: source.appendingPathComponent(item.path)), sha: item.sha)
        }
        return result
    }

    /// この iPad のパッケージを読む（読めないファイルがあれば止める）
    static func scanLocal(_ target: URL) throws -> PackageIndex {
        guard PackageScanner.isPackage(target) else { throw AppUpdateError.notAPackage(target.lastPathComponent) }
        let local = PackageScanner.index(of: target)
        let missing = local.placeholders + local.unreadable
        if !missing.isEmpty { throw UpdateCheckError.notAvailableLocally(missing.sorted()) }
        return local
    }

    static func compare(local: PackageIndex, remote: RemotePackage, localPackageSwift: String, remotePackageSwift: String?,
                        base: UpdateRecord?, sourceLabel: String) -> UpdateCheckResult {
        var equivalent = false
        if let remotePackageSwift {
            equivalent = PackageManifest.isEquivalent(localPackageSwift, remotePackageSwift)
            // GitHub の版の方が新しければ、版を上げるために Package.swift も更新に含める
            if equivalent, let incoming = PackageVersion.parse(packageSwift: remotePackageSwift),
               let current = PackageVersion.parse(packageSwift: localPackageSwift),
               PackageVersion.compare(incoming.display, current.display) == .orderedDescending {
                equivalent = false
            }
        }
        let diff = UpdateDiffer.diff(local: local.files, remote: remote.files, base: base?.manifest, manifestEquivalent: equivalent)
        return UpdateCheckResult(sourceLabel: sourceLabel, remote: remote, local: local, diff: diff, base: base,
                                 localPackageSwift: localPackageSwift, remotePackageSwift: remotePackageSwift)
    }
}
