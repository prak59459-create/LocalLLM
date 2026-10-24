import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// GitHub の回数制限の残り。
struct GitHubRateLimit: Equatable, Sendable {
    var limit: Int
    var remaining: Int
    var reset: Date?
}

/// 2 つのコミットの間のコミット（新しい順）。
struct GitHubComparison: Sendable {
    var aheadBy: Int
    var behindBy: Int
    var commits: [GitHubCommit]
}

/// GitHub からアプリのソースを取ってくる（API とファイルのダウンロード）。
struct GitHubClient: Sendable {
    var repository: GitHubRepository
    var token: String?
    var session: URLSession = .shared

    init(repository: GitHubRepository, token: String?, session: URLSession = .shared) {
        self.repository = repository
        let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.token = (trimmed?.isEmpty ?? true) ? nil : trimmed
        self.session = session
    }

    /// ブランチ（またはタグ・コミット）の最新のコミット。etag を渡すと、変わっていなければ nil を返す（回数制限に数えられない）。
    func latestCommit(ref: String, etag: String? = nil) async throws -> (commit: GitHubCommit?, etag: String?) {
        var request = GitHubAPI.request(GitHubAPI.commitURL(repository, ref: ref), token: token)
        if let etag { request.setValue(etag, forHTTPHeaderField: "If-None-Match") }
        let (data, response) = try await send(request, allowNotModified: true)
        let newTag = response.value(forHTTPHeaderField: "ETag")
        if response.statusCode == 304 { return (nil, etag) }
        return (try GitHubCommit.parse(data), newTag)
    }

    func commit(ref: String) async throws -> GitHubCommit {
        guard let commit = try await latestCommit(ref: ref).commit else { throw GitHubAPIError.unexpected }
        return commit
    }

    /// コミットのファイル一覧（全部の階層）
    func tree(sha: String, recursive: Bool = true) async throws -> GitHubTree {
        let (data, _) = try await send(GitHubAPI.request(GitHubAPI.treeURL(repository, sha: sha, recursive: recursive), token: token))
        return try GitHubTree.parse(data)
    }

    /// アプリのパッケージのファイル一覧。一覧が大きすぎて途中で切られたら、パッケージのフォルダだけを取り直す。
    func package(commit: GitHubCommit, preferredNames: [String]) async throws -> RemotePackage {
        let root = commit.treeSHA ?? commit.sha
        let tree = try await tree(sha: root)
        if !tree.truncated {
            guard let package = RemotePackage.locate(in: tree, preferredNames: preferredNames) else { throw AppUpdateError.packageNotFound }
            return package
        }
        // 切られた一覧から探せなければ、1 階層ずつたどる
        if let located = RemotePackage.locate(in: tree, preferredNames: preferredNames),
           let folder = tree.entries.first(where: { !$0.isFile && $0.path + "/" == located.prefix }) {
            let sub = try await self.tree(sha: folder.sha)
            let entries = sub.entries.map { entry in
                GitHubTreeEntry(path: located.prefix + entry.path, sha: entry.sha, size: entry.size, isFile: entry.isFile)
            }
            return RemotePackage.make(prefix: located.prefix, entries: entries)
        }
        let top = try await self.tree(sha: root, recursive: false)
        for name in preferredNames + ["LocalLLM.swiftpm"] {
            guard let folder = top.entries.first(where: { !$0.isFile && $0.path == name }) else { continue }
            let sub = try await self.tree(sha: folder.sha)
            return RemotePackage.make(prefix: name + "/", entries: sub.entries)
        }
        throw AppUpdateError.packageNotFound
    }

    func branches() async throws -> [String] {
        let (data, _) = try await send(GitHubAPI.request(GitHubAPI.branchesURL(repository), token: token))
        return GitHubAPI.parseBranches(data)
    }

    /// base から head までのコミット（新しい順）
    func compare(base: String, head: String) async throws -> GitHubComparison {
        let url = URL(string: "https://api.github.com/repos/\(repository.owner)/\(repository.name)/compare/\(base)...\(head)")!
        let (data, _) = try await send(GitHubAPI.request(url, token: token))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw GitHubAPIError.unexpected }
        let items = object["commits"] as? [[String: Any]] ?? []
        let commits = items.compactMap { item -> GitHubCommit? in
            guard let itemData = try? JSONSerialization.data(withJSONObject: item) else { return nil }
            return try? GitHubCommit.parse(itemData)
        }
        return GitHubComparison(aheadBy: object["ahead_by"] as? Int ?? commits.count, behindBy: object["behind_by"] as? Int ?? 0,
                                commits: commits.reversed())
    }

    /// 回数制限の残り（この問い合わせは回数に数えられない）
    func rateLimit() async throws -> GitHubRateLimit {
        let url = URL(string: "https://api.github.com/rate_limit")!
        let (data, _) = try await send(GitHubAPI.request(url, token: token))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let core = (object["resources"] as? [String: Any])?["core"] as? [String: Any] else { throw GitHubAPIError.unexpected }
        return GitHubRateLimit(limit: core["limit"] as? Int ?? 0, remaining: core["remaining"] as? Int ?? 0,
                               reset: (core["reset"] as? Double).map { Date(timeIntervalSince1970: $0) })
    }

    /// ファイルの中身を取ってきて、GitHub の一覧の識別子と合っているか確かめる（合わなければやり直す）
    func file(path: String, commit: String, expectedSHA: String, attempts: Int = 3) async throws -> Data {
        var lastError: Error = GitHubAPIError.integrity(path)
        for attempt in 0..<max(1, attempts) {
            try Task.checkCancellation()
            do {
                let (data, _) = try await send(GitHubAPI.fileRequest(repository, commit: commit, path: path, token: token))
                if GitBlob.sha(of: data) == expectedSHA { return data }
                lastError = GitHubAPIError.integrity(path)
            } catch let error as GitHubAPIError {
                // 見つからない・権限がない・回数制限はやり直しても同じ
                switch error {
                case .notFound, .unauthorized, .rateLimited: throw error
                default: lastError = error
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
            }
            if attempt + 1 < attempts { try await Task.sleep(nanoseconds: UInt64(500_000_000 * (attempt + 1))) }
        }
        throw lastError
    }

    private func send(_ request: URLRequest, allowNotModified: Bool = false) async throws -> (Data, HTTPURLResponse) {
        var request = request
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            if (error as? URLError)?.code == .cancelled { throw CancellationError() }
            throw error
        }
        guard let http = response as? HTTPURLResponse else { throw GitHubAPIError.unexpected }
        if allowNotModified, http.statusCode == 304 { return (data, http) }
        try GitHubAPI.check(status: http.statusCode, headers: http.allHeaderFields, body: data)
        return (data, http)
    }
}

/// 取ってくるファイル 1 つ。
struct UpdateDownloadItem: Sendable, Hashable {
    var path: String
    var sha: String
    var size: Int64
}

/// 変わったファイルだけを、いくつか同時にダウンロードする。
enum UpdateDownloader {
    /// まだ置いていないファイルを取ってきて objects に置く。progress は（終わった数, 全部の数, 取ってきたバイト数）。
    static func download(_ items: [UpdateDownloadItem], commit: String, client: GitHubClient, objects: UpdateObjectStore,
                         concurrency: Int = 6, progress: @escaping @Sendable (Int, Int, Int64) -> Void) async throws {
        let unique = Array(Dictionary(items.map { ($0.sha, $0) }, uniquingKeysWith: { first, _ in first }).values)
        let needed = unique.filter { !objects.contains($0.sha) }
        let total = needed.count
        progress(0, total, 0)
        guard total > 0 else { return }
        try await withThrowingTaskGroup(of: Int64.self) { group in
            var iterator = needed.makeIterator()
            func addNext() {
                guard let item = iterator.next() else { return }
                group.addTask {
                    let data = try await client.file(path: item.path, commit: commit, expectedSHA: item.sha)
                    try objects.store(data, sha: item.sha)
                    return Int64(data.count)
                }
            }
            for _ in 0..<max(1, concurrency) { addNext() }
            var done = 0
            var bytes: Int64 = 0
            while let size = try await group.next() {
                done += 1
                bytes += size
                progress(done, total, bytes)
                addNext()
            }
        }
    }
}
