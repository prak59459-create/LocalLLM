import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// GitHub のリポジトリ（owner/name）。
struct GitHubRepository: Hashable, Sendable, CustomStringConvertible {
    var owner: String
    var name: String

    /// "owner/name"、"https://github.com/owner/name(.git)"、"github.com/owner/name/tree/main" などを読む
    init?(_ text: String) {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://", "http://", "www.", "github.com/", "git@github.com:"] where trimmed.lowercased().hasPrefix(prefix) {
            trimmed = String(trimmed.dropFirst(prefix.count))
        }
        if trimmed.lowercased().hasPrefix("github.com/") { trimmed = String(trimmed.dropFirst("github.com/".count)) }
        let parts = trimmed.split(separator: "/").map(String.init)
        guard parts.count >= 2 else { return nil }
        var name = parts[1]
        if name.hasSuffix(".git") { name = String(name.dropLast(4)) }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        guard !parts[0].isEmpty, !name.isEmpty,
              parts[0].unicodeScalars.allSatisfy(allowed.contains), name.unicodeScalars.allSatisfy(allowed.contains) else { return nil }
        owner = parts[0]
        self.name = name
    }

    var description: String { "\(owner)/\(name)" }
}

/// コミット 1 つ。
struct GitHubCommit: Codable, Hashable, Sendable {
    var sha: String
    var message: String
    var date: Date?
    var author: String?
    var treeSHA: String?

    var shortSHA: String { String(sha.prefix(7)) }
    /// メッセージの 1 行目
    var title: String { message.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? message }

    static func parse(_ data: Data) throws -> GitHubCommit {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sha = object["sha"] as? String else { throw GitHubAPIError.unexpected }
        let commit = object["commit"] as? [String: Any]
        let committer = commit?["committer"] as? [String: Any]
        let author = commit?["author"] as? [String: Any]
        let dateText = (committer?["date"] as? String) ?? (author?["date"] as? String)
        let tree = commit?["tree"] as? [String: Any]
        return GitHubCommit(sha: sha, message: commit?["message"] as? String ?? "",
                            date: dateText.flatMap { ISO8601DateFormatter().date(from: $0) },
                            author: author?["name"] as? String, treeSHA: tree?["sha"] as? String)
    }
}

/// ファイル一覧（git の tree）の 1 項目。
struct GitHubTreeEntry: Hashable, Sendable {
    var path: String
    var sha: String
    var size: Int64
    var isFile: Bool
}

struct GitHubTree: Sendable {
    var entries: [GitHubTreeEntry]
    /// 項目が多すぎて途中までしか返されなかった
    var truncated: Bool

    static func parse(_ data: Data) throws -> GitHubTree {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = object["tree"] as? [[String: Any]] else { throw GitHubAPIError.unexpected }
        let entries = items.compactMap { item -> GitHubTreeEntry? in
            guard let path = item["path"] as? String, let sha = item["sha"] as? String, let type = item["type"] as? String else { return nil }
            let size = (item["size"] as? NSNumber)?.int64Value ?? 0
            // シンボリックリンク（120000）とサブモジュール（commit）はファイルとして扱わない
            let isLink = (item["mode"] as? String) == "120000"
            return GitHubTreeEntry(path: path, sha: sha, size: size, isFile: type == "blob" && !isLink)
        }
        return GitHubTree(entries: entries, truncated: object["truncated"] as? Bool ?? false)
    }
}

enum GitHubAPIError: Error, LocalizedError, Equatable {
    case unexpected
    case invalidRepository(String)
    case http(Int, String)
    case notFound
    case unauthorized
    case rateLimited(Date?)
    case integrity(String)

    var errorDescription: String? {
        switch self {
        case .unexpected:
            return L("GitHub からの返事の形式が想定外です", "Unexpected response from GitHub")
        case .invalidRepository(let text):
            return L("リポジトリ「\(text)」の書き方が分かりません（例: owner/name）", "Cannot read the repository \"\(text)\" (e.g. owner/name)")
        case .http(let status, let message):
            return L("GitHub でエラーが起きました（HTTP \(status)）\(message.isEmpty ? "" : ": " + message)",
                     "GitHub returned an error (HTTP \(status))\(message.isEmpty ? "" : ": " + message)")
        case .notFound:
            return L("リポジトリかブランチが見つかりません。非公開のリポジトリならトークンを入れてください。",
                     "Repository or branch not found. Enter a token for private repositories.")
        case .unauthorized:
            return L("トークンが正しくないか、権限がありません。", "The token is invalid or lacks permission.")
        case .rateLimited(let reset):
            let time = reset.map { DateFormatter.localizedString(from: $0, dateStyle: .none, timeStyle: .short) }
            return L("GitHub の回数制限に達しました\(time.map { "（\($0) に回復）" } ?? "")。少し待つか、トークンを入れてください。",
                     "GitHub rate limit reached\(time.map { " (resets at \($0))" } ?? ""). Wait a while or enter a token.")
        case .integrity(let path):
            return L("\(path) の中身が GitHub の一覧と一致しません（通信の途中で壊れた可能性があります）",
                     "The content of \(path) does not match GitHub's listing (it may have been corrupted in transit)")
        }
    }
}

/// GitHub の API・ファイルの URL。
enum GitHubAPI {
    static let userAgent = "LocalLLM-Updater"

    static func request(_ url: URL, token: String?, accept: String = "application/vnd.github+json") -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue(accept, forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        if let token, !token.isEmpty { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.timeoutInterval = 60
        return request
    }

    private static func encode(_ component: String) -> String {
        component.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "?#"))) ?? component
    }

    static func commitURL(_ repository: GitHubRepository, ref: String) -> URL {
        URL(string: "https://api.github.com/repos/\(repository.owner)/\(repository.name)/commits/\(encode(ref))")!
    }

    static func treeURL(_ repository: GitHubRepository, sha: String, recursive: Bool = true) -> URL {
        URL(string: "https://api.github.com/repos/\(repository.owner)/\(repository.name)/git/trees/\(encode(sha))\(recursive ? "?recursive=1" : "")")!
    }

    static func branchesURL(_ repository: GitHubRepository) -> URL {
        URL(string: "https://api.github.com/repos/\(repository.owner)/\(repository.name)/branches?per_page=100")!
    }

    /// ファイルの中身。トークンがあれば API（非公開でも取れる）、無ければ raw.githubusercontent.com（回数制限に数えられない）。
    static func fileRequest(_ repository: GitHubRepository, commit: String, path: String, token: String?) -> URLRequest {
        let encodedPath = path.split(separator: "/").map { encode(String($0)) }.joined(separator: "/")
        if let token, !token.isEmpty {
            let url = URL(string: "https://api.github.com/repos/\(repository.owner)/\(repository.name)/contents/\(encodedPath)?ref=\(commit)")!
            return request(url, token: token, accept: "application/vnd.github.raw")
        }
        let url = URL(string: "https://raw.githubusercontent.com/\(repository.owner)/\(repository.name)/\(commit)/\(encodedPath)")!
        var request = URLRequest(url: url)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 60
        return request
    }

    static func parseBranches(_ data: Data) -> [String] {
        guard let items = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else { return [] }
        return items.compactMap { $0["name"] as? String }
    }

    /// HTTP の失敗をエラーにする
    static func check(status: Int, headers: [AnyHashable: Any], body: Data) throws {
        guard !(200..<300).contains(status) else { return }
        func header(_ name: String) -> String? {
            headers.first { ($0.key as? String)?.lowercased() == name.lowercased() }?.value as? String
        }
        let message = ((try? JSONSerialization.jsonObject(with: body)) as? [String: Any])?["message"] as? String ?? ""
        if status == 403 || status == 429, header("x-ratelimit-remaining") == "0" || message.lowercased().contains("rate limit") {
            let reset = header("x-ratelimit-reset").flatMap(Double.init).map { Date(timeIntervalSince1970: $0) }
            throw GitHubAPIError.rateLimited(reset)
        }
        if status == 401 { throw GitHubAPIError.unauthorized }
        if status == 404 { throw GitHubAPIError.notFound }
        throw GitHubAPIError.http(status, message)
    }
}
