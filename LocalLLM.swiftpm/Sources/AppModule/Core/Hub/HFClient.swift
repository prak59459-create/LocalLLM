import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// 検索結果のモデル 1 つ。
struct HFModelSummary: Codable, Hashable, Identifiable, Sendable {
    var id: String
    var downloads: Int?
    var likes: Int?
    var pipelineTag: String?
    var libraryName: String?
    var tags: [String]
    var lastModified: Date?
    var gated: Bool
}

/// リポジトリ内のファイル 1 つ。
struct HFFileEntry: Codable, Hashable, Sendable {
    var path: String
    var size: Int64
    var lfsSHA256: String?
}

/// モデルの詳細情報。
struct HFModelInfo: Codable, Hashable, Sendable {
    var id: String
    var sha: String?
    var tags: [String]
    var pipelineTag: String?
    var libraryName: String?
    var gated: Bool
    var baseModels: [String]
    var siblings: [String]
}

enum HFError: Error, LocalizedError {
    case http(status: Int, url: URL)
    case invalidResponse(String)
    case network(String)

    var errorDescription: String? {
        switch self {
        case .http(let status, _):
            switch status {
            case 401, 403:
                return L("アクセスが拒否されました（\(status)）。ゲート付きのモデルなら、Hugging Face で利用規約に同意し、設定にトークンを入力してください。",
                         "Access denied (\(status)). For gated models, accept the terms on Hugging Face and enter your token in Settings.")
            case 404:
                return L("見つかりませんでした（404）。リポジトリ名やファイル名を確認してください。",
                         "Not found (404). Check the repository or file name.")
            case 429:
                return L("アクセスが集中しています（429）。少し待ってからもう一度試してください。",
                         "Too many requests (429). Please wait a moment and try again.")
            case 500...599:
                return L("Hugging Face のサーバーでエラーが起きました（\(status)）。時間をおいて試してください。",
                         "Hugging Face server error (\(status)). Please try again later.")
            default:
                return L("通信エラー（HTTP \(status)）", "HTTP error \(status)")
            }
        case .invalidResponse(let detail):
            return L("想定外の応答です: \(detail)", "Unexpected response: \(detail)")
        case .network(let detail):
            return L("ネットワークに接続できません: \(detail)", "Network error: \(detail)")
        }
    }
}

/// Hugging Face の API クライアント。
final class HFClient: @unchecked Sendable {
    static let defaultEndpoint = URL(string: "https://huggingface.co")!
    static let userAgent = "LocalLLM-iPad/0.1"

    let token: String?
    let endpoint: URL
    private let session: URLSession

    init(token: String? = nil, endpoint: URL = HFClient.defaultEndpoint, session: URLSession = .shared) {
        let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.token = (trimmed?.isEmpty ?? true) ? nil : trimmed
        self.endpoint = endpoint
        self.session = session
    }

    /// ダウンロードに付けるヘッダ
    var authorizationHeaders: [String: String] {
        var headers = ["User-Agent": Self.userAgent]
        if let token { headers["Authorization"] = "Bearer \(token)" }
        return headers
    }

    // MARK: - API

    func searchModels(query: String, coreMLOnly: Bool, textGenerationOnly: Bool, limit: Int) async throws -> [HFModelSummary] {
        var components = URLComponents(url: endpoint.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)!
        var items = [
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { items.append(URLQueryItem(name: "search", value: trimmed)) }
        if coreMLOnly { items.append(URLQueryItem(name: "filter", value: "coreml")) }
        if textGenerationOnly { items.append(URLQueryItem(name: "pipeline_tag", value: "text-generation")) }
        components.queryItems = items
        let (data, _) = try await get(components.url!)
        guard let array = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw HFError.invalidResponse("models")
        }
        return array.compactMap { item in
            guard let id = item["id"] as? String ?? item["modelId"] as? String else { return nil }
            return HFModelSummary(id: id,
                                  downloads: item["downloads"] as? Int,
                                  likes: item["likes"] as? Int,
                                  pipelineTag: item["pipeline_tag"] as? String,
                                  libraryName: item["library_name"] as? String,
                                  tags: item["tags"] as? [String] ?? [],
                                  lastModified: (item["lastModified"] as? String).flatMap(Self.parseDate),
                                  gated: Self.isGated(item["gated"]))
        }
    }

    func modelInfo(repo: String) async throws -> HFModelInfo {
        let (data, _) = try await get(endpoint.appendingPathComponent("api/models").appendingPathComponent(repo))
        guard let item = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HFError.invalidResponse("model info")
        }
        var baseModels: [String] = []
        if let card = item["cardData"] as? [String: Any] {
            if let single = card["base_model"] as? String {
                baseModels = [single]
            } else if let list = card["base_model"] as? [String] {
                baseModels = list
            }
        }
        let siblings = (item["siblings"] as? [[String: Any]] ?? []).compactMap { $0["rfilename"] as? String }
        return HFModelInfo(id: item["id"] as? String ?? repo,
                           sha: item["sha"] as? String,
                           tags: item["tags"] as? [String] ?? [],
                           pipelineTag: item["pipeline_tag"] as? String,
                           libraryName: item["library_name"] as? String,
                           gated: Self.isGated(item["gated"]),
                           baseModels: baseModels,
                           siblings: siblings)
    }

    /// リポジトリの全ファイル（再帰的に、ページングもたどる）
    func listFiles(repo: String, revision: String) async throws -> [HFFileEntry] {
        var url: URL? = treeURL(repo: repo, revision: revision)
        var files: [HFFileEntry] = []
        var pages = 0
        while let current = url, pages < 50 {
            pages += 1
            let (data, response) = try await get(current)
            files.append(contentsOf: try Self.parseTree(data))
            url = (response.value(forHTTPHeaderField: "Link")).flatMap(Self.nextLink)
        }
        return files
    }

    func treeURL(repo: String, revision: String) -> URL {
        var components = URLComponents(url: endpoint.appendingPathComponent("api/models").appendingPathComponent(repo)
            .appendingPathComponent("tree").appendingPathComponent(revision), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        return components.url!
    }

    /// https://huggingface.co/{repo}/resolve/{revision}/{path}
    func resolveURL(repo: String, path: String, revision: String) -> URL {
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "/?#[]@!$&'()*+,;="))
        let encodedPath = path.split(separator: "/", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: allowed) ?? String($0) }
            .joined(separator: "/")
        let encodedRevision = revision.addingPercentEncoding(withAllowedCharacters: allowed) ?? revision
        let base = endpoint.absoluteString.hasSuffix("/") ? String(endpoint.absoluteString.dropLast()) : endpoint.absoluteString
        return URL(string: "\(base)/\(repo)/resolve/\(encodedRevision)/\(encodedPath)")!
    }

    /// 小さいファイルを取得する
    func fetchData(repo: String, path: String, revision: String) async throws -> Data {
        try await get(resolveURL(repo: repo, path: path, revision: revision)).0
    }

    // MARK: - 補助

    private func get(_ url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        for (key, value) in authorizationHeaders { request.setValue(value, forHTTPHeaderField: key) }
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            if error is CancellationError { throw error }
            if (error as? URLError)?.code == .cancelled { throw CancellationError() }
            throw HFError.network(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else { throw HFError.invalidResponse("not HTTP") }
        guard (200..<300).contains(http.statusCode) else { throw HFError.http(status: http.statusCode, url: url) }
        return (data, http)
    }

    static func parseTree(_ data: Data) throws -> [HFFileEntry] {
        guard let array = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw HFError.invalidResponse("tree")
        }
        return array.compactMap { item in
            guard (item["type"] as? String) == "file", let path = item["path"] as? String else { return nil }
            let lfs = item["lfs"] as? [String: Any]
            let size = (lfs?["size"] as? Int) ?? (item["size"] as? Int) ?? 0
            return HFFileEntry(path: path, size: Int64(size), lfsSHA256: lfs?["oid"] as? String)
        }
    }

    /// Link: <https://...>; rel="next"
    static func nextLink(_ header: String) -> URL? {
        for part in header.components(separatedBy: ",") where part.contains("rel=\"next\"") {
            guard let start = part.firstIndex(of: "<"), let end = part.firstIndex(of: ">"), start < end else { continue }
            return URL(string: String(part[part.index(after: start)..<end]))
        }
        return nil
    }

    static func isGated(_ value: Any?) -> Bool {
        if let flag = value as? Bool { return flag }
        if let text = value as? String { return !text.isEmpty && text != "false" }
        return false
    }

    static func parseDate(_ text: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: text) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: text)
    }
}

// MARK: - リポジトリの解析

/// リポジトリ内のモデルの候補（.mlmodelc / .mlpackage のフォルダ）。
struct ModelVariant: Codable, Hashable, Identifiable, Sendable {
    enum Kind: String, Codable, Sendable {
        case compiled   // .mlmodelc
        case package    // .mlpackage
    }

    var id: String { path }
    var path: String
    var kind: Kind
    var files: [HFFileEntry]
    var totalBytes: Int64
    var displayName: String
}

struct RepoAnalysis: Hashable, Sendable {
    var variants: [ModelVariant]
    var supportFiles: [HFFileEntry]
    var hasTokenizer: Bool

    /// ダウンロードする付属ファイルの名前
    static let supportFileNames: Set<String> = [
        "tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json",
        "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "added_tokens.json",
    ]

    static func analyze(files: [HFFileEntry]) -> RepoAnalysis {
        var groups: [String: [HFFileEntry]] = [:]
        for file in files {
            guard let root = modelRoot(of: file.path) else { continue }
            groups[root, default: []].append(file)
        }
        let variants = groups.map { root, entries -> ModelVariant in
            let kind: ModelVariant.Kind = root.hasSuffix(".mlmodelc") ? .compiled : .package
            let name = (root as NSString).lastPathComponent
            let displayName = (name as NSString).deletingPathExtension
            return ModelVariant(path: root, kind: kind, files: entries.sorted { $0.path < $1.path },
                                totalBytes: entries.reduce(Int64(0)) { $0 + $1.size }, displayName: displayName)
        }
        .sorted { lhs, rhs in
            if lhs.kind != rhs.kind { return lhs.kind == .compiled }
            return lhs.path < rhs.path
        }
        let support = files.filter { supportFileNames.contains(($0.path as NSString).lastPathComponent) && modelRoot(of: $0.path) == nil }
        let hasTokenizer = support.contains { ($0.path as NSString).lastPathComponent == "tokenizer.json" }
        return RepoAnalysis(variants: variants, supportFiles: support.sorted { $0.path < $1.path }, hasTokenizer: hasTokenizer)
    }

    /// パスの中で最初に出てくる .mlmodelc / .mlpackage のフォルダ（入れ子の内側は無視）
    static func modelRoot(of path: String) -> String? {
        let components = path.split(separator: "/").map(String.init)
        for (index, component) in components.enumerated() where index < components.count - 1 {
            if component.hasSuffix(".mlmodelc") || component.hasSuffix(".mlpackage") {
                return components[0...index].joined(separator: "/")
            }
        }
        return nil
    }

    /// 選んだモデルのダウンロード対象（モデルのフォルダ + そのモデルに近い付属ファイル）
    func filesToDownload(for variant: ModelVariant) -> [HFFileEntry] {
        let variantParent = (variant.path as NSString).deletingLastPathComponent
        var chosen: [String: HFFileEntry] = [:]
        for file in supportFiles {
            let name = (file.path as NSString).lastPathComponent
            let candidateParent = (file.path as NSString).deletingLastPathComponent
            if let existing = chosen[name] {
                let existingParent = (existing.path as NSString).deletingLastPathComponent
                if Self.distance(from: variantParent, to: candidateParent) < Self.distance(from: variantParent, to: existingParent) {
                    chosen[name] = file
                }
            } else {
                chosen[name] = file
            }
        }
        return variant.files + chosen.values.sorted { $0.path < $1.path }
    }

    /// フォルダどうしの近さ（同じフォルダ 0、祖先は段数、それ以外は大きな値 + 深さ）
    private static func distance(from base: String, to folder: String) -> Int {
        if base == folder { return 0 }
        let baseParts = base.split(separator: "/")
        let folderParts = folder.split(separator: "/")
        if baseParts.starts(with: folderParts) { return baseParts.count - folderParts.count }
        return 100 + folderParts.count
    }
}
