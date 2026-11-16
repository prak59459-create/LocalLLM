import Foundation

/// ModelScope（modelscope.cn）。Hugging Face のほかに、GGUF のモデルがたくさん置いてある場所。
///
/// アプリの中では、ModelScope のリポジトリを "modelscope:持ち主/名前" という名前で扱う
/// （Hugging Face のリポジトリと同じ画面・同じダウンロードの仕組みで使えるように）。
/// 検索は、このアプリで動かせる GGUF のモデルだけにする。
enum ModelScope {
    static let prefix = "modelscope:"
    static let host = "modelscope.cn"
    static let base = "https://modelscope.cn"

    static func isModelScope(_ repo: String) -> Bool { repo.hasPrefix(prefix) }

    /// "modelscope:Qwen/Qwen2.5-0.5B-Instruct-GGUF" → "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    static func path(_ repo: String) -> String { String(repo.dropFirst(prefix.count)) }

    /// 画面に出す名前（"ModelScope · Qwen/…"）
    static func displayName(_ repo: String) -> String {
        isModelScope(repo) ? "ModelScope · " + path(repo) : repo
    }

    /// Hugging Face の "main" は、ModelScope では "master"
    static func revision(_ revision: String) -> String {
        revision == "main" || revision.isEmpty ? "master" : revision
    }

    private static func encoded(_ text: String) -> String {
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "/?#[]@!$&'()*+,;="))
        return text.addingPercentEncoding(withAllowedCharacters: allowed) ?? text
    }

    private static func encodedPath(_ path: String) -> String {
        path.split(separator: "/", omittingEmptySubsequences: false).map { encoded(String($0)) }.joined(separator: "/")
    }

    /// https://modelscope.cn/models/{持ち主}/{名前}/resolve/{revision}/{path}
    static func resolveURL(repo: String, path file: String, revision: String) -> URL {
        URL(string: "\(base)/models/\(encodedPath(path(repo)))/resolve/\(encoded(Self.revision(revision)))/\(encodedPath(file))")!
    }

    static func infoURL(repo: String) -> URL {
        URL(string: "\(base)/api/v1/models/\(encodedPath(path(repo)))")!
    }

    static func filesURL(repo: String, revision: String) -> URL {
        var components = URLComponents(string: "\(base)/api/v1/models/\(encodedPath(path(repo)))/repo/files")!
        components.queryItems = [URLQueryItem(name: "Revision", value: Self.revision(revision)),
                                 URLQueryItem(name: "Recursive", value: "true")]
        return components.url!
    }

    // MARK: - 検索

    /// 検索のページを表す URL（実際には、この URL に JSON を付けて PUT する。ページ送りのために URL で持つ）
    static func searchURL(query: String, sort: HFSearchSort, page: Int, limit: Int) -> URL {
        var components = URLComponents(string: "\(base)/api/v1/dolphin/models")!
        components.queryItems = [URLQueryItem(name: "Name", value: query.trimmingCharacters(in: .whitespacesAndNewlines)),
                                 URLQueryItem(name: "SortBy", value: sortName(sort)),
                                 URLQueryItem(name: "PageNumber", value: String(page)),
                                 URLQueryItem(name: "PageSize", value: String(limit))]
        return components.url!
    }

    static func isSearchURL(_ url: URL) -> Bool {
        url.host == host && url.path.hasSuffix("/dolphin/models")
    }

    private static func sortName(_ sort: HFSearchSort) -> String {
        switch sort {
        case .downloads, .trendingScore: return "DownloadsCount"
        case .likes: return "StarsCount"
        case .lastModified: return "GmtModified"
        }
    }

    private static func value(_ url: URL, _ name: String) -> String? {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == name }?.value
    }

    /// 検索の URL から、PUT で送る中身（GGUF のモデルだけに絞る）
    static func searchBody(for url: URL) -> Data {
        let body: [String: Any] = [
            "Name": value(url, "Name") ?? "",
            "SortBy": value(url, "SortBy") ?? "DownloadsCount",
            "PageNumber": Int(value(url, "PageNumber") ?? "1") ?? 1,
            "PageSize": Int(value(url, "PageSize") ?? "50") ?? 50,
            "Target": "",
            "SingleCriterion": [] as [Any],
            "Criterion": [["category": "libraries", "predicate": "contains", "values": ["gguf"]]],
        ]
        return (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
    }

    static func parseSearch(_ data: Data, requestURL: URL) throws -> HFSearchPage {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = (root["Data"] as? [String: Any])?["Model"] as? [String: Any] else {
            throw HFError.invalidResponse("ModelScope search")
        }
        let items = model["Models"] as? [[String: Any]] ?? []
        let models = items.compactMap { item -> HFModelSummary? in
            guard let owner = item["Path"] as? String, let name = item["Name"] as? String else { return nil }
            let task = (item["Tasks"] as? [[String: Any]])?.first?["Name"] as? String
            return HFModelSummary(id: prefix + owner + "/" + name,
                                  downloads: item["Downloads"] as? Int,
                                  likes: item["Stars"] as? Int,
                                  pipelineTag: task,
                                  libraryName: (item["Libraries"] as? [String])?.first,
                                  tags: item["Tags"] as? [String] ?? [],
                                  lastModified: (item["LastUpdatedTime"] as? Double).map { Date(timeIntervalSince1970: $0) },
                                  gated: false)
        }
        let total = model["TotalCount"] as? Int ?? 0
        let page = Int(value(requestURL, "PageNumber") ?? "1") ?? 1
        let size = Int(value(requestURL, "PageSize") ?? "50") ?? 50
        var next: URL?
        if page * size < total, !models.isEmpty, var components = URLComponents(url: requestURL, resolvingAgainstBaseURL: false) {
            components.queryItems = (components.queryItems ?? []).map {
                $0.name == "PageNumber" ? URLQueryItem(name: "PageNumber", value: String(page + 1)) : $0
            }
            next = components.url
        }
        return HFSearchPage(models: models, next: next)
    }

    // MARK: - リポジトリ

    static func parseFiles(_ data: Data) throws -> [HFFileEntry] {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let files = (root["Data"] as? [String: Any])?["Files"] as? [[String: Any]] else {
            throw HFError.invalidResponse("ModelScope files")
        }
        return files.compactMap { file in
            guard (file["Type"] as? String) != "tree", let path = file["Path"] as? String else { return nil }
            let size = (file["Size"] as? NSNumber)?.int64Value ?? 0
            return HFFileEntry(path: path, size: size, lfsSHA256: (file["IsLFS"] as? Bool) == true ? file["Sha256"] as? String : nil)
        }
    }

    static func parseInfo(_ data: Data, repo: String) throws -> HFModelInfo {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let item = root["Data"] as? [String: Any] else {
            throw HFError.invalidResponse("ModelScope model")
        }
        let tasks = (item["Tasks"] as? [[String: Any]])?.compactMap { $0["Name"] as? String } ?? []
        return HFModelInfo(id: repo, sha: nil, tags: item["Tags"] as? [String] ?? [], pipelineTag: tasks.first,
                           libraryName: (item["Libraries"] as? [String])?.first, gated: false, baseModels: [], siblings: [])
    }

    // MARK: - まとめて検索（Hugging Face と ModelScope）

    static let combinedScheme = "localllm-search"

    /// 2 つの場所の次のページを 1 つの URL にまとめる
    static func combinedURL(huggingFace: URL?, modelScope: URL?) -> URL? {
        guard huggingFace != nil || modelScope != nil else { return nil }
        var components = URLComponents()
        components.scheme = combinedScheme
        components.host = "all"
        components.queryItems = [huggingFace.map { URLQueryItem(name: "hf", value: $0.absoluteString) },
                                 modelScope.map { URLQueryItem(name: "ms", value: $0.absoluteString) }].compactMap { $0 }
        return components.url
    }

    static func combinedParts(_ url: URL) -> (huggingFace: URL?, modelScope: URL?) {
        (value(url, "hf").flatMap(URL.init(string:)), value(url, "ms").flatMap(URL.init(string:)))
    }

    /// 2 つの結果を交互に並べる（同じ名前は 1 つに）
    static func interleave(_ first: [HFModelSummary], _ second: [HFModelSummary]) -> [HFModelSummary] {
        var result: [HFModelSummary] = []
        var seen = Set<String>()
        for index in 0..<max(first.count, second.count) {
            for list in [first, second] where index < list.count && seen.insert(list[index].id).inserted {
                result.append(list[index])
            }
        }
        return result
    }
}
