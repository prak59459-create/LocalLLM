import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Web 検索の結果 1 件。
struct WebSearchResult: Codable, Hashable, Sendable {
    var title: String
    var url: String
    var snippet: String
}

/// 検索に使うサービス。
enum WebSearchProvider: String, Codable, CaseIterable, Identifiable, Sendable {
    case duckDuckGo
    case wikipedia
    case brave
    case searxng

    var id: String { rawValue }

    var label: String {
        switch self {
        case .duckDuckGo: return L("DuckDuckGo（キー不要）", "DuckDuckGo (no key)")
        case .wikipedia: return L("Wikipedia（キー不要）", "Wikipedia (no key)")
        case .brave: return L("Brave Search（API キー）", "Brave Search (API key)")
        case .searxng: return L("SearXNG（自分のサーバー）", "SearXNG (your server)")
        }
    }
}

enum WebToolError: Error, LocalizedError {
    case http(Int)
    case blocked
    case invalidURL(String)
    case missingKey
    case notText(String)

    var errorDescription: String? {
        switch self {
        case .http(let status): return L("Web の取得に失敗しました（HTTP \(status)）", "Web request failed (HTTP \(status))")
        case .blocked: return L("検索サービスに一時的に断られました。少し待つか、設定で別の検索サービスを選んでください。",
                                "The search service refused the request. Wait a bit or pick another service in Settings.")
        case .invalidURL(let text): return L("URL が正しくありません: \(text)", "Invalid URL: \(text)")
        case .missingKey: return L("API キーが設定されていません", "No API key is set")
        case .notText(let type): return L("テキストのページではありません（\(type)）", "Not a text page (\(type))")
        }
    }
}

/// Web 検索とページの取得（ネットワークを使う部分）。
struct WebSearchClient: Sendable {
    var provider: WebSearchProvider = .duckDuckGo
    var apiKey: String?
    var searxngURL: String?
    /// Wikipedia の言語（ja / en など）
    var language: String = "ja"
    var session: URLSession = .shared

    static let botUserAgent = "LocalLLM-iPad/1.0 (https://github.com/prak59459-create/LocalLLM)"
    static let userAgent = "Mozilla/5.0 (iPad; CPU OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"

    func search(_ query: String, limit: Int = 5) async throws -> [WebSearchResult] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        switch provider {
        case .duckDuckGo:
            var components = URLComponents(string: "https://html.duckduckgo.com/html/")!
            components.queryItems = [URLQueryItem(name: "q", value: trimmed)]
            let (data, status) = try await get(components.url!)
            if status == 202 || status == 403 { throw WebToolError.blocked }
            guard status == 200 else { throw WebToolError.http(status) }
            return Array(DuckDuckGoParser.parse(String(decoding: data, as: UTF8.self)).prefix(limit))
        case .wikipedia:
            var components = URLComponents(string: "https://\(language).wikipedia.org/w/api.php")!
            components.queryItems = [
                URLQueryItem(name: "action", value: "query"), URLQueryItem(name: "list", value: "search"),
                URLQueryItem(name: "srsearch", value: trimmed), URLQueryItem(name: "format", value: "json"),
                URLQueryItem(name: "srlimit", value: String(limit)), URLQueryItem(name: "utf8", value: "1"),
            ]
            // Wikimedia は、連絡先の分かる User-Agent を求める
            let (data, status) = try await get(components.url!, headers: ["User-Agent": Self.botUserAgent])
            guard status == 200 else { throw WebToolError.http(status) }
            return WikipediaParser.parseSearch(data, language: language)
        case .brave:
            guard let apiKey, !apiKey.isEmpty else { throw WebToolError.missingKey }
            var components = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")!
            components.queryItems = [URLQueryItem(name: "q", value: trimmed), URLQueryItem(name: "count", value: String(limit))]
            let (data, status) = try await get(components.url!, headers: ["X-Subscription-Token": apiKey, "Accept": "application/json"])
            guard status == 200 else { throw WebToolError.http(status) }
            return BraveParser.parse(data)
        case .searxng:
            guard let base = searxngURL, var components = URLComponents(string: base) else {
                throw WebToolError.invalidURL(searxngURL ?? "")
            }
            if !components.path.hasSuffix("/search") { components.path = (components.path.hasSuffix("/") ? components.path : components.path + "/") + "search" }
            components.queryItems = [URLQueryItem(name: "q", value: trimmed), URLQueryItem(name: "format", value: "json")]
            guard let url = components.url else { throw WebToolError.invalidURL(base) }
            let (data, status) = try await get(url)
            guard status == 200 else { throw WebToolError.http(status) }
            return Array(SearXNGParser.parse(data).prefix(limit))
        }
    }

    /// ページを取得して本文のテキストにする
    func fetchPage(_ address: String, maxCharacters: Int = 6000) async throws -> (title: String, text: String) {
        var text = address.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.lowercased().hasPrefix("http") { text = "https://" + text }
        guard let url = URL(string: text), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw WebToolError.invalidURL(address)
        }
        let isWikimedia = url.host?.hasSuffix("wikipedia.org") ?? false
        let (data, status, type) = try await getWithType(url, headers: isWikimedia ? ["User-Agent": Self.botUserAgent] : [:])
        guard (200..<300).contains(status) else { throw WebToolError.http(status) }
        let lowerType = type.lowercased()
        if lowerType.contains("json") || lowerType.hasPrefix("text/plain") || lowerType.contains("markdown") {
            let body = String(decoding: data.prefix(2_000_000), as: UTF8.self)
            return (url.absoluteString, String(body.prefix(maxCharacters)))
        }
        guard lowerType.isEmpty || lowerType.contains("html") || lowerType.contains("xml") else { throw WebToolError.notText(type) }
        let page = HTMLText.extract(data.prefix(3_000_000))
        return (page.title.isEmpty ? url.absoluteString : page.title, String(page.text.prefix(maxCharacters)))
    }

    private func get(_ url: URL, headers: [String: String] = [:]) async throws -> (Data, Int) {
        let (data, status, _) = try await getWithType(url, headers: headers)
        return (data, status)
    }

    private func getWithType(_ url: URL, headers: [String: String] = [:]) async throws -> (Data, Int, String) {
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("ja,en;q=0.8", forHTTPHeaderField: "Accept-Language")
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        let (data, response) = try await session.data(for: request)
        let http = response as? HTTPURLResponse
        return (data, http?.statusCode ?? 0, http?.value(forHTTPHeaderField: "Content-Type") ?? "")
    }

    /// 検索結果をモデルに渡す文章にする
    static func format(_ results: [WebSearchResult]) -> String {
        guard !results.isEmpty else { return L("検索結果はありませんでした。", "No results.") }
        return results.enumerated().map { index, result in
            "[\(index + 1)] \(result.title)\n\(result.url)\n\(result.snippet)"
        }.joined(separator: "\n\n")
    }
}

/// DuckDuckGo の HTML 版の検索結果を読む。
enum DuckDuckGoParser {
    static func parse(_ html: String) -> [WebSearchResult] {
        var results: [WebSearchResult] = []
        var cursor = html.startIndex
        while let anchor = html.range(of: "class=\"result__a\"", range: cursor..<html.endIndex) {
            // 広告は飛ばす（結果の div に result--ad が付く）
            let blockStart = html.range(of: "<div class=\"result", options: .backwards, range: html.startIndex..<anchor.lowerBound)
            let isAd = blockStart.map { html[$0.lowerBound..<anchor.lowerBound].contains("result--ad") } ?? false
            guard let tagStart = html.range(of: "<a", options: .backwards, range: html.startIndex..<anchor.lowerBound),
                  let tagEnd = html.range(of: ">", range: anchor.upperBound..<html.endIndex),
                  let close = html.range(of: "</a>", range: tagEnd.upperBound..<html.endIndex) else { break }
            let tag = String(html[tagStart.lowerBound..<tagEnd.upperBound])
            let title = HTMLText.plain(String(html[tagEnd.upperBound..<close.lowerBound]))
            var snippet = ""
            var next = close.upperBound
            if let snippetAnchor = html.range(of: "class=\"result__snippet\"", range: close.upperBound..<html.endIndex) {
                let nextResult = html.range(of: "class=\"result__a\"", range: close.upperBound..<html.endIndex)
                if nextResult == nil || snippetAnchor.lowerBound < nextResult!.lowerBound,
                   let snippetStart = html.range(of: ">", range: snippetAnchor.upperBound..<html.endIndex),
                   let snippetEnd = html.range(of: "</a>", range: snippetStart.upperBound..<html.endIndex) {
                    snippet = HTMLText.plain(String(html[snippetStart.upperBound..<snippetEnd.lowerBound]))
                    next = snippetEnd.upperBound
                }
            }
            cursor = next
            guard !isAd, let href = attribute("href", in: tag), let url = resolve(href) else { continue }
            results.append(WebSearchResult(title: title, url: url, snippet: snippet))
        }
        return results
    }

    static func attribute(_ name: String, in tag: String) -> String? {
        guard let start = tag.range(of: name + "=\"") else { return nil }
        guard let end = tag.range(of: "\"", range: start.upperBound..<tag.endIndex) else { return nil }
        return HTMLText.decodeEntities(String(tag[start.upperBound..<end.lowerBound]))
    }

    /// //duckduckgo.com/l/?uddg=<本当の URL> を元に戻す
    static func resolve(_ href: String) -> String? {
        var text = href
        if text.hasPrefix("//") { text = "https:" + text }
        if let components = URLComponents(string: text), components.path == "/l/",
           let target = components.queryItems?.first(where: { $0.name == "uddg" })?.value {
            return target
        }
        if text.contains("duckduckgo.com/y.js") { return nil }   // 広告
        return text.hasPrefix("http") ? text : nil
    }
}

enum WikipediaParser {
    static func parseSearch(_ data: Data, language: String) -> [WebSearchResult] {
        guard let value = try? JSONValue.parse(String(decoding: data, as: UTF8.self)),
              let items = value["query"]?["search"]?.arrayValue else { return [] }
        return items.compactMap { item in
            guard let title = item["title"]?.stringValue else { return nil }
            let path = title.replacingOccurrences(of: " ", with: "_")
                .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? title
            return WebSearchResult(title: title, url: "https://\(language).wikipedia.org/wiki/\(path)",
                                   snippet: HTMLText.plain(item["snippet"]?.stringValue ?? ""))
        }
    }
}

enum BraveParser {
    static func parse(_ data: Data) -> [WebSearchResult] {
        guard let value = try? JSONValue.parse(String(decoding: data, as: UTF8.self)),
              let items = value["web"]?["results"]?.arrayValue else { return [] }
        return items.compactMap { item in
            guard let title = item["title"]?.stringValue, let url = item["url"]?.stringValue else { return nil }
            return WebSearchResult(title: HTMLText.plain(title), url: url, snippet: HTMLText.plain(item["description"]?.stringValue ?? ""))
        }
    }
}

enum SearXNGParser {
    static func parse(_ data: Data) -> [WebSearchResult] {
        guard let value = try? JSONValue.parse(String(decoding: data, as: UTF8.self)),
              let items = value["results"]?.arrayValue else { return [] }
        return items.compactMap { item in
            guard let title = item["title"]?.stringValue, let url = item["url"]?.stringValue else { return nil }
            return WebSearchResult(title: title, url: url, snippet: item["content"]?.stringValue ?? "")
        }
    }
}

/// HTML から読める文章を取り出す（バイト列を直接読むので、最適化なしのビルドでも速い）。
enum HTMLText {
    /// タグを取り除き、文字参照を戻す（1 行の短い文用）
    static func plain(_ html: String) -> String {
        let text = extract(Data(html.utf8), keepTitle: false).text
        return text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// 本文（script / style などは除く。段落や見出しは改行にする）
    static func extract<D: DataProtocol>(_ data: D, keepTitle: Bool = true) -> (title: String, text: String) {
        let bytes = [UInt8](data)
        var output: [UInt8] = []
        output.reserveCapacity(bytes.count / 3)
        var title: [UInt8] = []
        var inTitle = false
        var skipUntil: [UInt8]? = nil   // </script> など
        var lastWasSpace = true
        var newlines = 2
        let skipTags: Set<String> = ["script", "style", "noscript", "svg", "template", "iframe", "nav", "footer"]
        let blockTags: Set<String> = ["p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
                                      "section", "article", "header", "pre", "blockquote", "hr", "dd", "dt", "main", "aside", "form"]
        var i = 0
        let n = bytes.count

        func appendSpace() {
            if !lastWasSpace {
                output.append(0x20)
                lastWasSpace = true
            }
        }
        func appendNewline() {
            while let last = output.last, last == 0x20 { output.removeLast() }
            // 段落の区切りは改行 1 つ（小さなモデルの文脈を節約する）
            if newlines < 1 {
                output.append(0x0A)
                newlines += 1
            }
            lastWasSpace = true
        }

        while i < n {
            let b = bytes[i]
            if let end = skipUntil {
                // 終わりのタグ（大文字小文字を無視）を探す
                if b == 0x3C, i + end.count <= n, matches(bytes, at: i, lowercased: end) {
                    i += end.count
                    skipUntil = nil
                    if let close = firstIndex(of: 0x3E, in: bytes, from: i) { i = close + 1 }
                } else {
                    i += 1
                }
                continue
            }
            if b == 0x3C {   // <
                // コメント
                if i + 3 < n && bytes[i + 1] == 0x21 && bytes[i + 2] == 0x2D && bytes[i + 3] == 0x2D {
                    if let end = find([0x2D, 0x2D, 0x3E], in: bytes, from: i + 4) { i = end + 3 } else { i = n }
                    continue
                }
                guard let close = firstIndex(of: 0x3E, in: bytes, from: i + 1) else { break }
                var nameStart = i + 1
                let isClosing = nameStart < close && bytes[nameStart] == 0x2F
                if isClosing { nameStart += 1 }
                var nameEnd = nameStart
                while nameEnd < close, isNameByte(bytes[nameEnd]) { nameEnd += 1 }
                let name = String(decoding: bytes[nameStart..<nameEnd], as: UTF8.self).lowercased()
                i = close + 1
                if name == "title" {
                    inTitle = !isClosing && keepTitle
                    continue
                }
                if !isClosing && skipTags.contains(name) && !(bytes[close - 1] == 0x2F) {
                    skipUntil = Array("</\(name)".utf8)
                    continue
                }
                if blockTags.contains(name) {
                    appendNewline()
                } else if name == "td" || name == "th" {
                    appendSpace()
                }
                continue
            }
            if b == 0x26 {   // &
                if let semicolon = firstIndex(of: 0x3B, in: bytes, from: i + 1), semicolon - i <= 10 {
                    let entity = String(decoding: bytes[(i + 1)..<semicolon], as: UTF8.self)
                    if let decoded = decodeEntity(entity) {
                        let utf8 = Array(decoded.utf8)
                        if inTitle { title.append(contentsOf: utf8) } else {
                            if decoded == " " { appendSpace() } else {
                                output.append(contentsOf: utf8)
                                lastWasSpace = false
                                newlines = 0
                            }
                        }
                        i = semicolon + 1
                        continue
                    }
                }
            }
            if inTitle {
                title.append(b)
                i += 1
                continue
            }
            if b == 0x20 || b == 0x0A || b == 0x0D || b == 0x09 {
                appendSpace()
            } else {
                output.append(b)
                lastWasSpace = false
                newlines = 0
            }
            i += 1
        }
        let text = String(decoding: output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let titleText = String(decoding: title, as: UTF8.self)
            .replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        return (titleText, text)
    }

    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var output = ""
        var rest = Substring(text)
        while let amp = rest.firstIndex(of: "&") {
            output += rest[..<amp]
            let after = rest[rest.index(after: amp)...]
            if let semicolon = after.firstIndex(of: ";"), after.distance(from: after.startIndex, to: semicolon) <= 10,
               let decoded = decodeEntity(String(after[..<semicolon])) {
                output += decoded
                rest = after[after.index(after: semicolon)...]
            } else {
                output += "&"
                rest = after
            }
        }
        return output + rest
    }

    static func decodeEntity(_ entity: String) -> String? {
        switch entity {
        case "amp": return "&"
        case "lt": return "<"
        case "gt": return ">"
        case "quot": return "\""
        case "apos": return "'"
        case "nbsp", "ensp", "emsp", "thinsp": return " "
        case "mdash": return "—"
        case "ndash": return "–"
        case "hellip": return "…"
        case "laquo": return "«"
        case "raquo": return "»"
        case "copy": return "©"
        case "reg": return "®"
        case "yen": return "¥"
        case "middot": return "·"
        default:
            if entity.hasPrefix("#x") || entity.hasPrefix("#X"), let code = UInt32(entity.dropFirst(2), radix: 16) {
                return Unicode.Scalar(code).map { String(Character($0)) }
            }
            if entity.hasPrefix("#"), let code = UInt32(entity.dropFirst()) {
                return Unicode.Scalar(code).map { String(Character($0)) }
            }
            return nil
        }
    }

    private static func isNameByte(_ b: UInt8) -> Bool {
        (b >= 0x61 && b <= 0x7A) || (b >= 0x41 && b <= 0x5A) || (b >= 0x30 && b <= 0x39) || b == 0x2D
    }

    private static func firstIndex(of byte: UInt8, in bytes: [UInt8], from start: Int) -> Int? {
        var i = start
        while i < bytes.count {
            if bytes[i] == byte { return i }
            i += 1
        }
        return nil
    }

    private static func find(_ needle: [UInt8], in bytes: [UInt8], from start: Int) -> Int? {
        guard !needle.isEmpty, bytes.count >= needle.count else { return nil }
        var i = start
        while i + needle.count <= bytes.count {
            if bytes[i] == needle[0] && matches(bytes, at: i, lowercased: needle) { return i }
            i += 1
        }
        return nil
    }

    /// bytes[at...] が needle（小文字）と大文字小文字を無視して一致するか
    private static func matches(_ bytes: [UInt8], at index: Int, lowercased needle: [UInt8]) -> Bool {
        guard index + needle.count <= bytes.count else { return false }
        for k in 0..<needle.count {
            var b = bytes[index + k]
            if b >= 0x41 && b <= 0x5A { b += 0x20 }
            if b != needle[k] { return false }
        }
        return true
    }
}
