import Foundation

/// 文章やコードを行単位の小さな塊（チャンク）に分ける。
enum KnowledgeChunker {
    /// 行頭がこれで始まる行は「区切りやすい場所」（宣言の始まり、見出し）
    private static let declarationPrefixes = [
        "func ", "class ", "struct ", "enum ", "extension ", "protocol ", "actor ", "init(", "init<", "init?(",
        "private ", "fileprivate ", "public ", "internal ", "open ", "final ", "static ", "@MainActor", "@Observable", "@Model",
        "def ", "async def ", "function ", "export ", "interface ", "type ", "const ", "let ", "var ", "#", "//", "///", "/*",
    ]

    static func chunks(source: String, text: String, maxCharacters: Int = 1200, overlapLines: Int = 2) -> [KnowledgeChunk] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
        var lines = normalized.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        guard !lines.isEmpty, lines.contains(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) else { return [] }
        let limit = max(maxCharacters, 40)

        // とても長い行は分ける（行番号は同じまま）
        var pieces: [(line: Int, text: String)] = []
        for (index, line) in lines.enumerated() {
            if line.count <= limit {
                pieces.append((index + 1, line))
            } else {
                var rest = Substring(line)
                while !rest.isEmpty {
                    pieces.append((index + 1, String(rest.prefix(limit))))
                    rest = rest.dropFirst(limit)
                }
            }
        }

        var chunks: [KnowledgeChunk] = []
        var start = 0
        while start < pieces.count {
            // まず上限まで詰める
            var end = start
            var size = 0
            while end < pieces.count {
                let added = pieces[end].text.count + (end > start ? 1 : 0)
                if size + added > limit && end > start { break }
                size += added
                end += 1
            }
            // 途中で切るなら、後ろ半分にある「区切りやすい行」の手前で切る
            if end < pieces.count {
                let minimum = start + max(1, (end - start) / 2)
                var cut = end
                for candidate in stride(from: end - 1, through: minimum, by: -1) where isBoundary(pieces[candidate].text) {
                    cut = candidate
                    break
                }
                if cut > start { end = cut }
            }
            let slice = pieces[start..<end]
            let chunkText = slice.map(\.text).joined(separator: "\n")
            if !chunkText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                chunks.append(KnowledgeChunk(id: chunks.count, source: source, startLine: slice.first!.line,
                                             endLine: slice.last!.line, text: chunkText))
            }
            if end >= pieces.count { break }
            // 次のチャンクは少し重ねて始める（必ず前に進む）
            start = max(start + 1, end - max(0, overlapLines))
        }
        return chunks
    }

    private static func isBoundary(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return true }
        // 字下げされていない宣言・見出し
        guard line.first.map({ !$0.isWhitespace }) ?? false else { return false }
        return declarationPrefixes.contains { trimmed.hasPrefix($0) }
    }
}

/// BM25 で検索する参考資料のインデックス。
struct KnowledgeIndex: Codable, Sendable {
    private(set) var chunks: [KnowledgeChunk] = []
    /// チャンクごとの語の出現回数
    private var termFrequencies: [[String: Int]] = []
    /// チャンクごとの語の数
    private var lengths: [Int] = []
    /// 語 → その語を含むチャンクの数
    private var documentFrequencies: [String: Int] = [:]
    private var nextID = 0

    static let k1 = 1.2
    static let b = 0.75

    init() {}

    var sources: [String] {
        var seen = Set<String>()
        var result: [String] = []
        for chunk in chunks where seen.insert(chunk.source).inserted { result.append(chunk.source) }
        return result
    }

    var chunkCount: Int { chunks.count }

    /// 追加する（同じ source があれば置き換える）。
    mutating func add(source: String, text: String) {
        remove(source: source)
        for var chunk in KnowledgeChunker.chunks(source: source, text: text) {
            chunk.id = nextID
            nextID += 1
            let terms = KnowledgeTokenizer.terms(chunk.text)
            var frequencies: [String: Int] = [:]
            for term in terms { frequencies[term, default: 0] += 1 }
            for term in frequencies.keys { documentFrequencies[term, default: 0] += 1 }
            chunks.append(chunk)
            termFrequencies.append(frequencies)
            lengths.append(terms.count)
        }
    }

    mutating func remove(source: String) {
        var keepChunks: [KnowledgeChunk] = []
        var keepFrequencies: [[String: Int]] = []
        var keepLengths: [Int] = []
        for index in chunks.indices {
            if chunks[index].source == source {
                for term in termFrequencies[index].keys {
                    let count = (documentFrequencies[term] ?? 1) - 1
                    documentFrequencies[term] = count > 0 ? count : nil
                }
            } else {
                keepChunks.append(chunks[index])
                keepFrequencies.append(termFrequencies[index])
                keepLengths.append(lengths[index])
            }
        }
        chunks = keepChunks
        termFrequencies = keepFrequencies
        lengths = keepLengths
    }

    /// BM25（k1 = 1.2, b = 0.75）。スコア 0 のものは返さない。
    func search(_ query: String, limit: Int = 4) -> [KnowledgeSearchResult] {
        guard !chunks.isEmpty, limit > 0 else { return [] }
        var queryTerms: [String] = []
        var seen = Set<String>()
        for term in KnowledgeTokenizer.terms(query) where seen.insert(term).inserted { queryTerms.append(term) }
        guard !queryTerms.isEmpty else { return [] }

        let count = Double(chunks.count)
        let averageLength = max(1, Double(lengths.reduce(0, +)) / count)
        var idf: [String: Double] = [:]
        for term in queryTerms {
            let frequency = Double(documentFrequencies[term] ?? 0)
            guard frequency > 0 else { continue }
            idf[term] = log(1 + (count - frequency + 0.5) / (frequency + 0.5))
        }
        guard !idf.isEmpty else { return [] }

        var scored: [(index: Int, score: Double)] = []
        for index in chunks.indices {
            let frequencies = termFrequencies[index]
            let length = Double(lengths[index])
            var score = 0.0
            for (term, weight) in idf {
                guard let tf = frequencies[term] else { continue }
                let value = Double(tf)
                score += weight * (value * (Self.k1 + 1)) / (value + Self.k1 * (1 - Self.b + Self.b * length / averageLength))
            }
            if score > 0 { scored.append((index, score)) }
        }
        scored.sort { $0.score != $1.score ? $0.score > $1.score : chunks[$0.index].id < chunks[$1.index].id }
        return scored.prefix(limit).map { KnowledgeSearchResult(chunk: chunks[$0.index], score: $0.score) }
    }
}

/// 検索結果をプロンプトに入れる文章にする。
enum KnowledgePromptBuilder {
    static func contextBlock(results: [KnowledgeSearchResult], maxCharacters: Int) -> String {
        guard !results.isEmpty, maxCharacters > 0 else { return "" }
        let header = L("以下は参考資料の抜粋です。質問に関係する場合だけ使ってください。",
                       "Here are excerpts from reference files. Use them only if they are relevant.") + "\n\n"
        var output = header
        for (number, result) in results.enumerated() {
            let chunk = result.chunk
            let fence = String(repeating: "`", count: max(3, longestBacktickRun(chunk.text) + 1))
            let title = "[\(number + 1)] \(chunk.source) (L\(chunk.startLine)-\(chunk.endLine))\n"
            var body = chunk.text
            var entry = title + fence + "\n" + body + "\n" + fence + "\n\n"
            if output.count + entry.count > maxCharacters {
                if number > 0 { break }
                // 1 件目も入りきらないときは、本文を切り詰める
                let room = maxCharacters - output.count - title.count - fence.count * 2 - 6
                guard room > 20 else { return "" }
                body = String(body.prefix(room)) + "…"
                entry = title + fence + "\n" + body + "\n" + fence + "\n\n"
            }
            output += entry
        }
        if output == header { return "" }
        while output.hasSuffix("\n") { output.removeLast() }
        return output
    }

    private static func longestBacktickRun(_ text: String) -> Int {
        var longest = 0
        var current = 0
        for character in text {
            if character == "`" {
                current += 1
                longest = max(longest, current)
            } else {
                current = 0
            }
        }
        return longest
    }
}
