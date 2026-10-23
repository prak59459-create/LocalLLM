import Foundation

/// 参考資料（自分のコードや文書）を行単位で切り出した小さな塊。
struct KnowledgeChunk: Codable, Hashable, Identifiable, Sendable {
    /// 番号。KnowledgeChunker が返す値はファイル内の通し番号（0 始まり）、
    /// KnowledgeIndex に入れたものはインデックス全体で重ならない番号。
    var id: Int
    /// ファイル名など
    var source: String
    /// 1 始まり
    var startLine: Int
    /// この行まで含む（1 始まり）
    var endLine: Int
    /// 元のテキストそのまま（行は "\n" でつなぐ）
    var text: String
}

/// 検索結果 1 件。
struct KnowledgeSearchResult: Hashable, Sendable {
    var chunk: KnowledgeChunk
    /// BM25 のスコア（大きいほど関係が深い。常に 0 より大きい）
    var score: Double
}
