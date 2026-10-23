import Foundation

/// トークナイザの最小インターフェース。
/// 実装は `HFTokenizer`（tokenizer.json を読む純 Swift 実装）。
/// テストではモックに差し替えられるよう、生成ループやプロンプト組み立てはこのプロトコルにだけ依存する。
protocol TextTokenizer: AnyObject {
    /// テキストをトークン ID 列に変換する。
    /// - Parameter addSpecialTokens: true なら post_processor（BOS 付与など）を適用する。
    ///   チャットテンプレートを描画した文字列には BOS などが既に含まれるので false を使う。
    func encode(_ text: String, addSpecialTokens: Bool) -> [Int]

    /// トークン ID 列を文字列に戻す。
    /// - Parameter skipSpecialTokens: true なら special な追加トークン（<|im_end|> など）を出力しない。
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String

    /// トークン文字列（例: "<|im_end|>"）から ID を引く。追加トークンも含む。
    func tokenID(for token: String) -> Int?

    /// ID からトークン文字列を引く（デコード前の生の表記）。
    func token(for id: Int) -> String?
}
