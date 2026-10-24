import Foundation

/// 推論でメモリを使い切らない（iOS に強制終了されない）ための見積もり。端末に依存しない計算だけを置く。
///
/// iOS はアプリがメモリの上限を超えると、何の知らせもなくアプリを終了する。なので:
/// - 残りが少なくなったら、1 回の計算を小さくする（プロンプトを少しずつ読む）
/// - さらに少なくなったら、生成を自分で止める（落ちる前に止めて、続きはまた頼めるようにする）
struct MemoryBudget: Equatable, Sendable {
    /// iOS が強制終了するまでの残り（バイト）
    var available: Int64
    /// アプリが使える上限の目安（使用中 + 残り、バイト）
    var appLimit: Int64

    private static let megabyte: Int64 = 1 << 20

    /// これを下回ったら、生成を止める
    var critical: Int64 { max(160 * Self.megabyte, appLimit / 16) }
    /// これを下回ったら、少しずつ計算する
    var low: Int64 { max(320 * Self.megabyte, appLimit / 8) }

    var isCritical: Bool { available < critical }
    var isLow: Bool { available < low }

    /// 1 回の推論の出力（logits）に使ってよいバイト数
    var logitsAllowance: Int64 { max(0, (available - critical) / 4) }

    /// プロンプトを流し込むときの 1 回あたりのトークン数。出力の logits [n, 語彙数] が使ってよい量に収まるようにする。
    func prefillChunk(requested: Int, vocabSize: Int, bytesPerLogit: Int) -> Int {
        let requested = max(1, requested)
        // 出力と、それを読み出すときの作業用の分
        let perToken = Int64(max(1, vocabSize)) * Int64(max(1, bytesPerLogit)) * 2
        let byMemory = Int(min(Int64(Int.max), logitsAllowance / perToken))
        var chunk = min(requested, max(1, byMemory))
        if isLow { chunk = min(chunk, 16) }
        return max(1, chunk)
    }

    /// 毎回系列全体を入力するモデル（ステートレス）で、安全に扱える系列の長さ。
    /// このモデルは 1 回ごとに [系列の長さ, 語彙数] の logits を出すので、長いほどメモリを使う。
    func statelessTokenLimit(vocabSize: Int, bytesPerLogit: Int) -> Int {
        let perToken = Int64(max(1, vocabSize)) * Int64(max(1, bytesPerLogit)) * 3
        return Int(min(Int64(Int.max), logitsAllowance / perToken))
    }
}
