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

/// 計算 1 回ごとに、残りのメモリがどれだけ減ったかを覚えておき、「次の 1 回で尽きないか」を見積もる。
///
/// 残りが今は十分でも、1 回の計算で一気に数百 MB 使うことがある（GPU が形ごとに作る作業用のメモリなど）。
/// 減り方を見ておけば、iOS に強制終了される前に止められる。
struct MemoryTrend: Equatable, Sendable {
    /// 1 回で減ったいちばん大きな量（バイト）
    private(set) var largestDrop: Int64 = 0
    private var last: Int64?
    private var samples = 0

    /// 最初のこの回数ぶんの減り方は数えない（KV キャッシュを作る・初回の準備など、1 度きりの分が大きいので）
    static let warmUpSamples = 2

    mutating func reset() {
        largestDrop = 0
        last = nil
        samples = 0
    }

    /// 計算の前の残りを記録する
    mutating func record(available: Int64) {
        defer {
            last = available
            samples += 1
        }
        guard let last, samples >= Self.warmUpSamples else { return }
        largestDrop = max(largestDrop, last - available)
    }

    /// 次の 1 回で見込む減り（いちばん大きかった減りの 1.5 倍。ただしアプリの上限の 1/8 まで）
    func expectedDrop(appLimit: Int64) -> Int64 {
        min(largestDrop, max(0, appLimit / 8)) * 3 / 2
    }

    /// 次の 1 回のあとも、止めるべき量（critical）を下回らなさそうか
    func isSafe(_ budget: MemoryBudget) -> Bool {
        budget.available - expectedDrop(appLimit: budget.appLimit) >= budget.critical
    }
}

/// メモリが足りなくなって止まったあと、短くして読み直すときの長さ。
enum MemoryRecovery {
    /// 自動で読み直す回数の上限（1 回の生成あたり）
    static let maxRetries = 2
    /// これより短くはしない
    static let minimumContext = 256

    /// 次に使うコンテキストの長さ。reached（止まるまでに読めたトークン数）の 3/5 にする。
    /// ほとんど読めていなければ（メモリが最初から足りない）、今の長さの 3/5 にする。
    /// これ以上短くできなければ nil。
    static func shrunkContext(current: Int, reached: Int) -> Int? {
        let basis = reached > minimumContext ? min(current, reached) : current
        let next = max(minimumContext, basis * 3 / 5)
        return next < current ? next : nil
    }

    /// うまく最後まで生成できたあと、縮めていた上限を少し戻す（モデルの最大まで。戻しきったら nil）
    static func relaxedCap(_ cap: Int, modelMax: Int) -> Int? {
        let next = cap + max(64, cap / 4)
        return next >= modelMax ? nil : next
    }
}
