import Foundation

/// 先読み（MTP）で 1 回に予想するトークンの数を、当たり方と、実際に測った時間から決める。
///
/// 位置ごとの当たる割合（直前まで当たったときの）と、「n 個をまとめて確かめる時間」「予想 1 つの時間」を覚えておき、
/// 1 秒あたりに進めるトークン数の見積もりがいちばん大きい数を選ぶ。予想しない（0）ほうが速ければ 0 にする。
/// 端末やモデルで手間の比が違うので、決め打ちにせず、動かしながら測った値を使う。
struct DraftPlanner {
    let maximum: Int
    /// i 番目の予想が当たる割合（i − 1 番目まで当たったとき）
    private(set) var rates: [Double]
    /// n 個をまとめて確かめるのにかかった時間（秒。添字が n。0 はまだ測っていない）
    private(set) var verifySeconds: [Double]
    /// 予想 1 つにかかった時間（秒。0 はまだ測っていない）
    private(set) var draftSeconds: Double = 0
    /// 次の判断まで、決めた数のまま進む回数など
    private var roundsSinceExplore = 0
    private var roundsSincePlain = 0

    /// 予想しない回が続いても、この回数ごとに 1 つ予想して、当たり方を確かめ直す
    static let exploreEvery = 16
    /// 予想し続けても、この回数ごとに予想しない回を入れて、ふつうの 1 回の時間を測り直す
    static let plainEvery = 48
    /// 新しく測った値の重み（時間が今の見積もりより長いとき）
    static let smoothing = 0.2
    /// 時間が今の見積もりより短いときの重み。最初の 1 回（重みを初めて読む・GPU が温まっていない）などの遅い値を、すぐに直す。
    /// 遅くなるほうはゆっくり追う（たまたま割り込まれた 1 回で、予想をやめてしまわないように）
    static let fasterSmoothing = 0.8

    init(maximum: Int) {
        self.maximum = max(0, maximum)
        rates = Array(repeating: 0.5, count: self.maximum)
        verifySeconds = Array(repeating: 0, count: self.maximum + 2)
    }

    /// n 個を確かめる時間（測っていなければ、測った近い数から見積もる）
    func verifyCost(_ count: Int) -> Double {
        let n = Swift.max(1, Swift.min(count, verifySeconds.count - 1))
        if verifySeconds[n] > 0 { return verifySeconds[n] }
        // 1 個の時間を、測ったものから推す（まとめても重みを読むのは 1 回なので、1 つ増えるごとに 15% 増えると見る）
        var one: Double = verifySeconds[1]
        if one <= 0 {
            for m in 2..<verifySeconds.count where verifySeconds[m] > 0 {
                one = verifySeconds[m] / (1 + 0.15 * Double(m - 1))
                break
            }
        }
        if one <= 0 { one = 1 }
        return one * (1 + 0.15 * Double(n - 1))
    }

    /// 予想 1 つの時間（測っていなければ、ふつうの 1 回の 1 割と見る）
    var draftCost: Double { draftSeconds > 0 ? draftSeconds : 0.1 * verifyCost(1) }

    /// k 個予想したときの、1 秒あたりに進めるトークン数の見積もり
    func throughput(drafts k: Int) -> Double {
        var expected: Double = 1
        var chance: Double = 1
        for index in 0..<Swift.min(k, rates.count) {
            chance *= rates[index]
            expected += chance
        }
        return expected / (Double(k) * draftCost + verifyCost(k + 1))
    }

    /// 今使う予想の数（0 なら予想しない）
    mutating func next() -> Int {
        guard maximum > 0 else { return 0 }
        // ふつうの 1 回の時間がまだ分からなければ、まず測る
        if verifySeconds[1] <= 0 { return 0 }
        var best = 0
        for k in 1...maximum where throughput(drafts: k) > throughput(drafts: best) { best = k }
        if best == 0 {
            roundsSinceExplore += 1
            if roundsSinceExplore >= Self.exploreEvery {
                roundsSinceExplore = 0
                return 1
            }
            return 0
        }
        roundsSinceExplore = 0
        roundsSincePlain += 1
        if roundsSincePlain >= Self.plainEvery {
            roundsSincePlain = 0
            return 0
        }
        return best
    }

    /// drafted 個予想して、accepted 個当たった
    mutating func record(drafted: Int, accepted: Int) {
        for index in 0..<Swift.min(drafted, maximum) {
            guard index <= accepted else { break }
            let hit: Double = index < accepted ? 1 : 0
            rates[index] = rates[index] * (1 - Self.smoothing) + hit * Self.smoothing
        }
    }

    /// 測った時間を覚える: drafted 個の予想に draftTime 秒、verified 個の確かめに verifyTime 秒
    mutating func recordTimes(drafted: Int, draftTime: Double, verified: Int, verifyTime: Double) {
        if drafted > 0, draftTime > 0 {
            draftSeconds = Self.blend(draftSeconds, draftTime / Double(drafted))
        }
        guard verified >= 1, verified < verifySeconds.count, verifyTime > 0 else { return }
        verifySeconds[verified] = Self.blend(verifySeconds[verified], verifyTime)
    }

    /// 今の見積もり old（0 はまだ無い）に、新しく測った value を混ぜる
    private static func blend(_ old: Double, _ value: Double) -> Double {
        guard old > 0 else { return value }
        let weight: Double = value < old ? fasterSmoothing : smoothing
        return old * (1 - weight) + value * weight
    }
}
