import XCTest
@testable import LocalLLMCore

/// 先読み（MTP）の予想の数の決め方
final class DraftPlannerTests: XCTestCase {
    /// 本体の 1 回 30 ms、まとめて確かめると 1 つ増えるごとに perExtra 秒、予想 1 つ draft 秒。予想はいつも当たるか、いつも外れる
    private func planner(maximum: Int = 4, perExtra: Double, draft: Double, alwaysHit: Bool) -> DraftPlanner {
        var planner = DraftPlanner(maximum: maximum)
        for n in 1...(maximum + 1) {
            planner.recordTimes(drafted: n - 1, draftTime: draft * Double(n - 1), verified: n,
                                verifyTime: 0.030 + perExtra * Double(n - 1))
        }
        for _ in 0..<100 { planner.record(drafted: maximum, accepted: alwaysHit ? maximum : 0) }
        return planner
    }

    func testMeasuresAPlainStepFirst() {
        var planner = DraftPlanner(maximum: 3)
        XCTAssertEqual(planner.next(), 0, "the plain step time is not known yet")
        planner.recordTimes(drafted: 0, draftTime: 0, verified: 1, verifyTime: 0.030)
        // 割合 0.5（最初の見積もり）・予想 1 つは 1 回の 1 割・確かめは 1 つ 15% 増し: 予想したほうが速い
        XCTAssertGreaterThan(planner.next(), 0)
    }

    func testDraftsAsManyAsAllowedWhenTheyAlwaysHitAndAreCheap() {
        var planner = planner(perExtra: 0.001, draft: 0.001, alwaysHit: true)
        XCTAssertEqual(planner.next(), 4)
        XCTAssertGreaterThan(planner.throughput(drafts: 4), 2 * planner.throughput(drafts: 0))
    }

    func testStopsDraftingWhenNothingHitsButChecksAgainSometimes() {
        var planner = planner(perExtra: 0.006, draft: 0.003, alwaysHit: false)
        var choices: [Int] = []
        for _ in 0..<(DraftPlanner.exploreEvery * 2) { choices.append(planner.next()) }
        XCTAssertEqual(choices.filter { $0 == 0 }.count, DraftPlanner.exploreEvery * 2 - 2)
        XCTAssertEqual(choices.filter { $0 == 1 }.count, 2, "tries one draft every \(DraftPlanner.exploreEvery) rounds")
    }

    /// CI の Mac の Gemma 4 E2B に近い数: 1 回 30 ms、確かめは 1 つ 3 ms 増し、予想 1 つ 2 ms、当たる割合は位置ごとに下がる
    func testRealisticCostsPickAFewDrafts() {
        var planner = DraftPlanner(maximum: 4)
        for n in 1...5 {
            planner.recordTimes(drafted: n - 1, draftTime: 0.002 * Double(n - 1), verified: n, verifyTime: 0.030 + 0.003 * Double(n - 1))
        }
        // 1 つ目は 45%、2 つ目は（1 つ目が当たったとき）30%、3 つ目以降は 20%
        let accepted = [0, 0, 1, 2, 0, 3, 1, 0, 0, 1]
        for _ in 0..<30 {
            for count in accepted { planner.record(drafted: 4, accepted: count) }
        }
        let k = planner.next()
        XCTAssertTrue((1...3).contains(k), "picked \(k)")
        XCTAssertGreaterThan(planner.throughput(drafts: k), planner.throughput(drafts: 0) * 1.1)
    }

    func testRemeasuresAPlainStepWhileDrafting() {
        var planner = planner(perExtra: 0.001, draft: 0.001, alwaysHit: true)
        var plain = 0
        for _ in 0..<DraftPlanner.plainEvery { if planner.next() == 0 { plain += 1 } }
        XCTAssertEqual(plain, 1)
    }

    /// 最初の 1 回だけ遅くても（重みを初めて読む など）、速く測れればすぐに見積もりを直す。遅くなるほうはゆっくり追う
    func testRecoversQuicklyFromASlowFirstMeasurement() {
        var planner = DraftPlanner(maximum: 2)
        planner.recordTimes(drafted: 1, draftTime: 0.030, verified: 2, verifyTime: 0.300)
        planner.recordTimes(drafted: 1, draftTime: 0.003, verified: 2, verifyTime: 0.035)
        // 速い値には 8 割の重み
        XCTAssertEqual(planner.draftCost, 0.030 * 0.2 + 0.003 * 0.8, accuracy: 1e-9)
        XCTAssertEqual(planner.verifyCost(2), 0.300 * 0.2 + 0.035 * 0.8, accuracy: 1e-9)
        for _ in 0..<2 { planner.recordTimes(drafted: 1, draftTime: 0.003, verified: 2, verifyTime: 0.035) }
        XCTAssertLessThan(planner.verifyCost(2), 0.040)
        XCTAssertLessThan(planner.draftCost, 0.004)
        // 1 回だけ遅くても、見積もりは 2 割しか動かない
        let before = planner.verifyCost(2)
        planner.recordTimes(drafted: 1, draftTime: 0.003, verified: 2, verifyTime: 0.100)
        XCTAssertEqual(planner.verifyCost(2), before * 0.8 + 0.100 * 0.2, accuracy: 1e-9)
    }

    func testEstimatesUnmeasuredBatchesFromMeasuredOnes() {
        var planner = DraftPlanner(maximum: 3)
        planner.recordTimes(drafted: 2, draftTime: 0.004, verified: 3, verifyTime: 0.039)
        // 3 個で 39 ms → 1 個は 30 ms、2 個は 34.5 ms と見る
        XCTAssertEqual(planner.verifyCost(1), 0.030, accuracy: 1e-9)
        XCTAssertEqual(planner.verifyCost(2), 0.0345, accuracy: 1e-9)
        XCTAssertEqual(planner.draftCost, 0.002, accuracy: 1e-9)
        // 新しく測った値は 2 割の重みで混ぜる
        planner.recordTimes(drafted: 2, draftTime: 0.006, verified: 3, verifyTime: 0.049)
        XCTAssertEqual(planner.verifyCost(3), 0.041, accuracy: 1e-9)
        XCTAssertEqual(planner.draftCost, 0.0022, accuracy: 1e-9)
    }
}
