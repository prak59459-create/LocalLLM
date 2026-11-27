import XCTest
@testable import LocalLLMCore

/// 型チェックを速くするために書き直した小さな関数が、前と同じ結果を返すか
final class ByteFormatTests: XCTestCase {
    func testFormatsLikeTheFilesApp() {
        XCTAssertEqual(ByteFormat.string(0), "0 B")
        XCTAssertEqual(ByteFormat.string(-5), "0 B")
        XCTAssertEqual(ByteFormat.string(999), "999 B")
        XCTAssertEqual(ByteFormat.string(1_000), "1.00 KB")
        XCTAssertEqual(ByteFormat.string(12_345), "12.3 KB")
        XCTAssertEqual(ByteFormat.string(278_000_000), "278 MB")
        XCTAssertEqual(ByteFormat.string(1_930_000_000), "1.93 GB")
        XCTAssertEqual(ByteFormat.string(5_000_000_000_000_000), "5000 TB")
    }

    /// 2^-53 を掛けるのは 2^53 で割るのと同じ値になる
    func testUniformRandomMatchesDivision() {
        var random = MiniRandom(seed: 42)
        var generator = SplitMix64(seed: 42)
        for _ in 0..<1000 {
            let value = random.nextUniform()
            XCTAssertEqual(value, Double(generator.next() >> 11) / Double(UInt64(1) << 53))
            XCTAssertTrue(value >= 0 && value < 1)
        }
    }
}
