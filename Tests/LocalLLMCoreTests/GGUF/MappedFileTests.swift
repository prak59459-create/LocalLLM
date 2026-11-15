import Foundation
import XCTest
@testable import LocalLLMCore

final class MappedFileTests: XCTestCase {
    func testUsageCountsMappedAndResidentBytes() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mapped-\(UUID().uuidString).bin")
        try Data(repeating: 7, count: 1 << 20).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let before = MappedFile.usage()
        var file: MappedFile? = try MappedFile(url: url)
        // 読むと RAM に載る
        var sum = 0
        for offset in stride(from: 0, to: 1 << 20, by: 4096) { sum += Int(file!.pointer.load(fromByteOffset: offset, as: UInt8.self)) }
        XCTAssertEqual(sum, 7 * 256)
        let during = MappedFile.usage()
        XCTAssertEqual(during.mapped - before.mapped, 1 << 20)
        XCTAssertGreaterThan(during.resident, before.resident)
        XCTAssertLessThanOrEqual(file!.residentBytes() ?? 0, 1 << 20)
        file = nil
        XCTAssertEqual(MappedFile.usage().mapped, before.mapped)
    }
}
