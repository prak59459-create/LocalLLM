import Foundation
import XCTest
@testable import LocalLLMCore

/// 4・5 ビットの置き方（GGUFGroupQuant）
final class GGUFGroupQuantTests: XCTestCase {
    /// 元の形式ごとの置き方: 4 ビット以下は 4 ビット、5 ビットは 5 ビット、6 ビット以上は 8 ビット（nil）
    func testStorageFollowsTheSourceBits() {
        XCTAssertEqual(GGUFGroupQuant.storage(for: .q4_0), .q4Groups)
        XCTAssertEqual(GGUFGroupQuant.storage(for: .q4_K), .q4Groups)
        XCTAssertEqual(GGUFGroupQuant.storage(for: .iq4_XS), .q4Groups)
        XCTAssertEqual(GGUFGroupQuant.storage(for: .q3_K), .q4Groups)
        XCTAssertEqual(GGUFGroupQuant.storage(for: .q5_0), .q5Groups)
        XCTAssertEqual(GGUFGroupQuant.storage(for: .q5_K), .q5Groups)
        XCTAssertNil(GGUFGroupQuant.storage(for: .q6_K))
        XCTAssertNil(GGUFGroupQuant.storage(for: .q8_0))
        XCTAssertNil(GGUFGroupQuant.storage(for: .f16))
        XCTAssertNil(GGUFGroupQuant.storage(for: .f32))
    }

    func testHalfRoundTrip() {
        for value: Float in [0, 1, -1, 0.5, 65504, -2.25, 1e-5, 3.0e-8, 0.1, 1234.5678] {
            let back = GGUFQuant.half(GGUFGroupQuant.half(value))
            XCTAssertEqual(back, value, accuracy: max(abs(value) * 1e-3, 6e-8), "\(value)")
        }
        XCTAssertEqual(GGUFGroupQuant.half(1), 0x3C00)
        XCTAssertEqual(GGUFGroupQuant.half(-2), 0xC000)
        XCTAssertEqual(GGUFGroupQuant.half(.infinity), 0x7C00)
        XCTAssertEqual(GGUFGroupQuant.half(1e9), 0x7C00)
        // 0.1 に最も近い半精度は 0x2E66
        XCTAssertEqual(GGUFGroupQuant.half(0.1), 0x2E66)
    }

    private struct Packed {
        var nibbles: [UInt8]
        var high: [UInt8]
        var params: [UInt16]
    }

    private func pack(_ type: GGUFTensorType, _ bytes: Data, columns: Int, storage: GGUFMatrix.Storage) throws -> Packed {
        let bits = GGUFGroupQuant.bits(storage)
        var packed = Packed(nibbles: [UInt8](repeating: 0, count: columns / 2),
                            high: [UInt8](repeating: 0, count: bits == 5 ? columns / 8 : 0),
                            params: [UInt16](repeating: 0, count: columns / 32 * 2))
        let scratch = GGUFGroupQuant.Scratch(columns: columns)
        try bytes.withUnsafeBytes { raw in
            try packed.nibbles.withUnsafeMutableBufferPointer { nibbles in
                try packed.high.withUnsafeMutableBufferPointer { high in
                    try packed.params.withUnsafeMutableBufferPointer { params in
                        let target = GGUFGroupQuant.RowTarget(nibbles: nibbles.baseAddress!,
                                                              high: bits == 5 ? high.baseAddress : nil,
                                                              params: params.baseAddress!)
                        try GGUFGroupQuant.packRow(type: type, source: raw.baseAddress!, columns: columns, bits: bits,
                                                   into: target, scratch: scratch)
                    }
                }
            }
        }
        return packed
    }

    private func withMatrix<T>(_ packed: inout Packed, rows: Int, columns: Int, storage: GGUFMatrix.Storage,
                               _ body: (GGUFMatrix) throws -> T) rethrows -> T {
        try packed.nibbles.withUnsafeBytes { nibbles in
            try packed.high.withUnsafeBytes { high in
                try packed.params.withUnsafeBytes { params in
                    let matrix = GGUFMatrix(rows: rows, columns: columns, storage: storage, data: nibbles.baseAddress!, scales: nil,
                                            highBits: storage == .q5Groups ? high.baseAddress : nil,
                                            groupParams: params.baseAddress!)
                    return try body(matrix)
                }
            }
        }
    }

    /// 本物の量子化ブロック（gguf パッケージで作ったもの）を 4・5 ビットに置き直しても、ほぼ同じ値に戻る。
    /// Q4_0・Q4_1・Q5_0・Q5_1・Q4_K・Q5_K は目盛りをそのまま写すので、ずれは倍率を半精度にした分だけ
    func testRepackingRealBlocks() throws {
        let entries = try XCTUnwrap(try GGUFFixtures.json("quant_blocks.json")["entries"] as? [[String: Any]])
        let exact: Set<GGUFTensorType> = [.q4_0, .q4_1, .q5_0, .q5_1, .q4_K, .q5_K]
        var checked: [String] = []
        for entry in entries {
            let name = entry["name"] as? String ?? "?"
            let type = try XCTUnwrap(GGUFTensorType(rawValue: UInt32(try XCTUnwrap(entry["type"] as? Int))), name)
            guard let storage = GGUFGroupQuant.storage(for: type) else { continue }
            let count = try XCTUnwrap(entry["count"] as? Int)
            let bytes = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(entry["bytes"] as? String)))
            let expected = try GGUFFixtures.floats(entry["expected"])
            var packed = try pack(type, bytes, columns: count, storage: storage)
            var output = [Float](repeating: .nan, count: count)
            withMatrix(&packed, rows: 1, columns: count, storage: storage) { matrix in
                output.withUnsafeMutableBufferPointer { GGUFGroupQuant.dequantizeRow(matrix, row: 0, into: $0.baseAddress!) }
            }
            for group in 0..<(count / 32) {
                let slice = expected[(group * 32)..<((group + 1) * 32)]
                // 特別な値（NaN・無限大）を入れたブロックは、本物の重みには無いので見ない
                guard slice.allSatisfy(\.isFinite) else { continue }
                // 半精度に入らない大きさの値を入れたブロックも、本物の重みには無いので見ない
                guard slice.allSatisfy({ abs($0) < 60_000 }) else { continue }
                let low = slice.min()!, high = slice.max()!
                let peak = max(abs(low), abs(high))
                // そのまま写せる形式: 半精度の丸め分。ほかの形式: 目盛りの半分まで
                let tolerance = exact.contains(type) ? peak * 2e-3 + 1e-7 : (high - low) / (storage == .q5Groups ? 62 : 30) * 1.01 + peak * 2e-3 + 1e-7
                for index in (group * 32)..<((group + 1) * 32) {
                    XCTAssertEqual(output[index], expected[index], accuracy: tolerance, "\(name)[\(index)]")
                }
            }
            checked.append(name)
        }
        print("GGUF group repack checked: \(checked)")
        XCTAssertTrue(checked.contains { $0.lowercased().contains("q4_k") }, "\(checked)")
        XCTAssertTrue(checked.contains { $0.lowercased().contains("q5_k") }, "\(checked)")
    }

    /// 線形層: 4・5 ビットの行列の掛け算が、Float に戻してから掛けたものと同じになる（何行も・何トークンも・一部の行だけでも）
    func testLinearMatchesDequantizedProduct() throws {
        let rows = 37, columns = 96
        var generator = SystemRandomNumberGenerator()
        for storage in [GGUFMatrix.Storage.q4Groups, .q5Groups] {
            let levels = storage == .q5Groups ? 32 : 16
            var packed = Packed(nibbles: [], high: [], params: [])
            for _ in 0..<(rows * columns / 2) { packed.nibbles.append(UInt8.random(in: 0...255, using: &generator)) }
            if storage == .q5Groups {
                for _ in 0..<(rows * columns / 8) { packed.high.append(UInt8.random(in: 0...255, using: &generator)) }
            }
            for _ in 0..<(rows * columns / 32) {
                packed.params.append(GGUFGroupQuant.half(Float.random(in: 0.001...0.05, using: &generator)))
                packed.params.append(GGUFGroupQuant.half(Float.random(in: -0.4...0.1, using: &generator)))
            }
            XCTAssertEqual(levels > 0, true)
            let count = 3
            let input = (0..<(count * columns)).map { _ in Float.random(in: -1...1, using: &generator) }
            let workers = GGUFWorkers(chunkCount: 4, widestRow: columns)
            try withMatrix(&packed, rows: rows, columns: columns, storage: storage) { matrix in
                var dense = [Float](repeating: 0, count: rows * columns)
                dense.withUnsafeMutableBufferPointer { buffer in
                    for r in 0..<rows { GGUFGroupQuant.dequantizeRow(matrix, row: r, into: buffer.baseAddress! + r * columns) }
                }
                var output = [Float](repeating: .nan, count: count * rows)
                input.withUnsafeBufferPointer { x in
                    output.withUnsafeMutableBufferPointer { y in
                        GGUFKernels.linear(matrix, input: x.baseAddress!, count: count, output: y.baseAddress!, workers: workers)
                    }
                }
                for n in 0..<count {
                    for r in 0..<rows {
                        var sum: Float = 0
                        for c in 0..<columns { sum += dense[r * columns + c] * input[n * columns + c] }
                        XCTAssertEqual(output[n * rows + r], sum, accuracy: 1e-3 * max(1, abs(sum)), "\(storage) n\(n) r\(r)")
                    }
                }
                // 一部の行だけ（q・k・v が 1 つの行列のときに使う）
                let part = matrix.slice(5..<17)
                var partOutput = [Float](repeating: .nan, count: part.rows)
                input.withUnsafeBufferPointer { x in
                    partOutput.withUnsafeMutableBufferPointer { y in
                        GGUFKernels.linear(part, input: x.baseAddress!, count: 1, output: y.baseAddress!, workers: workers)
                    }
                }
                for r in 0..<part.rows {
                    XCTAssertEqual(partOutput[r], output[5 + r], accuracy: 1e-4 * max(1, abs(output[5 + r])), "\(storage) slice \(r)")
                }
                // 埋め込みの 1 行
                var row = [Float](repeating: 0, count: columns)
                row.withUnsafeMutableBufferPointer { GGUFKernels.embeddingRow(matrix, row: 7, scale: 2, into: $0.baseAddress!) }
                for c in 0..<columns { XCTAssertEqual(row[c], 2 * dense[7 * columns + c], accuracy: 1e-6) }
            }
        }
    }
}
