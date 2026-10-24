import Foundation
import XCTest
@testable import LocalLLMCore

/// コンパイル済みモデルの小さいファイル（Tools/fetch_fixtures.sh で取得）
private enum CompiledFixture: String {
    case qwen = "finnvoorhees__coreml-Qwen2.5-0.5B-Instruct-4bit"
    case llama = "finnvoorhees__coreml-Llama-3.2-1B-Instruct-4bit"
    case anemll = "anemll__anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4"

    var folder: URL { TestFixtures.root.appendingPathComponent("compiled").appendingPathComponent(rawValue) }

    func mil() throws -> String {
        let url = folder.appendingPathComponent("model.mil")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw XCTSkip("fixture missing: \(url.path) (run Tools/fetch_fixtures.sh)")
        }
        return text
    }

    func data(_ name: String) throws -> Data {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent(name)) else {
            throw XCTSkip("fixture missing: \(name)")
        }
        return data
    }

    /// テスト用の .mlmodelc（重みは無いか、ANEMLL は RoPE の表の部分だけを本来の位置に置いた疎なファイル）
    func makeModelDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ctx-\(UUID().uuidString).mlmodelc")
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("weights"), withIntermediateDirectories: true)
        for name in ["model.mil", "coremldata.bin", "metadata.json"] {
            try data(name).write(to: directory.appendingPathComponent(name))
        }
        let parts = folder.appendingPathComponent("weight-parts")
        if let names = try? FileManager.default.contentsOfDirectory(atPath: parts.path), !names.isEmpty {
            let weights = directory.appendingPathComponent("weights/weight.bin")
            FileManager.default.createFile(atPath: weights.path, contents: nil)
            let handle = try FileHandle(forWritingTo: weights)
            for name in names {
                guard let offset = UInt64(name.replacingOccurrences(of: ".bin", with: "")) else { continue }
                try handle.seek(toOffset: offset)
                try handle.write(contentsOf: Data(contentsOf: parts.appendingPathComponent(name)))
            }
            try handle.close()
        }
        return directory
    }
}

private func count(_ needle: String, in text: String) -> Int {
    text.components(separatedBy: needle).count - 1
}

final class ProtoWireTests: XCTestCase {
    func testRoundTripOfRealDescription() throws {
        let data = try CompiledFixture.anemll.data("coremldata.bin")
        let bytes = [UInt8](data)
        let length = Int(MILBlob.readUInt64(bytes, 75))
        let description = Data(bytes[83..<(83 + length)])
        let fields = try ProtoWire.parse(description)
        XCTAssertEqual(ProtoWire.serialize(fields), description)
        XCTAssertEqual(fields.filter { $0.number == 20 }.count, 2)  // infer と prefill
    }

    func testVarints() throws {
        var bytes: [UInt8] = []
        ProtoWire.appendVarint(300, to: &bytes)
        XCTAssertEqual(bytes, [0xAC, 0x02])
        var index = 0
        XCTAssertEqual(try ProtoWire.readVarint(bytes, &index), 300)
        XCTAssertEqual(try ProtoWire.packedVarints(ProtoWire.packVarints([24, 1, 2, 131072, 64, -1])), [24, 1, 2, 131072, 64, -1])
    }
}

final class HalfAndBlobTests: XCTestCase {
    func testHalfConversion() {
        for value: Float in [0, 1, -1, 0.5, 0.333, 65504, -2.5e-5, 6.1e-5, 1e-7] {
            let back = Half.toFloat(Half.fromFloat(value))
            XCTAssertEqual(back, value, accuracy: max(abs(value) * 1e-3, 1e-7), "\(value)")
        }
        XCTAssertEqual(Half.fromFloat(1), 0x3C00)
        XCTAssertEqual(Half.fromFloat(-2), 0xC000)
        XCTAssertEqual(Half.toFloat(0x7BFF), 65504)
    }

    func testBlobFileRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("blob-\(UUID().uuidString).bin")
        let offsets = try MILBlob.write([(.float16, [0, 0.5, 1, -1]), (.float32, [3.25, -7])], to: url)
        XCTAssertEqual(offsets, [64, 192])
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        XCTAssertEqual(try MILBlob.readFloats(in: handle, at: offsets[0]), [0, 0.5, 1, -1])
        XCTAssertEqual(try MILBlob.readFloats(in: handle, at: offsets[1]), [3.25, -7])
        XCTAssertEqual(try MILBlob.entry(in: handle, at: offsets[1]).dataOffset, 256)
    }
}

final class RopeTableTests: XCTestCase {
    private func table(_ kind: RopeTable.Kind, theta: Double, rows: Int, columns: Int, layout: RopeTable.Layout) -> [Float] {
        let frequencies = RopeTable.standardFrequencies(theta: theta, columns: columns, layout: layout)
        var output: [Float] = []
        for row in 0..<rows {
            for column in 0..<columns {
                let angle = Double(row) * frequencies[column]
                // 半精度に丸める（実際の表と同じ精度）
                output.append(Half.toFloat(Half.fromFloat(Float(kind == .cos ? cos(angle) : sin(angle)))))
            }
        }
        return output
    }

    func testFitsStandardTables() throws {
        for (theta, layout) in [(1_000_000.0, RopeTable.Layout.halves), (10_000, .interleaved), (500_000, .halves)] {
            for kind in [RopeTable.Kind.sin, .cos] {
                let values = table(kind, theta: theta, rows: 1024, columns: 128, layout: layout)
                let fit = try RopeTable.fit(values, rows: 1024, columns: 128)
                XCTAssertEqual(fit.theta, theta, "\(theta) \(layout) \(kind)")
                XCTAssertLessThan(fit.maxError, 1e-3)
                // 長くした表は、本来の式と一致する
                let long = RopeTable.generate(fit, rows: 40_000)
                let expected = RopeTable.standardFrequencies(theta: theta, columns: 128, layout: layout)
                let row = 39_999
                for column in [0, 1, 17, 63, 64, 127] {
                    let angle = Double(row) * expected[column]
                    XCTAssertEqual(Double(long[row * 128 + column]), kind == .cos ? cos(angle) : sin(angle), accuracy: 1e-4)
                }
            }
        }
    }

    func testFitsTheRealAnemllTables() throws {
        let parts = CompiledFixture.anemll.folder.appendingPathComponent("weight-parts")
        guard let data = try? Data(contentsOf: parts.appendingPathComponent("291329088.bin")) else { throw XCTSkip("fixture missing") }
        let bytes = [UInt8](data)
        let values = stride(from: 64, to: bytes.count, by: 2).map { Half.toFloat(UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8) }
        let fit = try RopeTable.fit(values, rows: 1024, columns: 128)
        XCTAssertEqual(fit.kind, .sin)
        XCTAssertEqual(fit.theta, 1_000_000)
    }

    func testRejectsTablesThatAreNotRope() {
        let random = (0..<(64 * 32)).map { _ in Float.random(in: -1...1) }
        XCTAssertThrowsError(try RopeTable.fit(random, rows: 64, columns: 32))
        var almost = table(.cos, theta: 10_000, rows: 64, columns: 32, layout: .halves)
        almost[40 * 32 + 3] = 0.1
        XCTAssertThrowsError(try RopeTable.fit(almost, rows: 64, columns: 32))
    }
}

final class ContextResizerTests: XCTestCase {
    func testAnalyzesFlexibleStatefulModels() throws {
        let qwen = try ContextResizer.analyze(mil: CompiledFixture.qwen.mil())
        XCTAssertEqual(qwen.context, 2048)
        XCTAssertEqual(qwen.stateShapes, [[24, 1, 2, 2048, 64], [24, 1, 2, 2048, 64]])
        XCTAssertEqual(qwen.axis, 3)
        XCTAssertFalse(qwen.fixedShapes)
        XCTAssertTrue(qwen.ropeTables.isEmpty)
        XCTAssertEqual(qwen.kvBytesPerToken, 24 * 2 * 64 * 2 * 2)

        let llama = try ContextResizer.analyze(mil: CompiledFixture.llama.mil())
        XCTAssertEqual(llama.context, 2048)
        XCTAssertTrue(llama.ropeTables.isEmpty)
    }

    func testRewritesQwenWithoutTouchingWeights() throws {
        let mil = try CompiledFixture.qwen.mil()
        let analysis = try ContextResizer.analyze(mil: mil)
        let result = try ContextResizer.rewriteMIL(mil, analysis: analysis, newContext: 131_072)
        XCTAssertEqual(count("2048", in: result.text.replacingOccurrences(of: #"uint64\(\d+\)"#, with: "", options: .regularExpression)), 0)
        XCTAssertTrue(result.text.contains("state<tensor<fp16, [24, 1, 2, 131072, 64]>> key_cache"))
        XCTAssertTrue(result.text.contains("{\"input_ids\", [[1, 1], [1, 131072]]}"))
        XCTAssertTrue(result.text.contains("[1, 1, 2, 131072, 64]"))
        // 重みの行は変わらない
        let blobLines = mil.components(separatedBy: "\n").filter { $0.contains("BLOBFILE") }
        let newBlobLines = result.text.components(separatedBy: "\n").filter { $0.contains("BLOBFILE") }
        XCTAssertEqual(blobLines, newBlobLines)
        let again = try ContextResizer.analyze(mil: result.text)
        XCTAssertEqual(again.context, 131_072)
    }

    func testLlamaHiddenSizeIsNotMistakenForContext() throws {
        // Llama 3.2 1B は hidden size も 2048 なので、KV キャッシュに関係するものだけを変える
        let mil = try CompiledFixture.llama.mil()
        let analysis = try ContextResizer.analyze(mil: mil)
        let result = try ContextResizer.rewriteMIL(mil, analysis: analysis, newContext: 8192)
        XCTAssertEqual(count("[1, ?, 2048]", in: result.text), count("[1, ?, 2048]", in: mil))
        XCTAssertEqual(count("[2048, 2048]", in: result.text), count("[2048, 2048]", in: mil))
        XCTAssertEqual(count("[8192, 2048]", in: result.text), count("[8192, 2048]", in: mil))
        XCTAssertEqual(count("[16, 1, 8, 2048, 64]", in: result.text), 0)
        XCTAssertTrue(result.text.contains("state<tensor<fp16, [16, 1, 8, 8192, 64]>> key_cache"))
        XCTAssertEqual(try ContextResizer.analyze(mil: result.text).context, 8192)
    }

    func testRewritesAnemllFixedShapes() throws {
        let mil = try CompiledFixture.anemll.mil()
        let analysis = try ContextResizer.analyze(mil: mil)
        XCTAssertEqual(analysis.context, 512)
        XCTAssertEqual(analysis.axis, 2)
        XCTAssertTrue(analysis.fixedShapes)
        XCTAssertEqual(analysis.functions, ["infer", "prefill"])
        // infer と prefill がそれぞれ同じ 2 つの表（sin / cos）を指している
        XCTAssertEqual(analysis.ropeTables.count, 4)
        XCTAssertEqual(Set(analysis.ropeTables.map(\.rows)), [1024])
        XCTAssertEqual(Set(analysis.ropeTables.map(\.offset)), [291_329_088, 291_591_296])

        let redirects = Dictionary(uniqueKeysWithValues: analysis.ropeTables.enumerated().map { index, table in
            (table.variable, (path: "@model_path/weights/context_rope.bin", offset: UInt64(64 + index * 128), rows: 4096))
        })
        let result = try ContextResizer.rewriteMIL(mil, analysis: analysis, newContext: 4096, ropeRedirects: redirects)
        let withoutOffsets = result.text.replacingOccurrences(of: #"uint64\(\d+\)"#, with: "", options: .regularExpression)
        XCTAssertNil(withoutOffsets.range(of: #"(?<![\d.])512(?!\d)"#, options: .regularExpression))
        XCTAssertTrue(result.text.contains("tensor<fp16, [1, 1, 64, 4096]> causal_mask"))
        XCTAssertTrue(result.text.contains("state<tensor<fp16, [56, 8, 4096, 128]>> model_model_kv_cache_0"))
        XCTAssertTrue(result.text.contains("tensor<fp16, [1, 4096, 128]>(BLOBFILE(path = string(\"@model_path/weights/context_rope.bin\")"))
        XCTAssertEqual(try ContextResizer.analyze(mil: result.text).context, 4096)
    }

    func testWidensRopeIndicesForVeryLongContexts() throws {
        let mil = try CompiledFixture.anemll.mil()
        let analysis = try ContextResizer.analyze(mil: mil)
        let redirects = Dictionary(uniqueKeysWithValues: analysis.ropeTables.map {
            ($0.variable, (path: "@model_path/weights/context_rope.bin", offset: UInt64(64), rows: 65_536))
        })
        let result = try ContextResizer.rewriteMIL(mil, analysis: analysis, newContext: 65_536, ropeRedirects: redirects)
        XCTAssertFalse(result.text.contains("tensor<int16, [1]> select_0_to_int16 = cast("))
        XCTAssertTrue(result.text.contains("tensor<int32, [1]> select_0_to_int16 = cast(dtype = select_0_to_int16_dtype_0_ctx_int32"))
        XCTAssertTrue(result.text.contains("string select_0_to_uint16_dtype_0_ctx_int32 = const()"))
    }

    func testDescriptionAndMetadataAreRewritten() throws {
        let analysis = try ContextResizer.analyze(mil: CompiledFixture.qwen.mil())
        let description = try ContextResizer.rewriteDescriptionFile(try CompiledFixture.qwen.data("coremldata.bin"),
                                                                    analysis: analysis, newContext: 32_768)
        let bytes = [UInt8](description)
        let length = Int(MILBlob.readUInt64(bytes, 75))
        XCTAssertEqual(83 + length + 16, bytes.count)
        let fields = try ProtoWire.parse(Data(bytes[83..<(83 + length)]))
        let state = try XCTUnwrap(fields.first { $0.number == 13 }?.bytes)
        let feature = try ProtoWire.parse(state)
        let type = try ProtoWire.parse(XCTUnwrap(feature.first { $0.number == 3 }?.bytes))
        let stateType = try ProtoWire.parse(XCTUnwrap(type.first { $0.number == 8 }?.bytes))
        let array = try ProtoWire.parse(XCTUnwrap(stateType.first { $0.number == 1 }?.bytes))
        XCTAssertEqual(try ProtoWire.packedVarints(XCTUnwrap(array.first { $0.number == 1 }?.bytes)), [24, 1, 2, 32_768, 64])

        let metadata = try ContextResizer.rewriteMetadata(try CompiledFixture.qwen.data("metadata.json"), analysis: analysis, newContext: 32_768)
        let parsed = try CoreMLMetadata.parse(metadata)
        XCTAssertEqual(parsed.states.first?.shape, [24, 1, 2, 32_768, 64])
        XCTAssertEqual(parsed.interface.maxContext, 32_768)
    }

    func testResizeAndRestoreAnemllFolder() throws {
        let directory = try CompiledFixture.anemll.makeModelDirectory()
        let backup = directory.deletingLastPathComponent().appendingPathComponent("backup-\(UUID().uuidString)")
        let original = try String(contentsOf: directory.appendingPathComponent("model.mil"), encoding: .utf8)

        try ContextResizer.resize(modelDirectory: directory, to: 2048, backup: backup)
        let rewritten = try ContextResizer.analyze(modelDirectory: directory)
        XCTAssertEqual(rewritten.context, 2048)
        let metadata = try CoreMLMetadata.parse(Data(contentsOf: directory.appendingPathComponent("metadata.json")))
        XCTAssertEqual(metadata.states.first?.shape, [56, 8, 2048, 128])

        // 作り直した表は 2048 行で、元の表と最初の 1024 行が一致する
        let rope = directory.appendingPathComponent("weights/context_rope.bin")
        let handle = try FileHandle(forReadingFrom: rope)
        let table = try MILBlob.readFloats(in: handle, at: 64)
        let header = [UInt8](try Data(contentsOf: rope).prefix(8))
        try handle.close()
        XCTAssertEqual(MILBlob.readUInt32(header, 0), 2)  // sin と cos の 2 つだけ
        XCTAssertEqual(table.count, 2048 * 128)
        let parts = CompiledFixture.anemll.folder.appendingPathComponent("weight-parts")
        let source = [UInt8](try Data(contentsOf: parts.appendingPathComponent("291329088.bin")))
        let originalTable = stride(from: 64, to: source.count, by: 2).map { Half.toFloat(UInt16(source[$0]) | UInt16(source[$0 + 1]) << 8) }
        for index in stride(from: 0, to: originalTable.count, by: 997) {
            XCTAssertEqual(table[index], originalTable[index], accuracy: 1e-3)
        }

        // もう一度変えても、元のファイルから書き換える
        try ContextResizer.resize(modelDirectory: directory, to: 8192, backup: backup)
        XCTAssertEqual(try ContextResizer.analyze(modelDirectory: directory).context, 8192)

        try ContextResizer.restore(modelDirectory: directory, backup: backup)
        XCTAssertEqual(try String(contentsOf: directory.appendingPathComponent("model.mil"), encoding: .utf8), original)
        XCTAssertFalse(FileManager.default.fileExists(atPath: rope.path))
    }

    func testRejectsModelsWithoutState() {
        let mil = """
        program(1.3)
        {
            func main<ios18>(tensor<int32, [1, 512]> input_ids) {
                tensor<fp16, [1, 512, 100]> logits = const()[name = string("x"), val = tensor<fp16, [1, 512, 100]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
            } -> (logits);
        }
        """
        XCTAssertThrowsError(try ContextResizer.analyze(mil: mil))
    }
}

final class ModelConfigInfoTests: XCTestCase {
    func testReadsTrainedLength() {
        XCTAssertEqual(ModelConfigInfo.maxPositions(from: Data(#"{"max_position_embeddings": 32768}"#.utf8)), 32768)
        XCTAssertEqual(ModelConfigInfo.maxPositions(from: Data(#"{"text_config": {"max_position_embeddings": 131072}}"#.utf8)), 131_072)
        XCTAssertEqual(ModelConfigInfo.maxPositions(from: Data(#"{"n_positions": 1024}"#.utf8)), 1024)
        XCTAssertNil(ModelConfigInfo.maxPositions(from: Data("{}".utf8)))
    }
}
