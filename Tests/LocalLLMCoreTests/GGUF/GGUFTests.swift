import XCTest
@testable import LocalLLMCore

/// GGUF のテスト用ファイル（Tests/GGUFFixtures。Tools/gguf/make_fixtures.py が作る。リポジトリに入っている）
enum GGUFFixtures {
    static var root: URL {
        URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("GGUFFixtures")
    }

    static func json(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: root.appendingPathComponent(name))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    static func floats(_ base64: Any?) throws -> [Float] {
        let data = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(base64 as? String)))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func temporaryFile(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("gguf-test-\(UUID().uuidString)-\(name)")
    }

    /// 本物のモデルを置いたフォルダ（環境変数 LOCALLLM_GGUF_MODELS。無ければそのテストは飛ばす）
    static var realModels: URL? {
        ProcessInfo.processInfo.environment["LOCALLLM_GGUF_MODELS"].map { URL(fileURLWithPath: $0) }
    }
}

final class GGUFQuantTests: XCTestCase {
    /// どの量子化形式も、gguf パッケージ（Python）と同じ値に戻る
    func testDequantizationMatchesReference() throws {
        let entries = try XCTUnwrap(try GGUFFixtures.json("quant_blocks.json")["entries"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(entries.count, 20)
        for entry in entries {
            let name = entry["name"] as? String ?? "?"
            let type = try XCTUnwrap(GGUFTensorType(rawValue: UInt32(try XCTUnwrap(entry["type"] as? Int))), name)
            let count = try XCTUnwrap(entry["count"] as? Int)
            let bytes = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(entry["bytes"] as? String)))
            let expected = try GGUFFixtures.floats(entry["expected"])
            XCTAssertEqual(expected.count, count, name)
            XCTAssertEqual(bytes.count, count / type.block.elements * type.block.bytes, name)
            var output = [Float](repeating: .nan, count: count)
            try bytes.withUnsafeBytes { raw in
                try output.withUnsafeMutableBufferPointer { buffer in
                    try GGUFQuant.dequantize(type, source: raw.baseAddress!, count: count, into: buffer.baseAddress!)
                }
            }
            var mismatches = 0
            for index in 0..<count {
                let a = output[index], b = expected[index]
                if a.isNaN && b.isNaN { continue }
                if a == b { continue }
                if abs(a - b) <= 1e-5 * max(1, abs(b)) { continue }
                mismatches += 1
                if mismatches <= 3 { XCTFail("\(name)[\(index)]: \(a) != \(b)") }
            }
            XCTAssertEqual(mismatches, 0, name)
        }
    }

    func testHalfConversion() {
        XCTAssertEqual(GGUFQuant.half(0x3C00), 1)
        XCTAssertEqual(GGUFQuant.half(0xC000), -2)
        XCTAssertEqual(GGUFQuant.half(0x0001), 5.9604645e-8)
        XCTAssertEqual(GGUFQuant.half(0x7C00), .infinity)
        XCTAssertTrue(GGUFQuant.half(0x7E00).isNaN)
        XCTAssertEqual(GGUFQuant.half(0x3555), 0.33325195)
    }
}

final class GGUFModelTests: XCTestCase {
    static let families = ["llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3"]

    private func load(_ family: String, compact: Bool, chunks: Int? = nil) throws -> (GGUFModel, URL) {
        let source = GGUFFixtures.root.appendingPathComponent("tiny-\(family).gguf")
        let destination = GGUFFixtures.temporaryFile("\(family).llw")
        try GGUFConverter.convert(gguf: source, to: destination, compact: compact)
        return (try GGUFModel(store: try GGUFWeightStore(url: destination), chunkCount: chunks), destination)
    }

    private func reference(_ family: String) throws -> (tokens: [Int], logits: [[Float]]) {
        let json = try GGUFFixtures.json("tiny-\(family).json")
        let tokens = try XCTUnwrap(json["tokens"] as? [Int])
        let vocab = try XCTUnwrap(json["vocab"] as? Int)
        let flat = try GGUFFixtures.floats(json["logits"])
        XCTAssertEqual(flat.count, tokens.count * vocab)
        return (tokens, (0..<tokens.count).map { Array(flat[($0 * vocab)..<(($0 + 1) * vocab)]) })
    }

    private func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).map { abs($0 - $1) }.max() ?? .infinity
    }

    /// Float32 のまま変換したモデルは、どの構造でも llama.cpp と同じ logits になる（1 トークンずつ計算）
    func testEveryFamilyMatchesLlamaCppTokenByToken() throws {
        for family in Self.families {
            let (model, file) = try load(family, compact: false)
            defer { try? FileManager.default.removeItem(at: file) }
            let expected = try reference(family)
            // Gemma 系は GELU を llama.cpp が半精度の表で計算するので、その分だけずれる
            let tolerance: Float = family.hasPrefix("gemma") ? 0.01 : 0.001
            for (position, token) in expected.tokens.enumerated() {
                let logits = try model.evaluate([token])
                let difference = maxDifference(logits, expected.logits[position])
                XCTAssertLessThan(difference, tolerance, "\(family) position \(position)")
            }
            XCTAssertEqual(model.length, expected.tokens.count)
        }
    }

    /// まとめて読み込んでも（プロンプトの読み込み）、途中で巻き戻しても同じ結果になる
    func testBatchedAndRewoundEvaluationMatch() throws {
        for family in Self.families {
            let (model, file) = try load(family, compact: false, chunks: 3)
            defer { try? FileManager.default.removeItem(at: file) }
            let expected = try reference(family)
            let count = expected.tokens.count
            let tolerance: Float = family.hasPrefix("gemma") ? 0.01 : 0.001
            var logits = try model.evaluate(expected.tokens)
            XCTAssertLessThan(maxDifference(logits, expected.logits[count - 1]), tolerance, "\(family) batch")
            // 5 トークンまで巻き戻して、残りを 2 回に分けて読む
            model.truncate(to: 5)
            _ = try model.evaluate(Array(expected.tokens[5..<9]))
            XCTAssertLessThan(maxDifference(try model.evaluate([expected.tokens[9]]), expected.logits[9]), tolerance, "\(family) rewind")
            logits = try model.evaluate(Array(expected.tokens[10...]))
            XCTAssertLessThan(maxDifference(logits, expected.logits[count - 1]), tolerance, "\(family) rewind tail")
            // KV キャッシュを手放したあとも、最初から読み直せる
            model.releaseCache()
            XCTAssertEqual(model.length, 0)
            logits = try model.evaluate(expected.tokens)
            XCTAssertLessThan(maxDifference(logits, expected.logits[count - 1]), tolerance, "\(family) after release")
        }
    }

    /// 8 ビットにした重みでも、ほぼ同じ logits になる
    func testCompactWeightsStayClose() throws {
        for family in Self.families {
            let (model, file) = try load(family, compact: true)
            defer { try? FileManager.default.removeItem(at: file) }
            let expected = try reference(family)
            let logits = try model.evaluate(expected.tokens)
            let reference = expected.logits[expected.tokens.count - 1]
            // logits のばらつき（標準偏差 4 前後）に比べて十分小さい
            XCTAssertLessThan(maxDifference(logits, reference), 0.6, family)
            let best = try XCTUnwrap(reference.indices.max { reference[$0] < reference[$1] })
            let ranked = logits.indices.sorted { logits[$0] > logits[$1] }
            XCTAssertTrue(ranked.prefix(3).contains(best), family)
        }
    }

    func testConfigIsReadFromMetadata() throws {
        let file = try GGUFFile.open(GGUFFixtures.root.appendingPathComponent("tiny-gemma3.gguf"))
        XCTAssertEqual(file.architecture, "gemma3")
        let config = try GGUFModelConfig(file: file)
        XCTAssertEqual(config.family, .gemma3)
        XCTAssertEqual(config.layerCount, 3)
        XCTAssertEqual(config.headSize, 16)
        XCTAssertEqual(config.slidingWindow, 5)
        XCTAssertEqual(config.slidingLayers, [true, false, true])
        XCTAssertEqual(config.ropeBase, 1_000_000)
        XCTAssertEqual(config.slidingRopeBase, 10_000)
        XCTAssertTrue(config.tiedOutput)
        XCTAssertEqual(file.dominantQuantization, "F32")

        let phi = try GGUFModelConfig(file: try GGUFFile.open(GGUFFixtures.root.appendingPathComponent("tiny-phi3.gguf")))
        XCTAssertTrue(phi.hasFusedQKV)
        XCTAssertTrue(phi.hasFusedGateUp)
        XCTAssertFalse(phi.tiedOutput)
    }

    func testTruncatedHeaderAsksForMoreBytes() throws {
        let data = try Data(contentsOf: GGUFFixtures.root.appendingPathComponent("tiny-llama.gguf"))
        XCTAssertThrowsError(try GGUFFile.parse(data.prefix(300))) { error in
            guard case GGUFError.truncated = error else { return XCTFail("\(error)") }
        }
        XCTAssertThrowsError(try GGUFFile.parse(Data("not a gguf file".utf8))) { error in
            XCTAssertEqual(error as? GGUFError, .notGGUF)
        }
        // 先頭部分だけあれば、重みの中身が無くても読める
        let header = try GGUFFile.parse(data.prefix(try GGUFFile.parse(data).dataOffset))
        XCTAssertEqual(header.tensors.count, try GGUFFile.parse(data).tensors.count)
    }

    /// 本物のモデル（SmolLM2 135M）で、llama.cpp と同じ続きを生成する
    func testRealModelsMatchLlamaCpp() throws {
        guard let directory = GGUFFixtures.realModels else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        // 正解のある本物のモデルのうち、フォルダに置いてあるものを確かめる（CI では SmolLM2 だけ）
        let names = ["smollm2-135m-q4_k_m", "smollm2-135m-q8_0", "qwen2.5-0.5b-q4_k_m", "qwen3-0.6b-q4_k_m",
                     "gemma-3-1b-q4_k_m", "llama-3.2-1b-q4_k_m"]
        var checked = 0
        defer { print("GGUF real models checked: \(checked)") }
        for name in names {
            let json = try GGUFFixtures.json("real-\(name).json")
            let source = directory.appendingPathComponent(try XCTUnwrap(json["file"] as? String))
            guard FileManager.default.fileExists(atPath: source.path) else { continue }
            let destination = GGUFFixtures.temporaryFile("\(name).llw")
            defer { try? FileManager.default.removeItem(at: destination) }
            let started = Date()
            try GGUFConverter.convert(gguf: source, to: destination)
            print("converted \(name) in \(Date().timeIntervalSince(started)) s")
            let model = try GGUFModel(store: try GGUFWeightStore(url: destination), maxContext: 256)
            var logits = try model.evaluate(try XCTUnwrap(json["tokens"] as? [Int]))
            let top = try XCTUnwrap(json["top"] as? [Int])
            let topLogits = try XCTUnwrap(json["topLogits"] as? [Double])
            let std = try XCTUnwrap(json["std"] as? Double)
            XCTAssertEqual(logits.indices.max { logits[$0] < logits[$1] }, top[0], name)
            var worst = 0.0
            for (token, value) in zip(top, topLogits) {
                XCTAssertEqual(Double(logits[token]), value, accuracy: 0.15 * std, "\(name) token \(token)")
                worst = max(worst, abs(Double(logits[token]) - value))
            }
            print("DIFF \(name) worst top-logit difference \(worst) (std \(std))")
            var generated: [Int] = []
            let decodeStart = Date()
            for _ in 0..<8 {
                let next = try XCTUnwrap(logits.indices.max { logits[$0] < logits[$1] })
                generated.append(next)
                logits = try model.evaluate([next])
            }
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            XCTAssertEqual(generated, try XCTUnwrap(json["greedy"] as? [Int]), name)
            // 速さの目安（CI のログで見る）: 1 トークンずつの生成と、64 トークンまとめての読み込み
            let prompt = (0..<64).map { 100 + $0 }
            let prefillStart = Date()
            _ = try model.evaluate(prompt)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            print(String(format: "SPEED %@: decode %.1f tok/s, prefill %.1f tok/s", name, 8 / decodeSeconds, 64 / prefillSeconds))
            checked += 1
        }
    }
}
