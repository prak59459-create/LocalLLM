#if canImport(Metal)
import Metal
import XCTest
@testable import LocalLLMCore

/// GPU（Metal）での GGUF の計算が、CPU での計算・llama.cpp と同じ結果になるか（Mac の CI で動く。GPU が無ければ飛ばす）
final class GGUFMetalTests: XCTestCase {
    private func requireDevice() throws {
        guard GGUFMetalModel.isAvailable else {
            print("GGUF Metal: no Metal device on this machine")
            throw XCTSkip("no Metal device")
        }
    }

    private func convert(_ family: String, compact: Bool) throws -> URL {
        let source = GGUFFixtures.root.appendingPathComponent("tiny-\(family).gguf")
        let destination = GGUFFixtures.temporaryFile("metal-\(family).llw")
        try GGUFConverter.convert(gguf: source, to: destination, compact: compact)
        return destination
    }

    private func reference(_ family: String) throws -> (tokens: [Int], logits: [[Float]]) {
        let json = try GGUFFixtures.json("tiny-\(family).json")
        let tokens = try XCTUnwrap(json["tokens"] as? [Int])
        let vocab = try XCTUnwrap(json["vocab"] as? Int)
        let flat = try GGUFFixtures.floats(json["logits"])
        return (tokens, (0..<tokens.count).map { Array(flat[($0 * vocab)..<(($0 + 1) * vocab)]) })
    }

    private func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).map { abs($0 - $1) }.max() ?? .infinity
    }

    /// Float32 のまま変換したモデルは、どの構造でも GPU の計算が llama.cpp と同じ logits になる（1 トークンずつ）
    func testEveryFamilyMatchesLlamaCppTokenByToken() throws {
        try requireDevice()
        for family in GGUFModelTests.families {
            let file = try convert(family, compact: false)
            defer { try? FileManager.default.removeItem(at: file) }
            let model = try GGUFMetalModel(store: try GGUFWeightStore(url: file))
            let expected = try reference(family)
            let tolerance: Float = family.hasPrefix("gemma") ? 0.01 : 0.002
            var worst: Float = 0
            for (position, token) in expected.tokens.enumerated() {
                let logits = try model.evaluate([token])
                let difference = maxDifference(logits, expected.logits[position])
                worst = max(worst, difference)
                XCTAssertLessThan(difference, tolerance, "\(family) position \(position)")
            }
            XCTAssertEqual(model.length, expected.tokens.count)
            print("GGUF Metal \(family): worst difference from llama.cpp \(worst), mapped weights \(model.usesMappedWeights), device \(model.deviceName)")
        }
    }

    /// まとめて読み込んでも、途中で巻き戻しても、KV キャッシュを手放しても同じ結果になる
    func testBatchedAndRewoundEvaluationMatch() throws {
        try requireDevice()
        for family in GGUFModelTests.families {
            let file = try convert(family, compact: false)
            defer { try? FileManager.default.removeItem(at: file) }
            let model = try GGUFMetalModel(store: try GGUFWeightStore(url: file))
            let expected = try reference(family)
            let count = expected.tokens.count
            let tolerance: Float = family.hasPrefix("gemma") ? 0.01 : 0.002
            var logits = try model.evaluate(expected.tokens)
            XCTAssertLessThan(maxDifference(logits, expected.logits[count - 1]), tolerance, "\(family) batch")
            model.truncate(to: 5)
            _ = try model.evaluate(Array(expected.tokens[5..<9]))
            XCTAssertLessThan(maxDifference(try model.evaluate([expected.tokens[9]]), expected.logits[9]), tolerance, "\(family) rewind")
            logits = try model.evaluate(Array(expected.tokens[10...]), batch: 3)
            XCTAssertLessThan(maxDifference(logits, expected.logits[count - 1]), tolerance, "\(family) rewind tail")
            model.releaseCache()
            XCTAssertEqual(model.length, 0)
            logits = try model.evaluate(expected.tokens, batch: 5)
            XCTAssertLessThan(maxDifference(logits, expected.logits[count - 1]), tolerance, "\(family) after release")
        }
    }

    /// 8 ビットの重みでも、GPU と CPU の計算は同じ結果になる
    func testCompactWeightsMatchCPU() throws {
        try requireDevice()
        for family in GGUFModelTests.families {
            let file = try convert(family, compact: true)
            defer { try? FileManager.default.removeItem(at: file) }
            let store = try GGUFWeightStore(url: file)
            let gpu = try GGUFMetalModel(store: store)
            let cpu = try GGUFModel(store: store)
            let tokens = try reference(family).tokens
            let batched = GGUFModelFactory.relativeDifference(try gpu.evaluate(tokens), try cpu.evaluate(tokens))
            XCTAssertLessThan(batched, 0.002, "\(family) batch")
            let single = GGUFModelFactory.relativeDifference(try gpu.evaluate([tokens[3]]), try cpu.evaluate([tokens[3]]))
            XCTAssertLessThan(single, 0.002, "\(family) single")
            // 読み込むときの確認も通る
            XCTAssertNoThrow(try GGUFModelFactory.verify(gpu, against: cpu), family)
        }
    }

    /// 読み込みの入口は、GPU が使えれば GPU 用を返す。CPU だけの設定なら CPU 用を返す
    func testFactoryPicksTheGPU() throws {
        try requireDevice()
        let file = try convert("qwen2", compact: true)
        defer { try? FileManager.default.removeItem(at: file) }
        let made = try GGUFModelFactory.make(store: file, context: 32, useGPU: true)
        XCTAssertTrue(made.isMetal, made.note ?? "")
        XCTAssertNil(made.note)
        XCTAssertTrue(made.model.computeDescription.hasPrefix("GPU"))
        XCTAssertEqual(made.model.length, 0)
        let cpu = try GGUFModelFactory.make(store: file, context: 32, useGPU: false)
        XCTAssertFalse(cpu.isMetal)
        XCTAssertTrue(cpu.model.computeDescription.hasPrefix("CPU"))
    }

    /// 確認用の「ずれ」の計算
    func testRelativeDifference() {
        XCTAssertEqual(GGUFModelFactory.relativeDifference([1, 2, 3], [1, 2, 3]), 0)
        XCTAssertEqual(GGUFModelFactory.relativeDifference([1, .nan, 3], [1, 2, 3]), .infinity)
        XCTAssertEqual(GGUFModelFactory.relativeDifference([1, 2], [1, 2, 3]), .infinity)
        XCTAssertGreaterThan(GGUFModelFactory.relativeDifference([3, 2, 1], [1, 2, 3]), 1)
    }

    /// 本物のモデルで、GPU の計算が llama.cpp と同じ続きを生成する。速さも測る
    func testRealModelsOnTheGPU() throws {
        try requireDevice()
        guard let directory = GGUFFixtures.realModels else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        let names = ["smollm2-135m-q4_k_m", "smollm2-135m-q8_0", "qwen2.5-0.5b-q4_k_m", "qwen2.5-1.5b-q4_k_m"]
        var checked = 0
        defer { print("GGUF Metal real models checked: \(checked)") }
        for name in names {
            let json = try GGUFFixtures.json("real-\(name).json")
            let source = directory.appendingPathComponent(try XCTUnwrap(json["file"] as? String))
            guard FileManager.default.fileExists(atPath: source.path) else { continue }
            let destination = GGUFFixtures.temporaryFile("metal-\(name).llw")
            defer { try? FileManager.default.removeItem(at: destination) }
            var started = Date()
            try GGUFConverter.convert(gguf: source, to: destination)
            print("GGUF Metal: converted \(name) in \(Date().timeIntervalSince(started)) s")
            let store = try GGUFWeightStore(url: destination)
            started = Date()
            let gpu = try GGUFMetalModel(store: store, maxContext: 1024)
            print("GGUF Metal: prepared \(name) in \(Date().timeIntervalSince(started)) s, mapped weights \(gpu.usesMappedWeights), batch \(gpu.batchCapacity)")
            try check(name, json: json, gpu: gpu, store: store)
            checked += 1
        }
    }

    private func check(_ name: String, json: [String: Any], gpu: GGUFMetalModel, store: GGUFWeightStore) throws {
        let tokens = try XCTUnwrap(json["tokens"] as? [Int])
        var logits = try gpu.evaluate(tokens)
        let top = try XCTUnwrap(json["top"] as? [Int])
        XCTAssertEqual(logits.indices.max { logits[$0] < logits[$1] }, top[0], name)

        // CPU と同じ値か
        let cpu = try GGUFModel(store: store, maxContext: 256)
        var started = Date()
        let cpuLogits = try cpu.evaluate(tokens)
        let cpuPrefill = Double(tokens.count) / Date().timeIntervalSince(started)
        XCTAssertLessThan(GGUFModelFactory.relativeDifference(logits, cpuLogits), 0.01, name)
        started = Date()
        var cpuNext = cpuLogits
        for _ in 0..<3 {
            let next = try XCTUnwrap(cpuNext.indices.max { cpuNext[$0] < cpuNext[$1] })
            cpuNext = try cpu.evaluate([next])
        }
        let cpuDecode = 3 / Date().timeIntervalSince(started)

        // llama.cpp と同じ続きになるか
        var generated: [Int] = []
        for _ in 0..<8 {
            let next = try XCTUnwrap(logits.indices.max { logits[$0] < logits[$1] })
            generated.append(next)
            logits = try gpu.evaluate([next])
        }
        XCTAssertEqual(generated, try XCTUnwrap(json["greedy"] as? [Int]), name)

        // 速さ: 1 トークンずつの生成（短い文脈・長い文脈）と、まとめての読み込み
        started = Date()
        for index in 0..<24 { logits = try gpu.evaluate([100 + index]) }
        let shortDecode = 24 / Date().timeIntervalSince(started)
        let prompt = (0..<512).map { 200 + $0 % 300 }
        started = Date()
        _ = try gpu.evaluate(prompt)
        let prefill = 512 / Date().timeIntervalSince(started)
        started = Date()
        for index in 0..<24 { logits = try gpu.evaluate([100 + index]) }
        let longDecode = 24 / Date().timeIntervalSince(started)
        XCTAssertTrue(logits.allSatisfy(\.isFinite), name)
        print(String(format: "SPEED-GPU %@: decode %.1f tok/s (after %d tokens: %.1f), prefill %.1f tok/s | CPU decode %.1f, prefill %.1f",
                     name, shortDecode, gpu.length, longDecode, prefill, cpuDecode, cpuPrefill))
    }
}
#endif
