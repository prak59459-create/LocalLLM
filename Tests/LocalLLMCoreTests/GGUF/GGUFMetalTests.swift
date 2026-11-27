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

    private func convert(_ family: String, compact: Bool, groups: GGUFMatrix.Storage? = nil) throws -> URL {
        let source = GGUFFixtures.root.appendingPathComponent("tiny-\(family).gguf")
        let destination = GGUFFixtures.temporaryFile("metal-\(family).llw")
        try GGUFConverter.convert(gguf: source, to: destination, compact: compact, groups: groups)
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

    /// 4・5 ビットの重みでも、GPU と CPU の計算は同じ結果になる（1 トークンずつ・まとめて・8 個を超えるまとめ。
    /// FFN の gate と up をまとめた計算、正規化と足し算・q k の正規化と RoPE をまとめた計算も通る）
    func testGroupedWeightsMatchCPU() throws {
        try requireDevice()
        var failures: [String] = []
        let families = (GGUFModelTests.families + GGUFModelTests.moreFamilies).filter { !Self.cpuOnlyFamilies.contains($0) }
        for storage in [GGUFMatrix.Storage.q4Groups, .q5Groups] {
            for family in families {
                let file = try convert(family, compact: true, groups: storage)
                defer { try? FileManager.default.removeItem(at: file) }
                let store = try GGUFWeightStore(url: file)
                let gpu = try GGUFMetalModel(store: store)
                let cpu = try GGUFModel(store: store)
                let tokens = try reference(family).tokens
                var worst: Float = 0
                for token in tokens.prefix(6) {
                    worst = max(worst, GGUFModelFactory.relativeDifference(try gpu.evaluate([token]), try cpu.evaluate([token])))
                }
                // 8 個を超えるまとめ（まとめての計算は 8 トークンずつ）
                let rest = Array(tokens.dropFirst(6))
                let batched = GGUFModelFactory.relativeDifference(try gpu.evaluate(rest, batch: 64), try cpu.evaluate(rest, batch: 64))
                if worst > 0.002 || batched > 0.002 { failures.append("\(family) \(storage): \(worst) / \(batched)") }
            }
        }
        XCTAssertEqual(failures, [])
    }

    /// 1 回の計算の、部分ごとの GPU の時間が測れる（速さを調べる用）
    func testProfileReportsSections() throws {
        try requireDevice()
        let file = try convert("gemma4", compact: true, groups: .q4Groups)
        defer { try? FileManager.default.removeItem(at: file) }
        let gpu = try GGUFMetalModel(store: try GGUFWeightStore(url: file))
        let tokens = try reference("gemma4").tokens
        let expected = try gpu.evaluate(Array(tokens.prefix(4)))
        gpu.truncate(to: 3)
        let sections = try gpu.profile([tokens[3]])
        XCTAssertTrue(["per-layer inputs", "attention", "feed-forward", "per-layer input", "output"].allSatisfy { sections[$0] != nil },
                      "\(sections.keys.sorted())")
        // 測ったあとも、同じ結果が続く
        gpu.truncate(to: 3)
        XCTAssertLessThan(GGUFModelFactory.relativeDifference(try gpu.evaluate([tokens[3]]), expected), 1e-4)
        XCTAssertGreaterThan(gpu.lastGPUSeconds, 0)
    }

    /// ほかの構造も、GPU と CPU で同じ logits になる（CPU は llama.cpp と一致することを GGUFModelTests で確かめている）
    /// GPU ではまだ計算しない構造
    static let cpuOnlyFamilies: Set<String> = ["qwen35", "qwen3moe", "qwen2moe", "olmoe", "granitemoe"]

    func testMoreArchitecturesMatchCPU() throws {
        try requireDevice()
        var failures: [String] = []
        for family in GGUFModelTests.moreFamilies {
            for compact in [false, true] {
                let file = try convert(family, compact: compact)
                defer { try? FileManager.default.removeItem(at: file) }
                let store = try GGUFWeightStore(url: file)
                // GPU ではまだ計算しない構造（Qwen3.5・MoE）は、GPU 用を作らずに断り、読み込みの入口は CPU 用を返す
                if Self.cpuOnlyFamilies.contains(family) {
                    XCTAssertThrowsError(try GGUFMetalModel(store: store), family) { error in
                        guard case .unavailable = error as? GGUFMetalError else { return XCTFail("\(family): \(error)") }
                    }
                    XCTAssertFalse(try GGUFModelFactory.make(store: file, context: 64, useGPU: true).isMetal, family)
                    continue
                }
                let gpu = try GGUFMetalModel(store: store)
                let cpu = try GGUFModel(store: store)
                let tokens = try reference(family).tokens
                var worst: Float = 0
                for token in tokens {
                    worst = max(worst, GGUFModelFactory.relativeDifference(try gpu.evaluate([token]), try cpu.evaluate([token])))
                }
                gpu.releaseCache()
                cpu.releaseCache()
                let batched = GGUFModelFactory.relativeDifference(try gpu.evaluate(tokens, batch: 6), try cpu.evaluate(tokens, batch: 6))
                print("GGUF Metal \(family)\(compact ? " (8-bit)" : ""): relative difference from CPU \(worst), batched \(batched)")
                if worst > 0.002 || batched > 0.002 { failures.append("\(family) \(compact): \(worst) / \(batched)") }
            }
        }
        XCTAssertEqual(failures, [])
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
        // Gemma 4 E2B は大きいので、速さを測る専用の流れ（.github/workflows/bench.yml）だけで取ってくる
        let names = ["smollm2-135m-q4_k_m", "smollm2-135m-q8_0", "qwen2.5-0.5b-q4_k_m", "qwen2.5-1.5b-q4_k_m", "gemma-4-e2b-q4_0"]
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
            print("GGUF Metal: prepared \(name) in \(Date().timeIntervalSince(started)) s, mapped weights \(gpu.usesMappedWeights), batch \(gpu.batchCapacity), matrix kernels \(gpu.usesMatrixKernels) (\(gpu.matrixNote))")
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
        try profile(name, gpu: gpu, store: store)
    }

    /// 1 トークンにかかる時間（全体・GPU が動いていた時間）と、GPU の時間の内訳
    private func profile(_ name: String, gpu: GGUFMetalModel, store: GGUFWeightStore) throws {
        gpu.truncate(to: 64)
        var gpuSeconds = 0.0
        let started = Date()
        for index in 0..<16 {
            _ = try gpu.evaluate([300 + index])
            gpuSeconds += gpu.lastGPUSeconds
        }
        let wall = Date().timeIntervalSince(started) / 16
        let megabytes = Double(store.header.dataBytes) / 1_048_576
        print(String(format: "TIME-GPU %@: %.2f ms per token (GPU %.2f ms), weights file %.0f MB", name, wall * 1000,
                     gpuSeconds / 16 * 1000, megabytes))
        var sections: [String: Double] = [:]
        for index in 0..<4 {
            for (section, seconds) in try gpu.profile([400 + index]) { sections[section, default: 0] += seconds / 4 }
        }
        let parts = sections.sorted { $0.value > $1.value }.map { String(format: "%@ %.2f ms", $0.key, $0.value * 1000) }
        print("PROFILE-GPU \(name): " + parts.joined(separator: ", ")
              + String(format: " | total %.2f ms", sections.values.reduce(0, +) * 1000))
    }
}
#endif
