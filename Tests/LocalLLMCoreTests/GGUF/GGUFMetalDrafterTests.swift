#if canImport(Metal)
import Metal
import XCTest
@testable import LocalLLMCore

/// GPU（Metal）での Gemma 4 の MTP（先読み・推測デコード）が、CPU と numpy の正解と同じになるか（Mac の CI で動く）
final class GGUFMetalDrafterTests: XCTestCase {
    // CPU・numpy の正解は Float32 の KV キャッシュなので、小さなモデルとの比べは Float の KV キャッシュで行う
    // （本物の Gemma 4 は、アプリと同じ半精度の KV キャッシュで、先読みあり・なしが同じ続きになるかを確かめる）
    override func setUp() {
        super.setUp()
        GGUFMetalModel.halfPrecisionCache = false
    }

    override func tearDown() {
        GGUFMetalModel.halfPrecisionCache = true
        super.tearDown()
    }

    private func requireDevice() throws {
        guard GGUFMetalModel.isAvailable else { throw XCTSkip("no Metal device") }
    }

    private func convert(_ name: String, compact: Bool, groups: GGUFMatrix.Storage? = nil) throws -> GGUFWeightStore {
        let destination = GGUFFixtures.temporaryFile("metal-\(name).llw")
        try GGUFConverter.convert(gguf: GGUFFixtures.root.appendingPathComponent("\(name).gguf"), to: destination, compact: compact,
                                  groups: groups)
        addTeardownBlock { try? FileManager.default.removeItem(at: destination) }
        return try GGUFWeightStore(url: destination)
    }

    private func argmax(_ logits: [Float]) -> Int { GGUFDrafter.argmax(logits) ?? 0 }

    /// GPU の先読みが、numpy の正解（Float32 の重み）と CPU の先読み（8 ビット・4 ビットの重み）と同じトークンを予想する
    func testDraftsMatchNumpyAndCPU() throws {
        try requireDevice()
        let json = try GGUFFixtures.json("tiny-gemma4-assistant.json")
        let tokens = try XCTUnwrap(json["tokens"] as? [Int])
        let cases = try XCTUnwrap(json["cases"] as? [[String: Any]])
        for (compact, groups) in [(false, nil), (true, nil), (true, GGUFMatrix.Storage.q4Groups)] as [(Bool, GGUFMatrix.Storage?)] {
            let targetStore = try convert("tiny-gemma4", compact: compact, groups: groups)
            let drafterStore = try convert("tiny-gemma4-assistant", compact: compact)
            for item in cases {
                let prefix = try XCTUnwrap(item["prefix"] as? Int)
                let gpu = try GGUFMetalModel(store: targetStore)
                try gpu.attachDrafter(store: drafterStore)
                let cpu = try GGUFModel(store: targetStore)
                try cpu.attachDrafter(store: drafterStore)
                let first = argmax(try gpu.evaluate(Array(tokens.prefix(prefix))))
                XCTAssertEqual(first, argmax(try cpu.evaluate(Array(tokens.prefix(prefix)))))
                XCTAssertEqual(gpu.lastHidden.count, targetStore.config.embeddingSize)
                let drafts = try gpu.draft(after: first, hidden: gpu.lastHidden, count: 3)
                XCTAssertEqual(drafts, try cpu.draft(after: first, hidden: cpu.lastHidden, count: 3), "compact \(compact) prefix \(prefix)")
                if !compact {
                    XCTAssertEqual(first, item["first"] as? Int)
                    XCTAssertEqual(drafts, item["drafts"] as? [Int], "prefix \(prefix)")
                }
                XCTAssertEqual(gpu.length, prefix)
            }
        }
    }

    /// GPU でまとめて確かめた logits と隠れ状態が、CPU と同じ。受け入れたあとも、続けて正しく計算できる
    func testVerifyAndAcceptMatchCPU() throws {
        try requireDevice()
        let targetStore = try convert("tiny-gemma4", compact: true, groups: .q4Groups)
        let gpu = try GGUFMetalModel(store: targetStore)
        let cpu = try GGUFModel(store: targetStore)
        let prompt = [1, 5, 9, 14, 3]
        _ = try gpu.evaluate(prompt)
        _ = try cpu.evaluate(prompt)
        let batch = [7, 31, 2, 18]
        try gpu.verify(batch)
        try cpu.verify(batch)
        for index in batch.indices {
            XCTAssertLessThan(GGUFModelFactory.relativeDifference(gpu.verifiedLogits(index), cpu.verifiedLogits(index)), 0.002)
        }
        gpu.accept(1)
        cpu.accept(1)
        XCTAssertEqual(gpu.length, prompt.count + 2)
        XCTAssertLessThan(GGUFModelFactory.relativeDifference(gpu.lastHidden, cpu.lastHidden), 0.002)
        XCTAssertLessThan(GGUFModelFactory.relativeDifference(try gpu.evaluate([11]), try cpu.evaluate([11])), 0.002)
    }

    /// GPU で先読みを使っても、使わないときと同じ文章になる（予想の数を変えても）
    func testSpeculativeGreedyMatchesPlainGreedy() throws {
        try requireDevice()
        let targetStore = try convert("tiny-gemma4", compact: true, groups: .q4Groups)
        let drafterStore = try convert("tiny-gemma4-assistant", compact: true)
        let prompt = [1, 5, 9, 14, 3]
        let plain = try GGUFMetalModel(store: targetStore)
        var logits = try plain.evaluate(prompt)
        var expected: [Int] = []
        for _ in 0..<24 {
            let token = argmax(logits)
            expected.append(token)
            logits = try plain.evaluate([token])
        }
        for count in [1, 3, 4] {
            let gpu = try GGUFMetalModel(store: targetStore)
            try gpu.attachDrafter(store: drafterStore)
            var generated = [argmax(try gpu.evaluate(prompt))]
            while generated.count < expected.count {
                generated += try GGUFSpeculation.round(target: gpu, first: generated.last!, count: count) { logits, _ in self.argmax(logits) }
                XCTAssertEqual(gpu.length, prompt.count + generated.count - 1)
            }
            XCTAssertEqual(Array(generated.prefix(expected.count)), expected, "drafts \(count)")
        }
    }

    /// 本物の Gemma 4 E2B と先読み用のモデル: GPU でも同じ文章になり、どれだけ当たって、どれだけ速くなるか
    /// （LOCALLLM_GGUF_MODELS に gemma-4-E2B-it-Q4_0.gguf と mtp-gemma-4-E2B-it-Q8_0.gguf を置いたときだけ。速さを測る流れで動く）
    func testRealGemma4SpeculationOnTheGPU() throws {
        try requireDevice()
        guard let directory = GGUFFixtures.realModels else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        let targetFile = directory.appendingPathComponent("gemma-4-E2B-it-Q4_0.gguf")
        let drafterFile = directory.appendingPathComponent("mtp-gemma-4-E2B-it-Q8_0.gguf")
        guard FileManager.default.fileExists(atPath: targetFile.path), FileManager.default.fileExists(atPath: drafterFile.path) else {
            throw XCTSkip("Gemma 4 E2B and its MTP model are not in LOCALLLM_GGUF_MODELS")
        }
        // アプリと同じ半精度の KV キャッシュで（先読みあり・なしのどちらも同じ KV キャッシュなので、同じ続きになる）
        GGUFMetalModel.halfPrecisionCache = true
        func store(_ source: URL) throws -> GGUFWeightStore {
            let destination = GGUFFixtures.temporaryFile(source.lastPathComponent + ".llw")
            try GGUFConverter.convert(gguf: source, to: destination)
            addTeardownBlock { try? FileManager.default.removeItem(at: destination) }
            return try GGUFWeightStore(url: destination)
        }
        let targetStore = try store(targetFile)
        let drafterStore = try store(drafterFile)
        let mapped = try MappedFile(url: targetFile)
        let file = try GGUFFile.parse(Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapped.pointer), count: mapped.count,
                                           deallocator: .none))
        let files = try GGUFTokenizer.files(from: file)
        let tokenizer = try HFTokenizer(tokenizerJSON: files.tokenizerJSON, tokenizerConfigJSON: files.tokenizerConfigJSON)
        let prompts = ["Write a short story about a cat who learns to swim.", "日本の四季について、短く説明してください。",
                       "Explain in simple words how a rainbow forms."]
        try measureCosts(targetStore: targetStore, drafterStore: drafterStore,
                         prompt: tokenizer.encode("<bos><|turn>user\n\(prompts[0])<turn|>\n<|turn>model\n", addSpecialTokens: false))
        for text in prompts {
            let prompt = tokenizer.encode("<bos><|turn>user\n\(text)<turn|>\n<|turn>model\n", addSpecialTokens: false)
            let plain = try GGUFMetalModel(store: targetStore, maxContext: 1024)
            var logits = try plain.evaluate(prompt)
            var expected: [Int] = []
            var started = Date()
            for _ in 0..<64 {
                let token = argmax(logits)
                expected.append(token)
                logits = try plain.evaluate([token])
            }
            let plainSpeed = 64 / Date().timeIntervalSince(started)
            // 予想の数を決め打ちしたもの（1・2・3 個）と、アプリと同じく DraftPlanner が決めるもの（最大 3 個。0 は auto）
            for count in [1, 2, 3, 0] {
                let gpu = try GGUFMetalModel(store: targetStore, maxContext: 1024)
                try gpu.attachDrafter(store: drafterStore)
                var planner = DraftPlanner(maximum: 3)
                var generated = [argmax(try gpu.evaluate(prompt))]
                var rounds = 0, drafted = 0, accepted = 0
                started = Date()
                while generated.count < expected.count + 1 {
                    let wanted = count > 0 ? count : planner.next()
                    let roundStart = Date()
                    let drafts = wanted > 0 ? try gpu.draft(after: generated.last!, hidden: gpu.lastHidden, count: wanted) : []
                    let draftEnd = Date()
                    try gpu.verify([generated.last!] + drafts)
                    planner.recordTimes(drafted: drafts.count, draftTime: draftEnd.timeIntervalSince(roundStart), verified: drafts.count + 1,
                                        verifyTime: Date().timeIntervalSince(draftEnd))
                    var hits = 0
                    while true {
                        let token = argmax(gpu.verifiedLogits(hits))
                        generated.append(token)
                        guard hits < drafts.count, token == drafts[hits] else { break }
                        hits += 1
                    }
                    gpu.accept(hits)
                    planner.record(drafted: drafts.count, accepted: hits)
                    rounds += 1
                    drafted += drafts.count
                    accepted += hits
                }
                let speed = Double(generated.count - 1) / Date().timeIntervalSince(started)
                let name = count > 0 ? "drafts \(count)" : "auto (up to 3)"
                XCTAssertEqual(Array(generated.prefix(expected.count)), expected, "\(text) \(name)")
                print(String(format: "SPEED-MTP %@: %.1f tok/s (plain %.1f, x%.2f), %d of %d drafts accepted (%.0f%%), %d rounds | %@",
                             name, speed, plainSpeed, speed / plainSpeed, accepted, drafted,
                             100 * Double(accepted) / Double(max(drafted, 1)), rounds, String(text.prefix(24))))
            }
        }
    }

    /// 1 回の手間を測る: ふつうの 1 トークン、n 個まとめての確かめ、k 個の予想（時計の時間と GPU の時間、ミリ秒）
    private func measureCosts(targetStore: GGUFWeightStore, drafterStore: GGUFWeightStore, prompt: [Int]) throws {
        let model = try GGUFMetalModel(store: targetStore, maxContext: 1024)
        try model.attachDrafter(store: drafterStore)
        var token = argmax(try model.evaluate(prompt))
        func time(_ body: () throws -> Void) rethrows -> (wall: Double, gpu: Double) {
            var wall = 0.0, gpu = 0.0
            for _ in 0..<6 {
                let start = Date()
                try body()
                wall += Date().timeIntervalSince(start)
                gpu += model.lastGPUSeconds
            }
            return (wall / 6 * 1000, gpu / 6 * 1000)
        }
        let plain = try time {
            token = argmax(try model.evaluate([token]))
        }
        print(String(format: "COST-MTP plain step: %.1f ms (GPU %.1f)", plain.wall, plain.gpu))
        // まとめての計算のしかたごとに: n 個の確かめと、64 トークンのプロンプトの読み込み
        let choices: [(String, GGUFMetalModel.BatchKernel?)] = [("auto", nil), ("few", .few), ("matrix", .matrix), ("legacy", .legacy)]
        defer { GGUFMetalModel.batchKernelOverride = nil }
        for (name, choice) in choices {
            GGUFMetalModel.batchKernelOverride = choice
            for n in 1...5 {
                let verify = try time {
                    try model.verify(Array(repeating: token, count: n))
                    model.accept(0)
                }
                print(String(format: "COST-MTP %@ verify %d: %.1f ms (GPU %.1f) = x%.2f of a plain step", name, n, verify.wall,
                             verify.gpu, verify.wall / plain.wall))
            }
            let prefillModel = try GGUFMetalModel(store: targetStore, maxContext: 1024)
            let long = Array((prompt + prompt + prompt + prompt + prompt).prefix(64))
            let start = Date()
            _ = try prefillModel.evaluate(long)
            print(String(format: "COST-MTP %@ prefill 64 tokens: %.0f ms (%.1f tok/s)", name, Date().timeIntervalSince(start) * 1000,
                         64 / Date().timeIntervalSince(start)))
        }
        for k in 1...4 {
            let draft = try time {
                _ = try model.draft(after: token, hidden: model.lastHidden, count: k)
            }
            print(String(format: "COST-MTP draft %d: %.1f ms (GPU %.1f) = x%.2f of a plain step", k, draft.wall, draft.gpu,
                         draft.wall / plain.wall))
        }
        var logits = try model.evaluate([token])
        let start = Date()
        for _ in 0..<20 { token = argmax(logits) }
        print(String(format: "COST-MTP CPU argmax over %d logits: %.2f ms", logits.count, Date().timeIntervalSince(start) * 1000 / 20))
        logits = []
    }
}
#endif
