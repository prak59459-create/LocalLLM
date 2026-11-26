import XCTest
@testable import LocalLLMCore

/// Gemma 4 の MTP（先読み・推測デコード）。
/// 小さな本体（tiny-gemma4）と先読み用のモデル（tiny-gemma4-assistant）の正解は、Tools/gguf/make_mtp_fixtures.py が
/// llama.cpp の gemma4-assistant.cpp と同じ流れを numpy で計算したもの
final class GGUFDrafterTests: XCTestCase {
    private func convert(_ name: String, compact: Bool = false) throws -> GGUFWeightStore {
        let destination = GGUFFixtures.temporaryFile("\(name).llw")
        try GGUFConverter.convert(gguf: GGUFFixtures.root.appendingPathComponent("\(name).gguf"), to: destination, compact: compact)
        addTeardownBlock { try? FileManager.default.removeItem(at: destination) }
        return try GGUFWeightStore(url: destination)
    }

    private func argmax(_ logits: [Float]) -> Int { GGUFDrafter.argmax(logits) ?? 0 }

    func testReadsTheDraftModel() throws {
        let store = try convert("tiny-gemma4-assistant")
        let config = store.config
        XCTAssertEqual(config.family, .gemma4Assistant)
        XCTAssertEqual(config.drafterOutputSize, 32)
        XCTAssertEqual(config.layers.map(\.sliding), [true, false])
        XCTAssertEqual(config.layers.map(\.headSize), [8, 16])
        let target = try convert("tiny-gemma4").config
        let sources = try GGUFDrafterWeights.cacheLayers(drafter: config, target: target)
        XCTAssertEqual(sources.sliding, 2)
        XCTAssertEqual(sources.full, 3)
        // ほかのモデルとは組にならない
        XCTAssertThrowsError(try GGUFDrafterWeights.cacheLayers(drafter: config, target: try convert("tiny-gemma3").config))
    }

    /// 先読みの logits と予想が、numpy の正解と同じになる
    func testDraftsMatchNumpy() throws {
        let json = try GGUFFixtures.json("tiny-gemma4-assistant.json")
        let tokens = try XCTUnwrap(json["tokens"] as? [Int])
        let vocab = try XCTUnwrap(json["vocab"] as? Int)
        let cases = try XCTUnwrap(json["cases"] as? [[String: Any]])
        let targetStore = try convert("tiny-gemma4")
        let drafter = try GGUFDrafter(store: try convert("tiny-gemma4-assistant"), target: targetStore.config)
        // 本体の隠れ状態（出力層の前）も、numpy と同じ
        let hidden = try GGUFFixtures.floats(json["hidden"])
        let model = try GGUFModel(store: targetStore)
        let all = try model.evaluateAll(tokens)
        let d = targetStore.config.embeddingSize
        for row in 0..<tokens.count {
            let expected = Array(hidden[(row * d)..<((row + 1) * d)])
            XCTAssertLessThan(zip(all.hidden[row], expected).map { abs($0 - $1) }.max() ?? 1, 1e-4, "hidden \(row)")
        }
        for item in cases {
            let prefix = try XCTUnwrap(item["prefix"] as? Int)
            let target = try GGUFModel(store: targetStore)
            let logits = try target.evaluate(Array(tokens.prefix(prefix)))
            let first = argmax(logits)
            XCTAssertEqual(first, item["first"] as? Int)
            let expected = try GGUFFixtures.floats(item["logits"])
            var token = first, state = target.lastHidden
            var drafts: [Int] = []
            for step in 0..<3 {
                let (stepLogits, next) = drafter.step(token: token, hidden: state, position: prefix, target: target)
                let reference = Array(expected[(step * vocab)..<((step + 1) * vocab)])
                let difference = zip(stepLogits, reference).map { abs($0 - $1) }.max() ?? 1
                XCTAssertLessThan(difference, 1e-3, "prefix \(prefix) step \(step)")
                token = argmax(stepLogits)
                state = next
                drafts.append(token)
            }
            XCTAssertEqual(drafts, item["drafts"] as? [Int], "prefix \(prefix)")
            XCTAssertEqual(drafter.draft(after: first, hidden: target.lastHidden, position: prefix, count: 3, target: target), drafts)
            // 先読みは本体の KV キャッシュを変えない
            XCTAssertEqual(target.length, prefix)
        }
    }

    /// 先読みを使っても、使わないときと同じ文章になる（貪欲・8 ビットの重み・予想の数を変えても）
    func testSpeculativeGreedyMatchesPlainGreedy() throws {
        for compact in [false, true] {
            let targetStore = try convert("tiny-gemma4", compact: compact)
            let drafterStore = try convert("tiny-gemma4-assistant", compact: compact)
            let prompt = [1, 5, 9, 14, 3]
            let plain = try GGUFModel(store: targetStore)
            var logits = try plain.evaluate(prompt)
            var expected: [Int] = []
            for _ in 0..<24 {
                let token = argmax(logits)
                expected.append(token)
                logits = try plain.evaluate([token])
            }
            for count in [1, 2, 3, 5] {
                let target = try GGUFModel(store: targetStore)
                try target.attachDrafter(store: drafterStore)
                let first = argmax(try target.evaluate(prompt))
                var generated = [first]
                var accepted = 0, rounds = 0
                while generated.count < expected.count {
                    let tokens = try GGUFSpeculation.round(target: target, first: generated.last!, count: count) { logits, _ in
                        self.argmax(logits)
                    }
                    accepted += tokens.count - 1
                    rounds += 1
                    generated.append(contentsOf: tokens)
                    // 本体は、選んだ最後のトークンの手前まで読んでいる
                    XCTAssertEqual(target.length, prompt.count + generated.count - 1)
                }
                XCTAssertEqual(Array(generated.prefix(expected.count)), expected, "compact \(compact) count \(count)")
                print("GGUF MTP tiny (compact \(compact), drafts \(count)): \(accepted) of \(rounds * count) drafts accepted")
            }
        }
    }

    /// 受け入れ方: 予想と同じ間は進み、違ったら本体の選んだトークンで止める。全部当たれば、もう 1 つ選ぶ
    func testAcceptance() {
        var logits: [[Float]] = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
        var result = GGUFSpeculation.accept(drafts: [1, 0, 2], logits: &logits) { values, _ in GGUFDrafter.argmax(values) ?? 0 }
        XCTAssertEqual(result.accepted, 3)
        XCTAssertEqual(result.tokens, [1, 0, 2, 1])
        result = GGUFSpeculation.accept(drafts: [1, 2, 2], logits: &logits) { values, _ in GGUFDrafter.argmax(values) ?? 0 }
        XCTAssertEqual(result.accepted, 1)
        XCTAssertEqual(result.tokens, [1, 0])
        logits = [[0, 0, 1]]
        result = GGUFSpeculation.accept(drafts: [], logits: &logits) { values, _ in GGUFDrafter.argmax(values) ?? 0 }
        XCTAssertEqual(result.accepted, 0)
        XCTAssertEqual(result.tokens, [2])
    }

    /// 本物の Gemma 4 E2B と先読み用のモデルで: 先読みを使っても同じ文章になり、予想がそれなりに当たる
    /// （LOCALLLM_GGUF_MODELS に gemma-4-E2B-it-Q4_0.gguf と mtp-gemma-4-E2B-it-Q8_0.gguf を置いたときだけ）
    func testRealGemma4Drafter() throws {
        guard let directory = GGUFFixtures.realModels else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        let targetFile = directory.appendingPathComponent("gemma-4-E2B-it-Q4_0.gguf")
        let drafterFile = directory.appendingPathComponent("mtp-gemma-4-E2B-it-Q8_0.gguf")
        guard FileManager.default.fileExists(atPath: targetFile.path), FileManager.default.fileExists(atPath: drafterFile.path) else {
            throw XCTSkip("Gemma 4 E2B and its MTP model are not in LOCALLLM_GGUF_MODELS")
        }
        func store(_ source: URL) throws -> GGUFWeightStore {
            let destination = GGUFFixtures.temporaryFile(source.lastPathComponent + ".llw")
            try GGUFConverter.convert(gguf: source, to: destination)
            addTeardownBlock { try? FileManager.default.removeItem(at: destination) }
            return try GGUFWeightStore(url: destination)
        }
        let targetStore = try store(targetFile)
        let drafterStore = try store(drafterFile)
        // チャットの形の問い（Gemma 4 のトークナイザで分ける）
        let mapped = try MappedFile(url: targetFile)
        let file = try GGUFFile.parse(Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapped.pointer), count: mapped.count,
                                           deallocator: .none))
        let files = try GGUFTokenizer.files(from: file)
        let tokenizer = try HFTokenizer(tokenizerJSON: files.tokenizerJSON, tokenizerConfigJSON: files.tokenizerConfigJSON)
        let prompt = tokenizer.encode("<bos><|turn>user\nWrite a short story about a cat who learns to swim.<turn|>\n<|turn>model\n",
                                      addSpecialTokens: false)
        let plain = try GGUFModel(store: targetStore, maxContext: 256)
        var logits = try plain.evaluate(prompt)
        var expected: [Int] = []
        for _ in 0..<32 {
            let token = argmax(logits)
            expected.append(token)
            logits = try plain.evaluate([token])
        }
        let target = try GGUFModel(store: targetStore, maxContext: 256)
        try target.attachDrafter(store: drafterStore)
        let first = argmax(try target.evaluate(prompt))
        var generated = [first]
        var accepted = 0, drafted = 0
        while generated.count < expected.count {
            let tokens = try GGUFSpeculation.round(target: target, first: generated.last!, count: 3) { logits, _ in self.argmax(logits) }
            accepted += tokens.count - 1
            drafted += 3
            generated.append(contentsOf: tokens)
        }
        XCTAssertEqual(Array(generated.prefix(expected.count)), expected)
        let rate = Double(accepted) / Double(max(drafted, 1))
        print(String(format: "GGUF MTP Gemma 4 E2B: %d of %d drafts accepted (%.0f%%)", accepted, drafted, rate * 100))
        XCTAssertGreaterThan(rate, 0.2)
    }
}
