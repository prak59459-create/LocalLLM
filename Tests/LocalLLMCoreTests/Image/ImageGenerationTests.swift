import XCTest
@testable import LocalLLMCore

/// 画像生成のテスト用の正解（Tests/ImageFixtures。Tools/image/make_fixtures.py が作る。リポジトリに入っている）
enum ImageFixtures {
    static var root: URL {
        URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("ImageFixtures")
    }

    static func json(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: root.appendingPathComponent(name))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

final class DPMSolverSchedulerTests: XCTestCase {
    private func floats(_ value: Any?) throws -> [Float] {
        try XCTUnwrap(value as? [NSNumber]).map(\.floatValue)
    }

    /// 時刻の並びと各ステップの結果が、diffusers の DPMSolverMultistepScheduler と同じ
    func testMatchesDiffusers() throws {
        let cases = try XCTUnwrap(try ImageFixtures.json("dpm_solver.json")["cases"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(cases.count, 6)
        for item in cases {
            let steps = try XCTUnwrap(item["steps"] as? Int)
            let prediction: DPMSolverScheduler.Prediction = (item["prediction"] as? String) == "v_prediction" ? .velocity : .epsilon
            var scheduler = DPMSolverScheduler(stepCount: steps, prediction: prediction)
            XCTAssertEqual(scheduler.timesteps, try XCTUnwrap(item["timesteps"] as? [Int]), "steps \(steps)")
            var sample = try floats(item["initial"])
            let outputs = try XCTUnwrap(item["outputs"] as? [[NSNumber]])
            let expected = try XCTUnwrap(item["samples"] as? [[NSNumber]])
            for step in 0..<steps {
                sample = scheduler.step(output: outputs[step].map(\.floatValue), sample: sample)
                for (value, reference) in zip(sample, expected[step]) {
                    XCTAssertEqual(value, reference.floatValue, accuracy: 2e-4 * max(1, abs(reference.floatValue)),
                                   "steps \(steps) (\(prediction.rawValue)) step \(step)")
                }
            }
            XCTAssertTrue(scheduler.isFinished)
        }
    }

    func testGuidanceAndNoise() {
        XCTAssertEqual(DiffusionGuidance.combine(unconditional: [1, 2], conditional: [3, 1], scale: 7.5), [16, -5.5])
        var first = DiffusionNoise(seed: 42), second = DiffusionNoise(seed: 42), other = DiffusionNoise(seed: 43)
        let values = first.normals(4096)
        XCTAssertEqual(values, second.normals(4096))
        XCTAssertNotEqual(values, other.normals(4096))
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(values.count)
        XCTAssertEqual(mean, 0, accuracy: 0.08)
        XCTAssertEqual(variance, 1, accuracy: 0.12)
    }
}

final class CLIPTextTokenizerTests: XCTestCase {
    /// CLIP の語彙（./Tools/fetch_fixtures.sh が取ってくる。無ければ飛ばす）
    private func tokenizer() throws -> CLIPTextTokenizer {
        let folder = TestFixtures.root.appendingPathComponent("clip")
        guard let vocab = try? Data(contentsOf: folder.appendingPathComponent("vocab.json")),
              let merges = try? String(contentsOf: folder.appendingPathComponent("merges.txt"), encoding: .utf8) else {
            throw XCTSkip("CLIP fixtures missing (run ./Tools/fetch_fixtures.sh)")
        }
        return try CLIPTextTokenizer(vocabJSON: vocab, mergesText: merges)
    }

    func testMatchesHuggingFaceTokenizer() throws {
        let tokenizer = try tokenizer()
        XCTAssertEqual(tokenizer.startID, 49406)
        XCTAssertEqual(tokenizer.endID, 49407)
        let entries = try XCTUnwrap(try ImageFixtures.json("clip_tokens.json")["entries"] as? [[String: Any]])
        for entry in entries {
            let prompt = try XCTUnwrap(entry["prompt"] as? String)
            let expected = try XCTUnwrap(entry["ids"] as? [Int])
            XCTAssertEqual([tokenizer.startID] + tokenizer.tokens(prompt) + [tokenizer.endID], expected, prompt)
        }
    }

    func testFixedLengthEncoding() throws {
        let tokenizer = try tokenizer()
        let short = tokenizer.encode("a cat")
        XCTAssertEqual(short.count, 77)
        XCTAssertEqual(short.first, 49406)
        XCTAssertEqual(Array(short[3...]), [Int](repeating: 49407, count: 74))
        XCTAssertEqual(Array(tokenizer.encode("a cat", padID: 0)[4...]), [Int](repeating: 0, count: 73))
        let long = String(repeating: "very detailed painting, ", count: 60)
        XCTAssertTrue(tokenizer.isTruncated(long))
        XCTAssertFalse(tokenizer.isTruncated("a cat"))
        let encoded = tokenizer.encode(long)
        XCTAssertEqual(encoded.count, 77)
        XCTAssertEqual(encoded.last, 49407)
    }
}
