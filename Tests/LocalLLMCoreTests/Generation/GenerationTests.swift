import Foundation
import XCTest
@testable import LocalLLMCore

/// 1 バイト = 1 トークンのトークナイザ（UTF-8 の途中で切れるのを確かめる用）
final class ByteMockTokenizer: TextTokenizer {
    func encode(_ text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) } }
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: ids.filter { (0..<256).contains($0) }.map { UInt8($0) }, as: UTF8.self)
    }
    func tokenID(for token: String) -> Int? { token == "<eos>" ? 300 : nil }
    func token(for id: Int) -> String? { nil }
}

final class GenerationParametersTests: XCTestCase {
    func testDecodingFillsMissingFieldsWithDefaults() throws {
        let decoded = try JSONDecoder().decode(GenerationParameters.self, from: Data(#"{"temperature":0.2,"seed":5}"#.utf8))
        XCTAssertEqual(decoded.temperature, 0.2)
        XCTAssertEqual(decoded.seed, 5)
        XCTAssertEqual(decoded.topK, GenerationParameters.default.topK)
        XCTAssertEqual(decoded.maxNewTokens, GenerationParameters.default.maxNewTokens)
        let roundTrip = try JSONDecoder().decode(GenerationParameters.self, from: JSONEncoder().encode(decoded))
        XCTAssertEqual(roundTrip, decoded)
        XCTAssertEqual(try JSONDecoder().decode(GenerationParameters.self, from: Data("{}".utf8)), GenerationParameters())
    }

    func testSplitMix64KnownValues() {
        var generator = SplitMix64(seed: 0)
        XCTAssertEqual(generator.next(), 0xE220_A839_7B1D_CDAF)
        XCTAssertEqual(generator.next(), 0x6E78_9E6A_A1B9_65F4)
    }
}

final class TokenSamplerTests: XCTestCase {
    func parameters(temperature: Double = 1, topK: Int = 0, topP: Double = 1, minP: Double = 0,
                    penalty: Double = 1, seed: UInt64? = 1) -> GenerationParameters {
        var p = GenerationParameters()
        p.temperature = temperature
        p.topK = topK
        p.topP = topP
        p.minP = minP
        p.repetitionPenalty = penalty
        p.seed = seed
        return p
    }

    func testGreedyAndNaN() {
        var sampler = TokenSampler(parameters: parameters(temperature: 0))
        var logits: [Float] = [0.1, .nan, 3, -1]
        XCTAssertEqual(sampler.sample(logits: &logits, previousTokens: []), 2)
        var allBad: [Float] = [.nan, -.infinity]
        XCTAssertEqual(sampler.sample(logits: &allBad, previousTokens: []), 1 - 1)
        var empty: [Float] = []
        XCTAssertEqual(sampler.sample(logits: &empty, previousTokens: []), 0)
        // top-k 1 も貪欲法
        var topOne = TokenSampler(parameters: parameters(temperature: 1.5, topK: 1))
        var values: [Float] = [1, 5, 4.9]
        for _ in 0..<10 { XCTAssertEqual(topOne.sample(logits: &values, previousTokens: []), 1) }
    }

    func testRepetitionPenaltyLikeHuggingFace() {
        var sampler = TokenSampler(parameters: parameters(temperature: 0, penalty: 1.1))
        var logits: [Float] = [2, 1.9, -1, -1.05]
        XCTAssertEqual(sampler.sample(logits: &logits, previousTokens: [0, 0, 2]), 1)
        XCTAssertEqual(logits[0], 2 / 1.1, accuracy: 1e-6)
        XCTAssertEqual(logits[2], -1.1, accuracy: 1e-6)
        XCTAssertEqual(logits[3], -1.05)
        // 範囲外のトークンは無視する
        var other: [Float] = [1, 0]
        XCTAssertEqual(sampler.sample(logits: &other, previousTokens: [-1, 9]), 0)
    }

    func testRepetitionWindowOnlyLooksAtRecentTokens() {
        var p = parameters(temperature: 0, penalty: 2)
        p.repetitionWindow = 2
        var sampler = TokenSampler(parameters: p)
        var logits: [Float] = [3, 2]
        XCTAssertEqual(sampler.sample(logits: &logits, previousTokens: [0, 1, 1]), 0)
        XCTAssertEqual(logits[0], 3)
    }

    func frequencies(_ p: GenerationParameters, logits: [Float], draws: Int = 20_000) -> [Int] {
        var sampler = TokenSampler(parameters: p)
        var counts = [Int](repeating: 0, count: logits.count)
        for _ in 0..<draws {
            var copy = logits
            counts[sampler.sample(logits: &copy, previousTokens: [])] += 1
        }
        return counts
    }

    func testSamplingFollowsProbabilities() {
        let logits: [Float] = [log(0.7), log(0.2), log(0.1)]
        let counts = frequencies(parameters(), logits: logits)
        XCTAssertEqual(Double(counts[0]) / 20_000, 0.7, accuracy: 0.02)
        XCTAssertEqual(Double(counts[1]) / 20_000, 0.2, accuracy: 0.02)
        XCTAssertEqual(Double(counts[2]) / 20_000, 0.1, accuracy: 0.02)
        // 温度 0.5 なら確率は 2 乗に比例: 0.49 : 0.04 : 0.01
        let sharp = frequencies(parameters(temperature: 0.5), logits: logits)
        XCTAssertEqual(Double(sharp[0]) / 20_000, 0.49 / 0.54, accuracy: 0.02)
    }

    func testTopPAndMinPCutTheTail() {
        let logits: [Float] = [log(0.7), log(0.2), log(0.1)]
        let topP = frequencies(parameters(topP: 0.75), logits: logits, draws: 3000)
        XCTAssertEqual(topP[2], 0)
        XCTAssertGreaterThan(topP[1], 0)
        XCTAssertEqual(Double(topP[0]) / 3000, 0.7 / 0.9, accuracy: 0.04)
        let minP = frequencies(parameters(minP: 0.25), logits: logits, draws: 3000)
        XCTAssertEqual(minP[2], 0)
        XCTAssertGreaterThan(minP[1], 0)
        let topK = frequencies(parameters(topK: 2), logits: logits, draws: 3000)
        XCTAssertEqual(topK[2], 0)
        // top-p がとても小さくても 1 つは残る
        let tiny = frequencies(parameters(topP: 0.01), logits: logits, draws: 200)
        XCTAssertEqual(tiny[0], 200)
    }

    func testSeedMakesSamplingReproducible() {
        let logits: [Float] = (0..<50).map { Float($0 % 7) * 0.3 }
        func run(seed: UInt64) -> [Int] {
            var sampler = TokenSampler(parameters: parameters(seed: seed))
            return (0..<30).map { _ in
                var copy = logits
                return sampler.sample(logits: &copy, previousTokens: [])
            }
        }
        XCTAssertEqual(run(seed: 3), run(seed: 3))
        XCTAssertNotEqual(run(seed: 3), run(seed: 4))
    }
}

final class StreamingDetokenizerTests: XCTestCase {
    func testNeverEmitsBrokenUTF8() {
        let text = "日本語😀と English。"
        let detokenizer = StreamingDetokenizer(tokenizer: ByteMockTokenizer())
        var pieces: [String] = []
        for byte in text.utf8 {
            let piece = detokenizer.append(Int(byte))
            XCTAssertFalse(piece.contains("\u{FFFD}"))
            pieces.append(piece)
        }
        pieces.append(detokenizer.flush())
        XCTAssertEqual(pieces.joined(), text)
        XCTAssertEqual(detokenizer.text, text)
        XCTAssertEqual(detokenizer.tokens.count, text.utf8.count)
    }

    func testFlushReleasesIncompleteTail() {
        let detokenizer = StreamingDetokenizer(tokenizer: ByteMockTokenizer())
        XCTAssertEqual(detokenizer.append(0x61), "a")
        XCTAssertEqual(detokenizer.append(0xE3), "")
        XCTAssertEqual(detokenizer.flush(), "\u{FFFD}")
        XCTAssertEqual(detokenizer.flush(), "")
    }

    /// 実際のトークナイザで、少しずつ出した文字列が一括の decode と同じになること
    func testMatchesFullDecodeWithRealTokenizers() throws {
        let directory = TestFixtures.root.appendingPathComponent("tokenizers")
        guard let repos = try? FileManager.default.contentsOfDirectory(atPath: directory.path), !repos.isEmpty else {
            throw XCTSkip("tokenizer fixtures not downloaded")
        }
        let samples = [
            "こんにちは、世界！ 今日は🌸がきれいです。",
            "func hello() {\n    print(\"Hi\")  // コメント\n}\n",
            "  leading spaces and ünïcödé, 中文，한국어 😀👍🏽",
        ]
        for repo in repos.sorted() {
            let tokenizer = try HFTokenizer.load(directory: directory.appendingPathComponent(repo))
            for sample in samples {
                let ids = tokenizer.encode(sample, addSpecialTokens: false)
                let expected = tokenizer.decode(ids, skipSpecialTokens: true)
                let detokenizer = StreamingDetokenizer(tokenizer: tokenizer)
                var output = ""
                for id in ids { output += detokenizer.append(id) }
                output += detokenizer.flush()
                XCTAssertEqual(output, expected, "\(repo): \(sample)")
            }
        }
    }
}

final class StopSequenceMatcherTests: XCTestCase {
    func testHoldsPartialMatchesUntilDecided() {
        var matcher = StopSequenceMatcher(stopSequences: ["<|im_end|>", "\nUser:"])
        var result = matcher.feed("Hello <|")
        XCTAssertEqual(result.emit, "Hello ")
        XCTAssertFalse(result.stopped)
        result = matcher.feed("im_")
        XCTAssertEqual(result.emit, "")
        result = matcher.feed("end|> trailing")
        XCTAssertEqual(result.emit, "")
        XCTAssertTrue(result.stopped)
    }

    func testReleasesFalseAlarms() {
        var matcher = StopSequenceMatcher(stopSequences: ["\nUser:"])
        XCTAssertEqual(matcher.feed("line\nUs").emit, "line")
        XCTAssertEqual(matcher.feed("ually fine").emit, "\nUsually fine")
        XCTAssertEqual(matcher.feed("end\n").emit, "end")
        XCTAssertEqual(matcher.flush(), "\n")
    }

    func testStopInsideOneChunkAndEarliestWins() {
        var matcher = StopSequenceMatcher(stopSequences: ["END", "STOP"])
        let result = matcher.feed("abcSTOPxEND")
        XCTAssertEqual(result.emit, "abc")
        XCTAssertTrue(result.stopped)
        var none = StopSequenceMatcher(stopSequences: ["", ""])
        XCTAssertEqual(none.feed("anything <|").emit, "anything <|")
    }
}

final class StopTokenResolverTests: XCTestCase {
    func testCollectsIDsFromAllSources() {
        let tokenizer = CharacterMockTokenizer()
        let generationConfig = Data(#"{"eos_token_id": [151645, 151643]}"#.utf8)
        let config = Data(#"{"eos_token_id": 2}"#.utf8)
        let ids = StopTokenResolver.resolve(tokenizer: tokenizer, eosToken: "X", generationConfigJSON: generationConfig,
                                            configJSON: config, extraTokens: ["Y"])
        XCTAssertEqual(ids, [151645, 151643, 2, Int(("X" as Unicode.Scalar).value), Int(("Y" as Unicode.Scalar).value)])
        let none = StopTokenResolver.resolve(tokenizer: tokenizer, eosToken: nil, generationConfigJSON: Data("oops".utf8),
                                             configJSON: nil, extraTokens: [])
        XCTAssertTrue(none.isEmpty)
    }

    func testRealTokenizersHaveStopTokens() throws {
        let directory = TestFixtures.root.appendingPathComponent("tokenizers")
        guard let repos = try? FileManager.default.contentsOfDirectory(atPath: directory.path), !repos.isEmpty else {
            throw XCTSkip("tokenizer fixtures not downloaded")
        }
        for repo in repos.sorted() {
            let folder = directory.appendingPathComponent(repo)
            let tokenizer = try HFTokenizer.load(directory: folder)
            let ids = StopTokenResolver.resolve(tokenizer: tokenizer, eosToken: tokenizer.eosToken,
                                                generationConfigJSON: try? Data(contentsOf: folder.appendingPathComponent("generation_config.json")),
                                                configJSON: try? Data(contentsOf: folder.appendingPathComponent("config.json")),
                                                extraTokens: [])
            XCTAssertFalse(ids.isEmpty, repo)
        }
    }
}

final class GenerationStatsTests: XCTestCase {
    func testRates() {
        var stats = GenerationStats()
        XCTAssertEqual(stats.prefillTokensPerSecond, 0)
        XCTAssertEqual(stats.decodeTokensPerSecond, 0)
        stats.promptTokens = 120
        stats.reusedPromptTokens = 20
        stats.prefillSeconds = 0.5
        stats.generatedTokens = 30
        stats.decodeSeconds = 2
        XCTAssertEqual(stats.prefillTokensPerSecond, 200)
        XCTAssertEqual(stats.decodeTokensPerSecond, 15)
    }
}
