import Foundation
import XCTest
@testable import LocalLLMCore

/// tokenizer.json が無いときに、ほかの形式から作ったトークナイザが、本物の tokenizer.json と同じ結果になるか。
/// フィクスチャ（./Tools/fetch_fixtures.sh）が無ければスキップする。
final class TokenizerSynthesizerTests: XCTestCase {
    private let samples = [
        "Hello, world! How are you doing today?",
        "こんにちは、世界。今日はいい天気ですね。日本語のテキストも正しく分けられるか確かめます。",
        "func add(_ a: Int, _ b: Int) -> Int {\n    return a + b  // 足し算\n}\n",
        "  leading spaces and\ttabs\n\nnewlines   and trailing   ",
        "Numbers 1234567 and 3.14159, emoji 😀🎉, symbols <tag> & \"quotes\" 'single'.",
        "MixedCASE words, URLs like https://example.com/path?q=1, and snake_case_names.",
    ]

    private func fixture(_ name: String) throws -> URL {
        let folder = TestFixtures.root.appendingPathComponent("tokenizers").appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("fixture missing: \(name) (run ./Tools/fetch_fixtures.sh)")
        }
        return folder
    }

    private func assertSameEncoding(_ synthesized: HFTokenizer, _ original: HFTokenizer, file: StaticString = #filePath, line: UInt = #line) {
        for text in samples {
            let expected = original.encode(text, addSpecialTokens: false)
            let actual = synthesized.encode(text, addSpecialTokens: false)
            XCTAssertEqual(actual, expected, "encode: \(text)", file: file, line: line)
            XCTAssertEqual(synthesized.decode(actual, skipSpecialTokens: false), original.decode(expected, skipSpecialTokens: false),
                           "decode: \(text)", file: file, line: line)
        }
    }

    private func original(_ folder: URL) throws -> HFTokenizer {
        try HFTokenizer(tokenizerJSON: Data(contentsOf: folder.appendingPathComponent("tokenizer.json")),
                        tokenizerConfigJSON: try? Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json")))
    }

    /// SentencePiece（tokenizer.model）から作る
    func testSentencePieceMatchesTokenizerJSON() throws {
        var checked = 0
        for name in ["TinyLlama__TinyLlama-1.1B-Chat-v1.0", "unsloth__gemma-2-2b-it"] {
            let folder = TestFixtures.root.appendingPathComponent("tokenizers").appendingPathComponent(name)
            guard let model = try? Data(contentsOf: folder.appendingPathComponent("tokenizer.model")) else { continue }
            let config = try? Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json"))
            let json = try TokenizerSynthesizer.tokenizerJSON(from: .sentencePiece(model), tokenizerConfig: config)
            let synthesized = try HFTokenizer(tokenizerJSON: json, tokenizerConfigJSON: config)
            let reference = try original(folder)
            XCTAssertEqual(synthesized.vocabSize, reference.vocabSize, name)
            assertSameEncoding(synthesized, reference)
            checked += 1
        }
        if checked == 0 { throw XCTSkip("tokenizer.model fixtures missing (run ./Tools/fetch_fixtures.sh)") }
    }

    /// vocab.json + merges.txt から作る（Qwen の tokenizer.json から取り出して作り直す）
    func testVocabAndMergesMatchTokenizerJSON() throws {
        let folder = try fixture("finnvoorhees__coreml-Qwen2.5-0.5B-Instruct-4bit")
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))) as? [String: Any])
        let model = try XCTUnwrap(root["model"] as? [String: Any])
        let vocab = try JSONSerialization.data(withJSONObject: try XCTUnwrap(model["vocab"]))
        let mergeLines = (model["merges"] as? [Any] ?? []).compactMap { entry -> String? in
            if let text = entry as? String { return text }
            if let pair = entry as? [String], pair.count == 2 { return pair.joined(separator: " ") }
            return nil
        }
        let merges = "#version: 0.2\n" + mergeLines.joined(separator: "\n") + "\n"
        let config = try? Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json"))
        let json = try TokenizerSynthesizer.tokenizerJSON(from: .vocabMerges(vocab: vocab, merges: merges), tokenizerConfig: config)
        let synthesized = try HFTokenizer(tokenizerJSON: json, tokenizerConfigJSON: config)
        let reference = try original(folder)
        XCTAssertEqual(synthesized.vocabSize, reference.vocabSize)
        assertSameEncoding(synthesized, reference)
        // 追加トークン（<|im_start|> など）も 1 つのトークンになる
        XCTAssertEqual(synthesized.encode("<|im_start|>user", addSpecialTokens: false), reference.encode("<|im_start|>user", addSpecialTokens: false))
    }

    /// tiktoken 形式から作る（Llama 3.2 の tokenizer.json から取り出して作り直す）
    func testTiktokenMatchesTokenizerJSON() throws {
        let folder = try fixture("finnvoorhees__coreml-Llama-3.2-1B-Instruct-4bit")
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))) as? [String: Any])
        let vocab = try XCTUnwrap((root["model"] as? [String: Any])?["vocab"] as? [String: Int])
        var reverse: [Unicode.Scalar: UInt8] = [:]
        for (byte, scalar) in TokenizerSynthesizer.byteToUnicode.enumerated() { reverse[scalar] = UInt8(byte) }
        var lines: [String] = []
        for (token, id) in vocab.sorted(by: { $0.value < $1.value }) {
            let bytes = token.unicodeScalars.compactMap { reverse[$0] }
            guard bytes.count == token.unicodeScalars.count else { continue }
            lines.append(Data(bytes).base64EncodedString() + " \(id)")
        }
        let text = lines.joined(separator: "\n")
        XCTAssertTrue(TokenizerSynthesizer.isTiktokenText(Data(text.utf8)))
        let config = try? Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json"))
        let json = try TokenizerSynthesizer.tokenizerJSON(from: .tiktoken(text), tokenizerConfig: config)
        let synthesized = try HFTokenizer(tokenizerJSON: json, tokenizerConfigJSON: config)
        let reference = try original(folder)
        assertSameEncoding(synthesized, reference)
        XCTAssertEqual(synthesized.tokenID(for: "<|eot_id|>"), reference.tokenID(for: "<|eot_id|>"))
    }

    func testFileDetectionAndFallbackCandidates() {
        XCTAssertTrue(TokenizerSynthesizer.canBuild(fileNames: ["tokenizer.model"]))
        XCTAssertTrue(TokenizerSynthesizer.canBuild(fileNames: ["vocab.json", "merges.txt"]))
        XCTAssertFalse(TokenizerSynthesizer.canBuild(fileNames: ["vocab.json"]))
        XCTAssertTrue(TokenizerSynthesizer.canBuild(fileNames: ["qwen.tiktoken"]))
        XCTAssertFalse(TokenizerSynthesizer.isTiktokenText(Data([0x0A, 0x05, 0x3C, 0x75])))

        XCTAssertEqual(TokenizerFallback.candidates(repoID: "someone/coreml-Qwen2.5-Coder-1.5B-Instruct-4bit"),
                       ["Qwen/Qwen2.5-Coder-1.5B-Instruct"])
        XCTAssertEqual(TokenizerFallback.candidates(repoID: "x/my-model", baseModels: ["meta-llama/Llama-3.2-1B-Instruct"]),
                       ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"])
        XCTAssertEqual(TokenizerFallback.candidates(repoID: "anemll/anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5"), ["Qwen/Qwen3-0.6B"])
        XCTAssertTrue(TokenizerFallback.candidates(repoID: "x/unknown-model").isEmpty)
    }
}

/// 本物の Hugging Face から、トークナイザの無いモデルのためにトークナイザを取ってくる（LOCALLLM_LIVE_TESTS=1 のときだけ）
final class TokenizerFallbackLiveTests: XCTestCase {
    func testFetchesTokenizerForAFamily() async throws {
        guard ProcessInfo.processInfo.environment["LOCALLLM_LIVE_TESTS"] == "1" else {
            throw XCTSkip("set LOCALLLM_LIVE_TESTS=1 to run tests against huggingface.co")
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("tokfallback-\(UUID().uuidString)")
        let candidates = TokenizerFallback.candidates(repoID: "nobody/coreml-SmolLM2-135M-Instruct-without-tokenizer")
        let repo = await TokenizerFallback.fetch(client: HFClient(), candidates: candidates, into: folder)
        guard let repo else { throw XCTSkip("could not reach huggingface.co") }
        XCTAssertEqual(repo, "HuggingFaceTB/SmolLM2-360M-Instruct")
        let tokenizer = try HFTokenizer.load(directory: folder)
        XCTAssertGreaterThan(tokenizer.vocabSize, 1000)
        XCTAssertNotNil(tokenizer.chatTemplate)
    }
}
