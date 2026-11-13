import XCTest
@testable import LocalLLMCore

/// GGUF に入った語彙から作ったトークナイザが、元のモデルの tokenizer.json と同じ結果になるか。
/// フィクスチャ（./Tools/fetch_fixtures.sh が GGUF の先頭 16MB と tokenizer.json を取ってくる）が無ければ飛ばす。
final class GGUFTokenizerTests: XCTestCase {
    private let samples = [
        "Hello, world! How are you doing today?",
        "こんにちは、世界。今日はいい天気ですね。日本語のテキストも正しく分けられるか確かめます。",
        "func add(_ a: Int, _ b: Int) -> Int {\n    return a + b  // 足し算\n}\n",
        "  leading spaces and\ttabs\n\nnewlines   and trailing   ",
        "Numbers 1234567 and 3.14159, emoji 😀🎉, symbols <tag> & \"quotes\" 'single'.",
        "MixedCASE words, URLs like https://example.com/path?q=1, and snake_case_names.",
    ]

    /// GGUF の先頭部分 → 元のトークナイザのフィクスチャ → 特別なトークンを含む文
    private let pairs: [(header: String, tokenizer: String, chat: String)] = [
        ("Qwen__Qwen2.5-0.5B-Instruct-GGUF", "finnvoorhees__coreml-Qwen2.5-0.5B-Instruct-4bit", "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"),
        ("bartowski__Llama-3.2-1B-Instruct-GGUF", "finnvoorhees__coreml-Llama-3.2-1B-Instruct-4bit",
         "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"),
        ("bartowski__SmolLM2-135M-Instruct-GGUF", "finnvoorhees__coreml-SmolLM2-360M-Instruct-4bit", "<|im_start|>user\nHi 42<|im_end|>\n"),
        ("bartowski__DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "finnvoorhees__coreml-DeepSeek-R1-Distill-Qwen-1.5B-4bit",
         "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜><think>\n"),
        ("unsloth__Qwen3-0.6B-GGUF", "XDGCC__coreml-Qwen3-0.6B", "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        ("TheBloke__TinyLlama-1.1B-Chat-v1.0-GGUF", "TinyLlama__TinyLlama-1.1B-Chat-v1.0", "<|user|>\nHi</s>\n<|assistant|>\n"),
        ("microsoft__Phi-3-mini-4k-instruct-gguf", "microsoft__Phi-3-mini-4k-instruct", "<|user|>\nHi<|end|>\n<|assistant|>\n"),
        ("bartowski__gemma-2-2b-it-GGUF", "unsloth__gemma-2-2b-it", "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n"),
        ("bartowski__Mistral-7B-Instruct-v0.3-GGUF", "unsloth__mistral-7b-instruct-v0.3", "<s>[INST] Hi [/INST] Hello</s>"),
    ]

    private func header(_ name: String) -> GGUFFile? {
        let url = TestFixtures.root.appendingPathComponent("gguf-headers").appendingPathComponent(name + ".gguf")
        guard let data = try? Data(contentsOf: url, options: [.alwaysMapped]) else { return nil }
        return try? GGUFFile.parse(data)
    }

    func testTokenizersBuiltFromGGUFMatchTheOriginals() throws {
        var checked = 0
        for pair in pairs {
            let folder = TestFixtures.root.appendingPathComponent("tokenizers").appendingPathComponent(pair.tokenizer)
            guard let file = header(pair.header),
                  let originalJSON = try? Data(contentsOf: folder.appendingPathComponent("tokenizer.json")) else { continue }
            let original = try HFTokenizer(tokenizerJSON: originalJSON,
                                           tokenizerConfigJSON: try? Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json")))
            let files = try GGUFTokenizer.files(from: file)
            let built = try HFTokenizer(tokenizerJSON: files.tokenizerJSON, tokenizerConfigJSON: files.tokenizerConfigJSON)
            // GGUF の語彙は、埋め込みの行数に合わせて使わないトークンで埋めてあることがある
            XCTAssertGreaterThanOrEqual(built.vocabSize, original.vocabSize, pair.header)
            for text in samples + [pair.chat] {
                let expected = original.encode(text, addSpecialTokens: false)
                let actual = built.encode(text, addSpecialTokens: false)
                XCTAssertEqual(actual, expected, "\(pair.header) encode: \(text)")
                XCTAssertEqual(built.decode(actual, skipSpecialTokens: false), original.decode(expected, skipSpecialTokens: false),
                               "\(pair.header) decode: \(text)")
            }
            if original.bosToken != nil { XCTAssertEqual(built.bosToken, original.bosToken, pair.header) }
            XCTAssertEqual(built.eosToken, original.eosToken, pair.header)
            XCTAssertNotNil(built.chatTemplate, pair.header)
            checked += 1
        }
        if checked == 0 { throw XCTSkip("GGUF header fixtures missing (run ./Tools/fetch_fixtures.sh)") }
        print("GGUF tokenizers checked: \(checked)")
    }

    func testModelConfigsOfRealFiles() throws {
        var checked = 0
        for pair in pairs + [("bartowski__google_gemma-3-1b-it-GGUF", "", "")] {
            guard let file = header(pair.header) else { continue }
            let config = try GGUFModelConfig(file: file)
            XCTAssertGreaterThan(config.layerCount, 0, pair.header)
            XCTAssertEqual(config.vocabSize, file.metadata["tokenizer.ggml.tokens"]?.count, pair.header)
            XCTAssertGreaterThan(GGUFWeightStore.estimatedBytes(file: file, config: config, compact: true), file.parameterCount / 2, pair.header)
            checked += 1
        }
        if checked == 0 { throw XCTSkip("GGUF header fixtures missing (run ./Tools/fetch_fixtures.sh)") }
        if let gemma = header("bartowski__google_gemma-3-1b-it-GGUF") {
            let config = try GGUFModelConfig(file: gemma)
            XCTAssertEqual(config.family, .gemma3)
            XCTAssertEqual(config.slidingWindow, 512)
            XCTAssertEqual(config.slidingLayers.prefix(6), [true, true, true, true, true, false])
        }
    }
}

/// GGUF のリポジトリの見分け方・ファイル名の読み方・先頭部分からの情報。
final class GGUFRepositoryTests: XCTestCase {
    func testFileNames() {
        XCTAssertEqual(GGUFNaming.quantization(of: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertEqual(GGUFNaming.quantization(of: "qwen2.5-0.5b-instruct-q8_0.gguf"), "Q8_0")
        XCTAssertEqual(GGUFNaming.quantization(of: "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf"), "Q5_K_S")
        XCTAssertEqual(GGUFNaming.quantization(of: "SmolLM2-135M-Instruct-IQ3_XS.gguf"), "IQ3_XS")
        XCTAssertEqual(GGUFNaming.quantization(of: "dir/SmolLM2-135M-Instruct-f16.gguf"), "F16")
        XCTAssertNil(GGUFNaming.quantization(of: "model.gguf"))
        XCTAssertTrue(GGUFNaming.isStandaloneModel("a/Model-Q4_K_M.gguf"))
        XCTAssertFalse(GGUFNaming.isStandaloneModel("mmproj-model-f16.gguf"))
        XCTAssertFalse(GGUFNaming.isStandaloneModel("Model-Q4_K_M-00001-of-00003.gguf"))
        XCTAssertFalse(GGUFNaming.isStandaloneModel("Model.imatrix"))
        XCTAssertTrue(GGUFNaming.isReadable(quantization: "Q4_K_M"))
        XCTAssertFalse(GGUFNaming.isReadable(quantization: "IQ1_S"))
        XCTAssertLessThan(GGUFNaming.rank(of: "m-Q4_K_M.gguf"), GGUFNaming.rank(of: "m-Q8_0.gguf"))
    }

    func testRepositoryWithGGUFFiles() {
        func entry(_ path: String, _ size: Int64) -> HFFileEntry { HFFileEntry(path: path, size: size, lfsSHA256: nil) }
        let analysis = RepoAnalysis.analyze(files: [
            entry("README.md", 10), entry("config.json", 20),
            entry("Model-Q8_0.gguf", 800), entry("Model-Q4_K_M.gguf", 400), entry("Model-Q2_K.gguf", 200),
            entry("Model-f16-00001-of-00002.gguf", 900), entry("mmproj-Model-f16.gguf", 50),
        ], repoName: "Model-GGUF")
        XCTAssertTrue(analysis.hasGGUF)
        XCTAssertEqual(analysis.variants.map(\.path), ["Model-Q4_K_M.gguf", "Model-Q8_0.gguf", "Model-Q2_K.gguf"])
        XCTAssertTrue(analysis.variants.allSatisfy { $0.kind == .gguf })
        // GGUF はトークナイザが中に入っているので、付属ファイルは取らない
        XCTAssertEqual(analysis.filesToDownload(for: analysis.variants[0]).map(\.path), ["Model-Q4_K_M.gguf"])
    }

    func testSummaryOfTinyModel() throws {
        let file = try GGUFFile.open(GGUFFixtures.root.appendingPathComponent("tiny-qwen2.gguf"))
        let summary = GGUFSummary(file: file)
        XCTAssertTrue(summary.isSupported)
        XCTAssertEqual(summary.architecture, "qwen2")
        XCTAssertEqual(summary.contextLength, 64)
        XCTAssertEqual(summary.defaultContext, 256)
        XCTAssertGreaterThan(summary.storeBytes, Int64(summary.parameterCount))
        XCTAssertGreaterThan(summary.memoryBytes(context: 64), summary.storeBytes)
        // 変換後のファイルの大きさは、見積もりどおり
        let destination = GGUFFixtures.temporaryFile("summary.llw")
        defer { try? FileManager.default.removeItem(at: destination) }
        try GGUFConverter.convert(gguf: GGUFFixtures.root.appendingPathComponent("tiny-qwen2.gguf"), to: destination)
        let size = try XCTUnwrap(FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? NSNumber).int64Value
        XCTAssertLessThanOrEqual(size, summary.storeBytes)
        XCTAssertGreaterThan(size, summary.storeBytes - 16384)
    }

    func testUnsupportedArchitectureIsExplained() throws {
        // 構造の名前を書き換えた GGUF は、動かせない理由が付く
        var data = try Data(contentsOf: GGUFFixtures.root.appendingPathComponent("tiny-llama.gguf"))
        let needle = Data("llama.block_count".utf8)
        let range = try XCTUnwrap(data.range(of: needle))
        data.replaceSubrange(range, with: Data("llamb.block_count".utf8))
        let summary = GGUFSummary(file: try GGUFFile.parse(data))
        XCTAssertFalse(summary.isSupported)
        XCTAssertNotNil(summary.unsupportedReason)
    }

    /// Hugging Face から、GGUF の先頭部分だけを取ってきて読む（ネットにつながる環境だけ）
    func testLiveHeaderFetch() async throws {
        guard ProcessInfo.processInfo.environment["LOCALLLM_LIVE_TESTS"] == "1" else { throw XCTSkip("live tests are off") }
        let client = HFClient()
        let file = try await client.fetchGGUFHeader(repo: "bartowski/SmolLM2-135M-Instruct-GGUF", path: "SmolLM2-135M-Instruct-Q4_K_M.gguf")
        let summary = GGUFSummary(file: file)
        XCTAssertTrue(summary.isSupported, summary.unsupportedReason ?? "")
        XCTAssertEqual(summary.architecture, "llama")
        XCTAssertEqual(summary.layerCount, 30)
        XCTAssertTrue(summary.hasChatTemplate)
        XCTAssertEqual(summary.parameterLabel, "135M")
    }
}
