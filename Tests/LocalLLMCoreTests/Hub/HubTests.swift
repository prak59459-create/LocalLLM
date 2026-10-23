import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

final class ModelInterfaceTests: XCTestCase {
    func metadata(_ name: String) throws -> CoreMLMetadata {
        let url = TestFixtures.root.appendingPathComponent("coreml/\(name).json")
        guard let data = try? Data(contentsOf: url) else { throw XCTSkip("Core ML metadata fixtures not downloaded") }
        return try CoreMLMetadata.parse(data)
    }

    func testStatefulModelsFromHuggingFace() throws {
        let cases: [(String, Int)] = [
            ("finnvoorhees__coreml-Qwen2.5-0.5B-Instruct-4bit", 2048),
            ("finnvoorhees__coreml-Llama-3.2-1B-Instruct-4bit", 2048),
            ("XDGCC__coreml-Qwen3-0.6B", 2048),
            ("pywind__coreml-EXAONE-4.0-1.2B-4bit", 512),
        ]
        for (name, context) in cases {
            let meta = try metadata(name)
            guard case .statefulCausalLM(let ids, let mask, let logits, let maskType, let maxContext) = meta.interface else {
                XCTFail("\(name): \(meta.interface)")
                continue
            }
            XCTAssertEqual(ids, "input_ids", name)
            XCTAssertEqual(mask, "causal_mask", name)
            XCTAssertEqual(logits, "logits", name)
            XCTAssertEqual(maskType, .float16, name)
            XCTAssertEqual(maxContext, context, name)
            XCTAssertEqual(meta.availabilityIOS, "18.0", name)
            XCTAssertEqual(meta.inputs.first?.shapeRange, [1...1, 1...context], name)
            XCTAssertTrue(meta.inputs[0].isFlexible)
        }
        // [24, 1, 2, 2048, 64] × fp16 × (key + value)
        XCTAssertEqual(try metadata("finnvoorhees__coreml-Qwen2.5-0.5B-Instruct-4bit").stateBytes, 24 * 2 * 2048 * 64 * 2 * 2)
    }

    func testUnsupportedModelsExplainWhy() throws {
        let explicitCache = try metadata("aufklarer__Qwen3-0.6B-Chat-CoreML").interface
        guard case .unsupported(let reason) = explicitCache else { return XCTFail("\(explicitCache)") }
        XCTAssertTrue(reason.contains("position_ids"), reason)
        let embeds = try metadata("aufklarer__Qwen3.5-0.8B-Chat-CoreML").interface
        XCTAssertFalse(embeds.isSupported)
        XCTAssertNil(embeds.maxContext)
    }

    func testStatelessClassification() {
        let ids = TensorSpec(name: "input_ids", dataType: .int32, shape: [1, 64], shapeRange: nil)
        let mask = TensorSpec(name: "attention_mask", dataType: .int32, shape: [1, 64], shapeRange: nil)
        let logits = TensorSpec(name: "logits", dataType: .float16, shape: [1, 64, 32000], shapeRange: nil)
        XCTAssertEqual(ModelInterface.classify(inputs: [ids, mask], outputs: [logits], states: []),
                       .statelessCausalLM(inputIDsName: "input_ids", attentionMaskName: "attention_mask", logitsName: "logits",
                                          fixedLength: 64, maxContext: 64))
        let flexible = TensorSpec(name: "tokens", dataType: .int32, shape: [1, 1], shapeRange: [1...1, 1...512])
        XCTAssertEqual(ModelInterface.classify(inputs: [flexible], outputs: [logits], states: []),
                       .statelessCausalLM(inputIDsName: "tokens", attentionMaskName: nil, logitsName: "logits",
                                          fixedLength: nil, maxContext: 512))
        XCTAssertFalse(ModelInterface.classify(inputs: [], outputs: [logits], states: []).isSupported)
        XCTAssertFalse(ModelInterface.classify(inputs: [ids], outputs: [], states: []).isSupported)
        // 状態はあるのにマスクがない
        let state = TensorSpec(name: "key_cache", dataType: .float16, shape: [2, 1, 2, 128, 64], shapeRange: nil)
        XCTAssertFalse(ModelInterface.classify(inputs: [ids], outputs: [logits], states: [state]).isSupported)
    }

    func testMetadataParsingEdgeCases() throws {
        let json = """
        [{"specificationVersion": "9", "inputSchema": [
            {"name": "input_ids", "dataType": "Int32", "shape": "[1, 1]", "shapeRange": "[[1, 1], [1, -1]]"}],
          "outputSchema": [{"name": "out", "dataType": "Float32", "shape": "[]"}]}]
        """
        let meta = try CoreMLMetadata.parse(Data(json.utf8))
        XCTAssertEqual(meta.specificationVersion, 9)
        XCTAssertEqual(meta.inputs[0].upperBound(axis: 1), Int(Int32.max))
        XCTAssertEqual(meta.outputs[0].shape, [])
        XCTAssertThrowsError(try CoreMLMetadata.parse(Data("42".utf8)))
        XCTAssertGreaterThan(MemoryEstimator.estimateBytes(weightBytes: 1_000, stateBytes: 0, vocabSize: 1000, prefillChunk: 64),
                             150_000_000)
    }
}

final class RepoAnalysisTests: XCTestCase {
    func tree(_ name: String) throws -> [HFFileEntry] {
        let url = TestFixtures.root.appendingPathComponent("trees/\(name).json")
        guard let data = try? Data(contentsOf: url) else { throw XCTSkip("tree fixtures not downloaded") }
        return try HFClient.parseTree(data)
    }

    func testSingleCompiledModel() throws {
        let analysis = RepoAnalysis.analyze(files: try tree("finnvoorhees__coreml-Qwen2.5-0.5B-Instruct-4bit"))
        XCTAssertEqual(analysis.variants.map(\.path), ["Qwen2.5-0.5B-Instruct-4bit.mlmodelc"])
        XCTAssertEqual(analysis.variants[0].kind, .compiled)
        XCTAssertEqual(analysis.variants[0].displayName, "Qwen2.5-0.5B-Instruct-4bit")
        XCTAssertGreaterThan(analysis.variants[0].totalBytes, 100_000_000)
        XCTAssertTrue(analysis.hasTokenizer)
        let download = analysis.filesToDownload(for: analysis.variants[0]).map(\.path)
        XCTAssertTrue(download.contains("tokenizer.json"))
        XCTAssertTrue(download.contains("Qwen2.5-0.5B-Instruct-4bit.mlmodelc/weights/weight.bin"))
    }

    func testCompiledVariantsComeFirst() throws {
        let analysis = RepoAnalysis.analyze(files: try tree("aufklarer__Qwen3-0.6B-Chat-CoreML"))
        XCTAssertEqual(analysis.variants.map(\.kind), [.compiled, .compiled, .package, .package])
        XCTAssertEqual(analysis.variants.first?.path, "Qwen3Chat-INT4.mlmodelc")
    }

    func testTokenizerInSubfolder() throws {
        let analysis = RepoAnalysis.analyze(files: try tree("mlboydaisuke__qwen2.5-0.5b-coreml"))
        XCTAssertEqual(analysis.variants.map(\.path), ["model.mlpackage"])
        XCTAssertTrue(analysis.hasTokenizer)
        XCTAssertTrue(analysis.filesToDownload(for: analysis.variants[0]).map(\.path).contains("hf_model/tokenizer.json"))
    }

    func testNoTokenizer() throws {
        let analysis = RepoAnalysis.analyze(files: try tree("apple__mistral-coreml"))
        XCTAssertEqual(analysis.variants.count, 2)
        XCTAssertFalse(analysis.hasTokenizer)
    }

    func testPicksClosestSupportFiles() {
        let files = [
            HFFileEntry(path: "a/model.mlmodelc/coremldata.bin", size: 10, lfsSHA256: nil),
            HFFileEntry(path: "b/model.mlmodelc/coremldata.bin", size: 10, lfsSHA256: nil),
            HFFileEntry(path: "tokenizer.json", size: 1, lfsSHA256: nil),
            HFFileEntry(path: "b/tokenizer.json", size: 2, lfsSHA256: nil),
            HFFileEntry(path: "b/model.mlmodelc/tokenizer.json", size: 3, lfsSHA256: nil),
        ]
        let analysis = RepoAnalysis.analyze(files: files)
        XCTAssertEqual(analysis.supportFiles.map(\.path), ["b/tokenizer.json", "tokenizer.json"])
        let a = analysis.variants.first { $0.path == "a/model.mlmodelc" }!
        let b = analysis.variants.first { $0.path == "b/model.mlmodelc" }!
        XCTAssertEqual(analysis.filesToDownload(for: a).map(\.path), ["a/model.mlmodelc/coremldata.bin", "tokenizer.json"])
        XCTAssertEqual(analysis.filesToDownload(for: b).map(\.path),
                       ["b/model.mlmodelc/coremldata.bin", "b/model.mlmodelc/tokenizer.json", "b/tokenizer.json"])
        XCTAssertNil(RepoAnalysis.modelRoot(of: "model.mlmodelc"))
        XCTAssertEqual(RepoAnalysis.modelRoot(of: "x/y.mlpackage/Data/z.mlmodelc/w"), "x/y.mlpackage")
    }
}

final class HFClientTests: XCTestCase {
    func testHelpers() {
        let link = #"<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel="next", <https://x>; rel="prev""#
        XCTAssertEqual(HFClient.nextLink(link)?.absoluteString, "https://huggingface.co/api/models/x/tree/main?cursor=abc")
        XCTAssertNil(HFClient.nextLink(#"<https://x>; rel="prev""#))
        XCTAssertTrue(HFClient.isGated("auto"))
        XCTAssertTrue(HFClient.isGated(true))
        XCTAssertFalse(HFClient.isGated(false))
        XCTAssertFalse(HFClient.isGated("false"))
        XCTAssertFalse(HFClient.isGated(nil))
        XCTAssertNotNil(HFClient.parseDate("2024-10-01T12:34:56.000Z"))
        XCTAssertNotNil(HFClient.parseDate("2024-10-01T12:34:56Z"))
        XCTAssertNil(HFClient.parseDate("yesterday"))
    }

    func testURLs() {
        let client = HFClient(token: "  hf_abc \n")
        XCTAssertEqual(client.token, "hf_abc")
        XCTAssertEqual(client.authorizationHeaders["Authorization"], "Bearer hf_abc")
        XCTAssertNil(HFClient(token: "   ").authorizationHeaders["Authorization"])
        XCTAssertEqual(client.resolveURL(repo: "org/repo", path: "a b/モデル.mlmodelc/weight#1.bin", revision: "main").absoluteString,
                       "https://huggingface.co/org/repo/resolve/main/a%20b/%E3%83%A2%E3%83%87%E3%83%AB.mlmodelc/weight%231.bin")
        XCTAssertEqual(client.treeURL(repo: "org/repo", revision: "main").absoluteString,
                       "https://huggingface.co/api/models/org/repo/tree/main?recursive=true")
    }

    func testParseTreeSkipsFoldersAndReadsLFSSize() throws {
        let json = """
        [{"type": "directory", "path": "m.mlmodelc"},
         {"type": "file", "path": "m.mlmodelc/weights/weight.bin", "size": 134, "lfs": {"oid": "abc", "size": 999}},
         {"type": "file", "path": "config.json", "size": 12}]
        """
        let files = try HFClient.parseTree(Data(json.utf8))
        XCTAssertEqual(files, [HFFileEntry(path: "m.mlmodelc/weights/weight.bin", size: 999, lfsSHA256: "abc"),
                               HFFileEntry(path: "config.json", size: 12, lfsSHA256: nil)])
        XCTAssertThrowsError(try HFClient.parseTree(Data("{}".utf8)))
    }

    func testHTTPErrorsAreReadable() {
        let url = URL(string: "https://huggingface.co/x")!
        for status in [401, 404, 429, 503, 418] {
            XCTAssertFalse((HFError.http(status: status, url: url).errorDescription ?? "").isEmpty)
        }
        XCTAssertTrue(ModelDownloadJob.isRetryable(HFError.http(status: 503, url: url)))
        XCTAssertTrue(ModelDownloadJob.isRetryable(HFError.network("offline")))
        XCTAssertFalse(ModelDownloadJob.isRetryable(HFError.http(status: 404, url: url)))
    }
}

/// 本物の Hugging Face に接続するテスト（環境変数 LOCALLLM_LIVE_TESTS=1 のときだけ）
final class HFLiveTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["LOCALLLM_LIVE_TESTS"] == "1" else {
            throw XCTSkip("set LOCALLLM_LIVE_TESTS=1 to run tests against huggingface.co")
        }
    }

    func testSearchInfoAndTree() async throws {
        let client = HFClient()
        let results = try await client.searchModels(query: "qwen", coreMLOnly: true, textGenerationOnly: false, limit: 5)
        XCTAssertFalse(results.isEmpty)
        XCTAssertTrue(results.allSatisfy { $0.tags.contains("coreml") || $0.libraryName == "coreml" }, "\(results.map(\.id))")
        let repo = "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit"
        let info = try await client.modelInfo(repo: repo)
        XCTAssertEqual(info.id, repo)
        XCTAssertTrue(info.siblings.contains("tokenizer.json"))
        let files = try await client.listFiles(repo: repo, revision: "main")
        let analysis = RepoAnalysis.analyze(files: files)
        XCTAssertEqual(analysis.variants.first?.path, "Qwen2.5-0.5B-Instruct-4bit.mlmodelc")
        let config = try await client.fetchData(repo: repo, path: "config.json", revision: "main")
        XCTAssertNotNil(try JSONSerialization.jsonObject(with: config) as? [String: Any])
    }

    func testMissingRepoIs404() async {
        do {
            _ = try await HFClient().modelInfo(repo: "this-user-does-not-exist-123/nothing")
            XCTFail("expected an error")
        } catch HFError.http(let status, _) {
            XCTAssertTrue([401, 404].contains(status), "\(status)")
        } catch {
            XCTFail("\(error)")
        }
    }

    /// LFS のファイル（CDN へのリダイレクトあり）を途中から再開できること
    func testResumeThroughRedirect() async throws {
        let client = HFClient()
        let repo = "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit"
        let files = try await client.listFiles(repo: repo, revision: "main")
        // 小さめの LFS ファイルを選ぶ
        let lfs = try XCTUnwrap(files.filter { $0.lfsSHA256 != nil && $0.size < 20_000_000 }.min { $0.size < $1.size })
        let url = client.resolveURL(repo: repo, path: lfs.path, revision: "main")
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("live-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let full = folder.appendingPathComponent("full.bin")
        let downloader = ResumableDownloader()
        try await downloader.download(from: url, to: full, expectedSize: lfs.size, headers: client.authorizationHeaders) { _ in }
        let expected = try Data(contentsOf: full)
        XCTAssertEqual(Int64(expected.count), lfs.size)
        // 半分だけある .part から続きを取る
        let resumed = folder.appendingPathComponent("resumed.bin")
        try expected.prefix(expected.count / 2).write(to: resumed.appendingPathExtension("part"))
        try await downloader.download(from: url, to: resumed, expectedSize: lfs.size, headers: client.authorizationHeaders) { _ in }
        XCTAssertEqual(try Data(contentsOf: resumed), expected)
        XCTAssertFalse(FileManager.default.fileExists(atPath: resumed.appendingPathExtension("part").path))
    }
}
