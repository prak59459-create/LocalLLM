import Foundation
import XCTest
@testable import LocalLLMCore

final class SimpleYAMLTests: XCTestCase {
    static let anemllChunked = """
    model_info:
      name: anemll-Qwen-Qwen3-1.7B-ctx2048
      version: 0.3.5
      description: |
        Demonstarates running Qwen on Apple Neural Engine
        Context length: 2048
      license: MIT
      architecture: qwen3
      parameters:
        context_length: 2048
        batch_size: 64
        lut_ffn: 6
        num_chunks: 2
        model_prefix: qwen
        embeddings: qwen_embeddings.mlmodelc
        lm_head: qwen_lm_head_lut6.mlmodelc
        ffn: qwen_FFN_PF_lut6_chunk_01of02.mlmodelc
        split_lm_head: 16
        prefill_dynamic_slice: true

    # =====
    # command_line: "./convert.sh --chunk 2"
    """

    static let anemllMonolithic = """
    model_info:
      name: anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic
      model_type: monolithic
      parameters:
        context_length: 2048
        batch_size: 64
        monolithic_model: qwen25_monolithic_full_lut6.mlmodelc
        split_lm_head: 16
        vocab_size: 151936
        lm_head_chunk_sizes: [9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496, 9496]
        functions:
          - infer
          - prefill
    """

    func testParsesNestedMapsListsAndBlocks() {
        let yaml = SimpleYAML.parse(Self.anemllChunked)
        let parameters = yaml["model_info"]?["parameters"]
        XCTAssertEqual(yaml["model_info"]?["name"]?.string, "anemll-Qwen-Qwen3-1.7B-ctx2048")
        XCTAssertEqual(yaml["model_info"]?["license"]?.string, "MIT")
        XCTAssertEqual(parameters?["context_length"]?.int, 2048)
        XCTAssertEqual(parameters?["prefill_dynamic_slice"]?.bool, true)
        XCTAssertEqual(parameters?["ffn"]?.string, "qwen_FFN_PF_lut6_chunk_01of02.mlmodelc")
        let mono = SimpleYAML.parse(Self.anemllMonolithic)["model_info"]?["parameters"]
        XCTAssertEqual(mono?["lm_head_chunk_sizes"]?.list?.count, 16)
        XCTAssertEqual(mono?["functions"]?.list?.compactMap(\.string), ["infer", "prefill"])
        XCTAssertEqual(SimpleYAML.parse("a: 'x'\nb: \"y\" # comment\n")["b"]?.string, "y")
    }
}

final class SplitModelAnalysisTests: XCTestCase {
    func entry(_ path: String, _ size: Int64 = 10) -> HFFileEntry {
        HFFileEntry(path: path, size: size, lfsSHA256: nil)
    }

    func modelFiles(_ root: String, size: Int64 = 1000) -> [HFFileEntry] {
        [entry("\(root)/coremldata.bin"), entry("\(root)/model.mil"), entry("\(root)/metadata.json"),
         entry("\(root)/weights/weight.bin", size)]
    }

    func testANEMLLChunkedWithMeta() throws {
        var files = [entry("meta.yaml"), entry("tokenizer.json"), entry("tokenizer_config.json"), entry("config.json"), entry("chat.py")]
        for root in ["qwen_embeddings.mlmodelc", "qwen_FFN_PF_lut6_chunk_01of02.mlmodelc", "qwen_FFN_PF_lut6_chunk_02of02.mlmodelc",
                     "qwen_lm_head.mlmodelc", "qwen_lm_head_lut6.mlmodelc"] {
            files += modelFiles(root)
        }
        let analysis = RepoAnalysis.analyze(files: files, metaFiles: ["meta.yaml": SimpleYAMLTests.anemllChunked], repoName: "anemll-qwen")
        XCTAssertEqual(analysis.variants.count, 1, "\(analysis.variants.map(\.path))")
        let variant = try XCTUnwrap(analysis.variants.first)
        XCTAssertEqual(variant.kind, .pipeline)
        let pipeline = try XCTUnwrap(variant.pipeline)
        XCTAssertEqual(pipeline.embeddings, "qwen_embeddings.mlmodelc")
        XCTAssertEqual(pipeline.chunks, ["qwen_FFN_PF_lut6_chunk_01of02.mlmodelc", "qwen_FFN_PF_lut6_chunk_02of02.mlmodelc"])
        XCTAssertEqual(pipeline.lmHead, "qwen_lm_head_lut6.mlmodelc")
        XCTAssertEqual(pipeline.contextLength, 2048)
        XCTAssertEqual(pipeline.batchSize, 64)
        XCTAssertFalse(pipeline.isMonolithic)
        // 使わない qwen_lm_head.mlmodelc はダウンロードしない
        XCTAssertFalse(variant.files.contains { $0.path.hasPrefix("qwen_lm_head.mlmodelc/") })
        XCTAssertTrue(variant.files.contains { $0.path == "meta.yaml" })
        let download = analysis.filesToDownload(for: variant).map(\.path)
        XCTAssertTrue(download.contains("tokenizer.json"))
        XCTAssertEqual(RepoAnalysis.metaFilePaths(in: files), ["meta.yaml"])
    }

    func testANEMLLMonolithic() throws {
        let files = [entry("meta.yaml"), entry("tokenizer.json")] + modelFiles("qwen25_monolithic_full_lut6.mlmodelc")
        let analysis = RepoAnalysis.analyze(files: files, metaFiles: ["meta.yaml": SimpleYAMLTests.anemllMonolithic])
        let variant = try XCTUnwrap(analysis.variants.first)
        XCTAssertEqual(analysis.variants.count, 1)
        XCTAssertEqual(variant.kind, .pipeline)
        XCTAssertEqual(variant.pipeline?.isMonolithic, true)
        XCTAssertEqual(variant.pipeline?.chunks, ["qwen25_monolithic_full_lut6.mlmodelc"])
        XCTAssertEqual(variant.pipeline?.argmaxOffsets(count: 16)[1], 9496)
    }

    func testGuessesSplitModelWithoutMeta() throws {
        var files = [entry("tokenizer.json")]
        for root in ["m/llama_embeddings.mlmodelc", "m/llama_FFN_PF_lut6_chunk_02of02.mlmodelc",
                     "m/llama_FFN_PF_lut6_chunk_01of02.mlmodelc", "m/llama_lm_head_lut6.mlmodelc"] {
            files += modelFiles(root)
        }
        let analysis = RepoAnalysis.analyze(files: files)
        let pipeline = try XCTUnwrap(analysis.variants.first?.pipeline)
        XCTAssertEqual(pipeline.chunks, ["m/llama_FFN_PF_lut6_chunk_01of02.mlmodelc", "m/llama_FFN_PF_lut6_chunk_02of02.mlmodelc"])
        XCTAssertEqual(pipeline.embeddings, "m/llama_embeddings.mlmodelc")
        // チャンクが欠けていたら分割モデルにしない
        let missing = files.filter { !$0.path.contains("chunk_02of02") }
        XCTAssertNil(RepoAnalysis.analyze(files: missing).variants.first { $0.kind == .pipeline })
    }

    func testLooseCompiledFoldersAndArchives() throws {
        let files = [entry("tokenizer.json"), entry("Qwen/coremldata.bin"), entry("Qwen/model.mil"),
                     entry("Qwen/weights/weight.bin", 500), entry("Qwen/analytics/coremldata.bin"),
                     entry("Pkg/Manifest.json"), entry("Pkg/Data/com.apple.CoreML/model.mlmodel"),
                     entry("Other.mlmodelc.zip", 2000), entry("notes.zip")]
        let analysis = RepoAnalysis.analyze(files: files)
        let compiled = try XCTUnwrap(analysis.variants.first { $0.kind == .compiled })
        XCTAssertEqual(compiled.path, "Qwen")
        XCTAssertEqual(compiled.displayName, "Qwen")
        XCTAssertEqual(compiled.files.count, 4)
        XCTAssertEqual(compiled.localPath(for: "Qwen/weights/weight.bin"), "Qwen.mlmodelc/weights/weight.bin")
        XCTAssertEqual(compiled.localPath(for: "tokenizer.json"), "tokenizer.json")
        let package = try XCTUnwrap(analysis.variants.first { $0.kind == .package })
        XCTAssertEqual(package.localPath(for: "Pkg/Manifest.json"), "Pkg.mlpackage/Manifest.json")
        XCTAssertEqual(analysis.variants.filter { $0.kind == .archive }.map(\.path), ["Other.mlmodelc.zip"])
        XCTAssertTrue(analysis.hasTokenizer)

        // リポジトリ直下に置かれたコンパイル済みモデル
        let root = RepoAnalysis.analyze(files: [entry("coremldata.bin"), entry("model.mil"), entry("weights/weight.bin"),
                                                entry("tokenizer.json"), entry("README.md")], repoName: "tiny")
        let rootModel = try XCTUnwrap(root.variants.first)
        XCTAssertEqual(rootModel.files.map(\.path), ["coremldata.bin", "model.mil", "weights/weight.bin"])
        XCTAssertEqual(rootModel.localPath(for: "weights/weight.bin"), "tiny.mlmodelc/weights/weight.bin")
        XCTAssertEqual(rootModel.localPath(for: "tokenizer.json"), "tokenizer.json")
    }

    /// 実際の ANEMLL のリポジトリ（フィクスチャがあれば）
    func testRealANEMLLTree() throws {
        let url = TestFixtures.root.appendingPathComponent("trees/anemll__anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4.json")
        let metaURL = TestFixtures.root.appendingPathComponent("trees/anemll__anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4.meta.yaml")
        guard let data = try? Data(contentsOf: url), let meta = try? String(contentsOf: metaURL, encoding: .utf8) else {
            throw XCTSkip("ANEMLL tree fixture not downloaded")
        }
        let analysis = RepoAnalysis.analyze(files: try HFClient.parseTree(data), metaFiles: ["meta.yaml": meta])
        let pipeline = try XCTUnwrap(analysis.variants.first?.pipeline)
        XCTAssertEqual(pipeline.chunks, ["qwen_FFN_PF_lut6_chunk_01of01.mlmodelc"])
        XCTAssertEqual(pipeline.lmHead, "qwen_lm_head_lut6.mlmodelc")
        XCTAssertEqual(pipeline.contextLength, 512)
    }
}

final class ZipArchiveTests: XCTestCase {
    /// Python の zipfile（ZIP_DEFLATED）で作った小さな ZIP
    static let deflated = "UEsDBBQAAAAIAGhNV13Fw0oVIQAAADkBAAAYAAAATW9kZWwubWxtb2RlbGMvbW9kZWwubWlsKyjKTy9KzNUw1DPQ5EorzUtWyE3MzNPQVKiuHeXi4wIAUEsDBBQAAAAIAGhNV10mTAu3GAEAAAAEAAAdAAAATW9kZWwubWxtb2RlbGMvY29yZW1sZGF0YS5iaW5jYGRiZmFlY+fg5OLm4eXjFxAUEhYRFROXkJSSlpGVk1dQVFJWUVVT19DU0tbR1dM3MDQyNjE1M7ewtLK2sbWzd3B0cnZxdXP38PTy9vH18w8IDAoOCQ0Lj4iMio6JjYtPSExKTklNS8/IzMrOyc3LLygsKi4pLSuvqKyqrqmtq29obGpuaW1r7+js6u7p7eufMHHS5ClTp02fMXPW7Dlz581fsHDR4iVLly1fsXLV6jVr163fsHHT5i1bt23fsXPX7j179+0/cPDQ4SNHjx0/cfLU6TNnz52/cPHS5StXr12/cfPW7Tt3791/8PDR4ydPnz1/8fLV6zdv373/8PHT5y9fv33/8fPX7z9///1nGPX/qP9HsP8BUEsDBBQAAAAIAGhNV12GphA2BwAAAAUAAAAKAAAAcmVhZG1lLnR4dMtIzcnJBwBQSwECFAMUAAAACABoTVddxcNKFSEAAAA5AQAAGAAAAAAAAAAAAAAAgAEAAAAATW9kZWwubWxtb2RlbGMvbW9kZWwubWlsUEsBAhQDFAAAAAgAaE1XXSZMC7cYAQAAAAQAAB0AAAAAAAAAAAAAAIABVwAAAE1vZGVsLm1sbW9kZWxjL2NvcmVtbGRhdGEuYmluUEsBAhQDFAAAAAgAaE1XXYamEDYHAAAABQAAAAoAAAAAAAAAAAAAAIABqgEAAHJlYWRtZS50eHRQSwUGAAAAAAMAAwDJAAAA2QEAAAAA"

    func temporaryFolder() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("zip-\(UUID().uuidString)")
    }

    func testStoredRoundTrip() throws {
        var writer = ZipWriter()
        writer.add(path: "Project/", data: Data())
        writer.add(path: "Project/Package.swift", data: Data("// swift-tools-version: 5.9\n".utf8))
        writer.add(path: "Project/Sources/日本語.swift", data: Data("let x = \"ねこ\"\n".utf8))
        let archive = try ZipArchive(data: writer.finish())
        XCTAssertEqual(archive.entries.map(\.path), ["Project/", "Project/Package.swift", "Project/Sources/日本語.swift"])
        let folder = temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let written = try archive.extract(to: folder, stripping: "Project/", inflate: nil)
        XCTAssertEqual(written.sorted(), ["Package.swift", "Sources/日本語.swift"])
        XCTAssertEqual(try String(contentsOf: folder.appendingPathComponent("Sources/日本語.swift"), encoding: .utf8), "let x = \"ねこ\"\n")
        XCTAssertEqual(CRC32.checksum(Data("123456789".utf8)), 0xCBF4_3926)
    }

    func testDeflatedEntries() throws {
        let archive = try ZipArchive(data: try XCTUnwrap(Data(base64Encoded: Self.deflated)))
        XCTAssertEqual(archive.entries.count, 3)
        let readme = try XCTUnwrap(archive.entries.first { $0.path == "readme.txt" })
        guard let inflate = ZipArchive.systemInflater else {
            XCTAssertThrowsError(try archive.contents(of: readme, inflate: nil))
            throw XCTSkip("no Deflate support on this platform")
        }
        XCTAssertEqual(String(decoding: try archive.contents(of: readme, inflate: inflate), as: UTF8.self), "hello")
        let folder = temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        try archive.extract(to: folder, inflate: inflate)
        let mil = try String(contentsOf: folder.appendingPathComponent("Model.mlmodelc/model.mil"), encoding: .utf8)
        XCTAssertTrue(mil.hasPrefix("program(1.0)"))
        XCTAssertEqual(try Data(contentsOf: folder.appendingPathComponent("Model.mlmodelc/coremldata.bin")).count, 1024)
    }

    func testRejectsBrokenAndUnsafeArchives() throws {
        XCTAssertThrowsError(try ZipArchive(data: Data("not a zip".utf8)))
        var writer = ZipWriter()
        writer.add(path: "../escape.txt", data: Data("x".utf8))
        let archive = try ZipArchive(data: writer.finish())
        let folder = temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        XCTAssertThrowsError(try archive.extract(to: folder, inflate: nil))
        XCTAssertFalse(ZipArchive.isSafe("/etc/passwd"))
        XCTAssertTrue(ZipArchive.isSafe("a/b..c/d"))
    }
}
