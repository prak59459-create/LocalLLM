import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

/// ModelScope（Hugging Face 以外の置き場所）
final class ModelScopeTests: XCTestCase {
    func testNamesAndURLs() {
        let repo = "modelscope:Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        XCTAssertTrue(ModelScope.isModelScope(repo))
        XCTAssertFalse(ModelScope.isModelScope("Qwen/Qwen2.5-0.5B-Instruct-GGUF"))
        XCTAssertEqual(ModelScope.path(repo), "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        XCTAssertEqual(ModelScope.displayName(repo), "ModelScope · Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        XCTAssertEqual(ModelScope.resolveURL(repo: repo, path: "q4 k.gguf", revision: "main").absoluteString,
                       "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/q4%20k.gguf")
        let client = HFClient(token: "secret")
        XCTAssertEqual(client.resolveURL(repo: repo, path: "a.gguf", revision: "main").host, "modelscope.cn")
        XCTAssertEqual(client.resolveURL(repo: "Qwen/X", path: "a.gguf", revision: "main").host, "huggingface.co")
        // Hugging Face のトークンは ModelScope に送らない
        XCTAssertNil(client.headers(forRepo: repo)["Authorization"])
        XCTAssertEqual(client.headers(forRepo: "Qwen/X")["Authorization"], "Bearer secret")
    }

    func testSearchURLsAndPaging() throws {
        let client = HFClient()
        let only = client.searchURL(query: "qwen", source: .modelScope, textGenerationOnly: true, sort: .likes, limit: 50)
        XCTAssertTrue(ModelScope.isSearchURL(only))
        let body = try XCTUnwrap(try JSONSerialization.jsonObject(with: ModelScope.searchBody(for: only)) as? [String: Any])
        XCTAssertEqual(body["Name"] as? String, "qwen")
        XCTAssertEqual(body["SortBy"] as? String, "StarsCount")
        XCTAssertEqual(body["PageNumber"] as? Int, 1)
        XCTAssertNotNil(body["Criterion"])
        // 「すべて」と GGUF は、Hugging Face と ModelScope をまとめて探す
        for source in [HFSearchSource.all, .gguf] {
            let url = client.searchURL(query: "q", source: source, textGenerationOnly: false, sort: .downloads, limit: 50)
            XCTAssertEqual(url.scheme, ModelScope.combinedScheme)
            let parts = ModelScope.combinedParts(url)
            XCTAssertEqual(parts.huggingFace?.host, "huggingface.co")
            XCTAssertTrue(parts.modelScope.map(ModelScope.isSearchURL) ?? false)
        }
        XCTAssertEqual(client.searchURL(query: "q", source: .coreML, textGenerationOnly: false, sort: .downloads, limit: 50).host,
                       "huggingface.co")
        XCTAssertNil(ModelScope.combinedURL(huggingFace: nil, modelScope: nil))

        let page = Data("""
        {"Code":200,"Data":{"Model":{"TotalCount":120,"Models":[
          {"Name":"Qwen2.5-0.5B-Instruct-GGUF","Path":"Qwen","Downloads":143443,"Stars":12,"Libraries":["gguf"],"Tags":["gguf"],
           "LastUpdatedTime":1726842501,"Tasks":[{"Name":"text-generation"}]},
          {"Name":"broken"}]}}}
        """.utf8)
        let parsed = try ModelScope.parseSearch(page, requestURL: only)
        XCTAssertEqual(parsed.models.map(\.id), ["modelscope:Qwen/Qwen2.5-0.5B-Instruct-GGUF"])
        XCTAssertEqual(parsed.models.first?.downloads, 143443)
        XCTAssertEqual(parsed.models.first?.pipelineTag, "text-generation")
        let next = try XCTUnwrap(parsed.next)
        XCTAssertEqual(try (JSONSerialization.jsonObject(with: ModelScope.searchBody(for: next)) as? [String: Any])?["PageNumber"] as? Int, 2)
    }

    func testFilesAndInterleaving() throws {
        let files = try ModelScope.parseFiles(Data("""
        {"Code":200,"Data":{"Files":[{"Path":"a/b.gguf","Size":491000000,"Type":"blob","IsLFS":true,"Sha256":"abc"},
          {"Path":"a","Type":"tree"},{"Path":"README.md","Size":12,"Type":"blob","IsLFS":false,"Sha256":"x"}]}}
        """.utf8))
        XCTAssertEqual(files, [HFFileEntry(path: "a/b.gguf", size: 491_000_000, lfsSHA256: "abc"),
                               HFFileEntry(path: "README.md", size: 12, lfsSHA256: nil)])
        func summary(_ id: String) -> HFModelSummary {
            HFModelSummary(id: id, downloads: nil, likes: nil, pipelineTag: nil, libraryName: nil, tags: [], lastModified: nil, gated: false)
        }
        XCTAssertEqual(ModelScope.interleave([summary("a"), summary("b"), summary("c")], [summary("x"), summary("a")]).map(\.id),
                       ["a", "x", "b", "c"])
    }
}

/// 本物の ModelScope に接続するテスト（LOCALLLM_LIVE_TESTS=1 のときだけ）
final class ModelScopeLiveTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["LOCALLLM_LIVE_TESTS"] == "1" else {
            throw XCTSkip("set LOCALLLM_LIVE_TESTS=1 to run tests against the web")
        }
    }

    func testSearchListAndReadHeader() async throws {
        let client = HFClient()
        let page: HFSearchPage
        do {
            page = try await client.searchModels(query: "Qwen2.5-0.5B-Instruct-GGUF", source: .modelScope, textGenerationOnly: true,
                                                 sort: .downloads, limit: 10)
        } catch let error as URLError where error.code == .timedOut {
            throw XCTSkip("ModelScope did not answer this runner")
        } catch HFError.network {
            throw XCTSkip("ModelScope is not reachable from this runner")
        }
        let repo = try XCTUnwrap(page.models.first { $0.id == "modelscope:Qwen/Qwen2.5-0.5B-Instruct-GGUF" }?.id,
                                 "\(page.models.map(\.id))")
        let files = try await client.listFiles(repo: repo, revision: "main")
        let file = try XCTUnwrap(files.first { $0.path.lowercased().hasSuffix("q4_k_m.gguf") }, "\(files.map(\.path))")
        XCTAssertGreaterThan(file.size, 100_000_000)
        let header = try await client.fetchGGUFHeader(repo: repo, path: file.path)
        XCTAssertEqual(header.architecture, "qwen2")
        let info = try await client.modelInfo(repo: repo)
        XCTAssertEqual(info.id, repo)
        // まとめて探すと、両方から出てくる
        let all = try await client.searchModels(query: "Qwen2.5-0.5B-Instruct-GGUF", source: .all, textGenerationOnly: false,
                                                sort: .downloads, limit: 20)
        XCTAssertTrue(all.models.contains { ModelScope.isModelScope($0.id) })
        XCTAssertTrue(all.models.contains { !ModelScope.isModelScope($0.id) })
    }
}
