import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

/// テスト用のサーバー（URLProtocol）。Range に対応し、途中で切断したり、Range を無視したりできる。
final class MockFileProtocol: URLProtocol {
    struct Behavior {
        var body = Data()
        var status: Int? = nil
        var honorRange = true
        /// このバイト数を送ったところで接続を切る
        var failAfter: Int? = nil
        /// 送る前に待つ秒数
        var delay: TimeInterval = 0
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _behavior = Behavior()
    nonisolated(unsafe) private static var _requests: [URLRequest] = []

    static var behavior: Behavior {
        get { lock.lock(); defer { lock.unlock() }; return _behavior }
        set { lock.lock(); _behavior = newValue; lock.unlock() }
    }

    static var requests: [URLRequest] {
        lock.lock(); defer { lock.unlock() }
        return _requests
    }

    static func reset(_ behavior: Behavior) {
        lock.lock()
        _behavior = behavior
        _requests = []
        lock.unlock()
    }

    private var stopped = false

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "mock.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let behavior = Self.behavior
        Self.lock.lock()
        Self._requests.append(request)
        Self.lock.unlock()
        if behavior.delay > 0 {
            let box = UncheckedBox(value: self)
            DispatchQueue.global().asyncAfter(deadline: .now() + behavior.delay) { box.value.respond(behavior) }
        } else {
            respond(behavior)
        }
    }

    private func respond(_ behavior: Behavior) {
        if stopped { return }
        var start = 0
        if behavior.honorRange, let range = request.value(forHTTPHeaderField: "Range"),
           let value = Int(range.replacingOccurrences(of: "bytes=", with: "").replacingOccurrences(of: "-", with: "")) {
            start = value
        }
        let status: Int
        if let forced = behavior.status {
            status = forced
        } else if start >= behavior.body.count && start > 0 {
            status = 416
        } else {
            status = start > 0 ? 206 : 200
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: [:])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        guard status == 200 || status == 206 else {
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        var chunk = behavior.body.count > start ? behavior.body[start...] : Data()[...]
        if let failAfter = behavior.failAfter, failAfter < chunk.count {
            chunk = chunk.prefix(failAfter)
            client?.urlProtocol(self, didLoad: Data(chunk))
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            return
        }
        if !chunk.isEmpty { client?.urlProtocol(self, didLoad: Data(chunk)) }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {
        stopped = true
    }
}

final class ResumableDownloaderTests: XCTestCase {
    var folder: URL!
    let url = URL(string: "https://mock.test/repo/resolve/main/model.bin")!
    let body = Data((0..<50_000).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ 7) })

    override func setUp() {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("dl-\(UUID().uuidString)")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: folder)
    }

    func makeDownloader() -> ResumableDownloader {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockFileProtocol.self]
        return ResumableDownloader(configuration: configuration)
    }

    var destination: URL { folder.appendingPathComponent("sub/model.bin") }
    var part: URL { destination.appendingPathExtension("part") }

    func testFreshDownload() async throws {
        MockFileProtocol.reset(.init(body: body))
        let reported = ProgressRecorder()
        try await makeDownloader().download(from: url, to: destination, expectedSize: Int64(body.count),
                                            headers: ["User-Agent": "test"]) { reported.add($0) }
        XCTAssertEqual(try Data(contentsOf: destination), body)
        XCTAssertFalse(FileManager.default.fileExists(atPath: part.path))
        XCTAssertEqual(reported.values.last, Int64(body.count))
        XCTAssertEqual(MockFileProtocol.requests.first?.value(forHTTPHeaderField: "User-Agent"), "test")
        XCTAssertNil(MockFileProtocol.requests.first?.value(forHTTPHeaderField: "Range"))
    }

    func testResumeAfterConnectionLoss() async throws {
        MockFileProtocol.reset(.init(body: body, failAfter: 20_000))
        let downloader = makeDownloader()
        do {
            try await downloader.download(from: url, to: destination, expectedSize: Int64(body.count), headers: [:]) { _ in }
            XCTFail("expected failure")
        } catch {
            XCTAssertTrue(ModelDownloadJob.isRetryable(error), "\(error)")
        }
        XCTAssertEqual(try Data(contentsOf: part), body.prefix(20_000))

        MockFileProtocol.reset(.init(body: body))
        try await downloader.download(from: url, to: destination, expectedSize: Int64(body.count), headers: [:]) { _ in }
        XCTAssertEqual(MockFileProtocol.requests.last?.value(forHTTPHeaderField: "Range"), "bytes=20000-")
        XCTAssertEqual(try Data(contentsOf: destination), body)
        XCTAssertFalse(FileManager.default.fileExists(atPath: part.path))
    }

    func testServerIgnoringRangeRestartsFromZero() async throws {
        try FileManager.default.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 0xFF, count: 1234).write(to: part)
        MockFileProtocol.reset(.init(body: body, honorRange: false))
        try await makeDownloader().download(from: url, to: destination, expectedSize: Int64(body.count), headers: [:]) { _ in }
        XCTAssertEqual(try Data(contentsOf: destination), body)
    }

    func testCompletePartFileNeedsNoRequest() async throws {
        try FileManager.default.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try body.write(to: part)
        MockFileProtocol.reset(.init(body: body))
        try await makeDownloader().download(from: url, to: destination, expectedSize: Int64(body.count), headers: [:]) { _ in }
        XCTAssertEqual(try Data(contentsOf: destination), body)
        XCTAssertTrue(MockFileProtocol.requests.isEmpty)
    }

    func testRangeNotSatisfiableWithUnknownSizeRemovesPart() async throws {
        try FileManager.default.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 1, count: 60_000).write(to: part)
        MockFileProtocol.reset(.init(body: body))
        do {
            try await makeDownloader().download(from: url, to: destination, expectedSize: nil, headers: [:]) { _ in }
            XCTFail("expected failure")
        } catch {
            XCTAssertFalse(FileManager.default.fileExists(atPath: part.path))
        }
        // もう一度なら最初から取れる
        try await makeDownloader().download(from: url, to: destination, expectedSize: nil, headers: [:]) { _ in }
        XCTAssertEqual(try Data(contentsOf: destination), body)
    }

    func testOversizedPartIsDiscarded() async throws {
        try FileManager.default.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 1, count: 60_000).write(to: part)
        MockFileProtocol.reset(.init(body: body))
        try await makeDownloader().download(from: url, to: destination, expectedSize: Int64(body.count), headers: [:]) { _ in }
        XCTAssertEqual(try Data(contentsOf: destination), body)
        XCTAssertNil(MockFileProtocol.requests.first?.value(forHTTPHeaderField: "Range"))
    }

    func testHTTPErrorAndSizeMismatch() async throws {
        MockFileProtocol.reset(.init(body: body, status: 404))
        do {
            try await makeDownloader().download(from: url, to: destination, expectedSize: 10, headers: [:]) { _ in }
            XCTFail("expected 404")
        } catch HFError.http(let status, _) {
            XCTAssertEqual(status, 404)
        }
        MockFileProtocol.reset(.init(body: body))
        do {
            try await makeDownloader().download(from: url, to: destination, expectedSize: Int64(body.count + 5), headers: [:]) { _ in }
            XCTFail("expected size mismatch")
        } catch HFError.invalidResponse {
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        }
    }

    func testCancellationKeepsPartialFile() async throws {
        MockFileProtocol.reset(.init(body: body, delay: 2))
        let downloader = makeDownloader()
        let destination = self.destination
        let url = self.url
        let task = Task {
            try await downloader.download(from: url, to: destination, expectedSize: 50_000, headers: [:]) { _ in }
        }
        try await Task.sleep(nanoseconds: 200_000_000)
        task.cancel()
        do {
            try await task.value
            XCTFail("expected cancellation")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testJobDownloadsAllFilesAndRetries() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockFileProtocol.self]
        MockFileProtocol.reset(.init(body: body, failAfter: 10_000))
        let items = [
            ModelDownloadJob.Item(url: url, relativePath: "m.mlmodelc/weights/weight.bin", expectedSize: Int64(body.count)),
        ]
        let job = ModelDownloadJob(items: items, destinationDirectory: folder, headers: [:], maxRetries: 3,
                                   downloader: ResumableDownloader(configuration: configuration))
        // 1 回目は途中で切れる。再試行の前にサーバーを直す
        let fixer = Task {
            try await Task.sleep(nanoseconds: 300_000_000)
            MockFileProtocol.behavior = .init(body: body)
        }
        let progress = ProgressRecorder()
        try await job.run { progress.add($0.completedBytes) }
        _ = try? await fixer.value
        XCTAssertEqual(try Data(contentsOf: folder.appendingPathComponent("m.mlmodelc/weights/weight.bin")), body)
        XCTAssertEqual(progress.values.last, Int64(body.count))
        XCTAssertGreaterThanOrEqual(MockFileProtocol.requests.count, 2)
        // もう一度実行しても、そろっているファイルはダウンロードしない
        let before = MockFileProtocol.requests.count
        try await job.run { _ in }
        XCTAssertEqual(MockFileProtocol.requests.count, before)
    }
}

struct UncheckedBox<Value>: @unchecked Sendable {
    let value: Value
}

final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Int64] = []

    func add(_ value: Int64) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }

    var values: [Int64] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
