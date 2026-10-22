import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// 1 ファイルの再開可能ダウンロード。
/// destination + ".part" に書き込み、完了したら destination へ移動する。
/// 途中で止まっても .part が残るので、次回は Range ヘッダで続きから取得する。
final class ResumableDownloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private final class Transfer {
        let destination: URL
        let partURL: URL
        let expectedSize: Int64?
        let headers: [String: String]
        let progress: @Sendable (Int64) -> Void
        let continuation: CheckedContinuation<Void, Error>
        var handle: FileHandle?
        var bytesOnDisk: Int64
        var lastReport = Date.distantPast
        var failure: Error?

        init(destination: URL, partURL: URL, expectedSize: Int64?, headers: [String: String], bytesOnDisk: Int64,
             progress: @escaping @Sendable (Int64) -> Void, continuation: CheckedContinuation<Void, Error>) {
            self.destination = destination
            self.partURL = partURL
            self.expectedSize = expectedSize
            self.headers = headers
            self.bytesOnDisk = bytesOnDisk
            self.progress = progress
            self.continuation = continuation
        }
    }

    private let lock = NSLock()
    private var transfers: [Int: Transfer] = [:]
    private var session: URLSession!

    init(configuration: URLSessionConfiguration = .default) {
        super.init()
        let configuration = configuration
        configuration.timeoutIntervalForRequest = 60
        #if !canImport(FoundationNetworking)
        configuration.waitsForConnectivity = true
        #endif
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
    }

    deinit {
        session?.invalidateAndCancel()
    }

    /// ダウンロードする。Task をキャンセルすると中断し、.part は残す。
    func download(from url: URL, to destination: URL, expectedSize: Int64?, headers: [String: String],
                  progress: @escaping @Sendable (_ bytesOnDisk: Int64) -> Void) async throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let partURL = destination.appendingPathExtension("part")
        var existing = (try? fileManager.attributesOfItem(atPath: partURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        if let expectedSize, existing > expectedSize {
            try? fileManager.removeItem(at: partURL)
            existing = 0
        }
        if let expectedSize, existing == expectedSize, expectedSize > 0 {
            // 前回、最後まで取れていた
            try Self.finish(partURL: partURL, destination: destination)
            progress(existing)
            return
        }

        var request = URLRequest(url: url)
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        if existing > 0 { request.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range") }
        let startBytes = existing

        let handle = TaskHandle()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let task = session.dataTask(with: request)
                let transfer = Transfer(destination: destination, partURL: partURL, expectedSize: expectedSize, headers: headers,
                                        bytesOnDisk: startBytes, progress: progress, continuation: continuation)
                lock.lock()
                transfers[task.taskIdentifier] = transfer
                handle.task = task
                let cancelled = handle.cancelled
                lock.unlock()
                if cancelled || Task.isCancelled {
                    task.cancel()
                } else {
                    task.resume()
                }
            }
        } onCancel: {
            lock.lock()
            handle.cancelled = true
            let task = handle.task
            lock.unlock()
            task?.cancel()
        }
    }

    /// キャンセル処理と開始処理の間で URLSessionDataTask を受け渡す（lock で守る）
    private final class TaskHandle: @unchecked Sendable {
        var task: URLSessionDataTask?
        var cancelled = false
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let transfer = transfer(for: dataTask) else {
            completionHandler(.cancel)
            return
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        do {
            switch status {
            case 206:
                // 続きから
                let handle = try FileHandle(forWritingTo: transfer.partURL)
                try handle.seekToEnd()
                transfer.handle = handle
            case 200:
                // 最初から（Range が無視された）
                _ = FileManager.default.createFile(atPath: transfer.partURL.path, contents: nil)
                transfer.handle = try FileHandle(forWritingTo: transfer.partURL)
                try transfer.handle?.truncate(atOffset: 0)
                transfer.bytesOnDisk = 0
            case 416:
                // 範囲外: 既に全部ある可能性
                transfer.failure = RangeNotSatisfiable()
                completionHandler(.cancel)
                return
            default:
                transfer.failure = HFError.http(status: status, url: response.url ?? dataTask.originalRequest?.url ?? URL(fileURLWithPath: "/"))
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        } catch {
            transfer.failure = error
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let transfer = transfer(for: dataTask), let handle = transfer.handle else { return }
        do {
            try handle.write(contentsOf: data)
            transfer.bytesOnDisk += Int64(data.count)
            let now = Date()
            if now.timeIntervalSince(transfer.lastReport) > 0.1 {
                transfer.lastReport = now
                transfer.progress(transfer.bytesOnDisk)
            }
        } catch {
            transfer.failure = error
            dataTask.cancel()
        }
    }

    /// リダイレクト先（CDN）にも Range などのヘッダを引き継ぐ
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        var redirected = request
        if let original = task.originalRequest {
            if let range = original.value(forHTTPHeaderField: "Range") { redirected.setValue(range, forHTTPHeaderField: "Range") }
            if let agent = original.value(forHTTPHeaderField: "User-Agent") { redirected.setValue(agent, forHTTPHeaderField: "User-Agent") }
            // 認証ヘッダは同じホストへのリダイレクトのときだけ付ける（CDN の署名付き URL には不要）
            if request.url?.host == original.url?.host, let auth = original.value(forHTTPHeaderField: "Authorization") {
                redirected.setValue(auth, forHTTPHeaderField: "Authorization")
            }
        }
        completionHandler(redirected)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let transfer = transfers.removeValue(forKey: task.taskIdentifier)
        lock.unlock()
        guard let transfer else { return }
        try? transfer.handle?.close()
        transfer.handle = nil

        if transfer.failure is RangeNotSatisfiable {
            let size = (try? FileManager.default.attributesOfItem(atPath: transfer.partURL.path)[.size] as? NSNumber)?.int64Value ?? 0
            if let expected = transfer.expectedSize, size == expected {
                do {
                    try Self.finish(partURL: transfer.partURL, destination: transfer.destination)
                    transfer.continuation.resume()
                } catch {
                    transfer.continuation.resume(throwing: error)
                }
            } else {
                try? FileManager.default.removeItem(at: transfer.partURL)
                transfer.continuation.resume(throwing: HFError.invalidResponse(L("途中のファイルが壊れていたので削除しました。もう一度試してください。",
                                                                                "The partial file was invalid and has been removed. Please try again.")))
            }
            return
        }
        if let failure = transfer.failure {
            transfer.continuation.resume(throwing: failure)
            return
        }
        if let error {
            if (error as? URLError)?.code == .cancelled {
                transfer.continuation.resume(throwing: CancellationError())
            } else {
                transfer.continuation.resume(throwing: HFError.network(error.localizedDescription))
            }
            return
        }
        transfer.progress(transfer.bytesOnDisk)
        if let expected = transfer.expectedSize, transfer.bytesOnDisk != expected {
            if transfer.bytesOnDisk > expected { try? FileManager.default.removeItem(at: transfer.partURL) }
            transfer.continuation.resume(throwing: HFError.invalidResponse(
                L("サイズが一致しません（\(transfer.bytesOnDisk) / \(expected) バイト）", "Size mismatch (\(transfer.bytesOnDisk) / \(expected) bytes)")))
            return
        }
        do {
            try Self.finish(partURL: transfer.partURL, destination: transfer.destination)
            transfer.continuation.resume()
        } catch {
            transfer.continuation.resume(throwing: error)
        }
    }

    private func transfer(for task: URLSessionTask) -> Transfer? {
        lock.lock()
        defer { lock.unlock() }
        return transfers[task.taskIdentifier]
    }

    private static func finish(partURL: URL, destination: URL) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destination.path) { try fileManager.removeItem(at: destination) }
        try fileManager.moveItem(at: partURL, to: destination)
    }

    private struct RangeNotSatisfiable: Error {}
}

/// ダウンロード全体の進み具合
struct DownloadProgress: Hashable, Sendable {
    var completedBytes: Int64
    var totalBytes: Int64
    var currentFile: String?
    var bytesPerSecond: Double

    var fractionCompleted: Double {
        totalBytes > 0 ? min(1, Double(completedBytes) / Double(totalBytes)) : 0
    }

    var estimatedSecondsRemaining: Double? {
        guard bytesPerSecond > 1, totalBytes > completedBytes else { return nil }
        return Double(totalBytes - completedBytes) / bytesPerSecond
    }
}

/// 複数のファイルを順番にダウンロードする。失敗したら少し待って再試行する。
final class ModelDownloadJob: @unchecked Sendable {
    struct Item: Hashable, Sendable {
        var url: URL
        var relativePath: String
        var expectedSize: Int64?
    }

    let items: [Item]
    let destinationDirectory: URL
    let headers: [String: String]
    let maxRetries: Int
    private let downloader: ResumableDownloader

    init(items: [Item], destinationDirectory: URL, headers: [String: String], maxRetries: Int = 5,
         downloader: ResumableDownloader = ResumableDownloader()) {
        self.items = items
        self.destinationDirectory = destinationDirectory
        self.headers = headers
        self.maxRetries = maxRetries
        self.downloader = downloader
    }

    func run(progress: @escaping @Sendable (DownloadProgress) -> Void) async throws {
        let total = items.reduce(Int64(0)) { $0 + ($1.expectedSize ?? 0) }
        let meter = SpeedMeter()
        var finishedBytes: Int64 = 0
        for item in items {
            try Task.checkCancellation()
            let destination = destinationDirectory.appendingPathComponent(item.relativePath)
            // 既にそろっているファイルは飛ばす
            if let size = Self.fileSize(destination), item.expectedSize == nil || size == item.expectedSize {
                finishedBytes += size
                progress(DownloadProgress(completedBytes: finishedBytes, totalBytes: total, currentFile: item.relativePath,
                                          bytesPerSecond: meter.speed))
                continue
            }
            var attempt = 0
            let base = finishedBytes
            while true {
                do {
                    try await downloader.download(from: item.url, to: destination, expectedSize: item.expectedSize,
                                                  headers: headers) { bytes in
                        let completed = base + bytes
                        meter.record(completed)
                        progress(DownloadProgress(completedBytes: completed, totalBytes: total, currentFile: item.relativePath,
                                                  bytesPerSecond: meter.speed))
                    }
                    break
                } catch {
                    if error is CancellationError || Task.isCancelled { throw CancellationError() }
                    attempt += 1
                    guard attempt <= maxRetries, Self.isRetryable(error) else { throw error }
                    let delay = min(16.0, pow(2.0, Double(attempt - 1)))
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
            finishedBytes += Self.fileSize(destination) ?? (item.expectedSize ?? 0)
        }
        progress(DownloadProgress(completedBytes: finishedBytes, totalBytes: total, currentFile: nil, bytesPerSecond: meter.speed))
    }

    static func isRetryable(_ error: Error) -> Bool {
        if case HFError.http(let status, _) = error {
            return status == 408 || status == 429 || status >= 500
        }
        return true
    }

    static func fileSize(_ url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else { return nil }
        return (attributes[.size] as? NSNumber)?.int64Value
    }
}

/// 直近の速度（なめらかにした値）
private final class SpeedMeter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastBytes: Int64?
    private var lastTime = Date()
    private var smoothed: Double = 0

    var speed: Double {
        lock.lock()
        defer { lock.unlock() }
        return smoothed
    }

    func record(_ bytes: Int64) {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        guard let previous = lastBytes else {
            lastBytes = bytes
            lastTime = now
            return
        }
        let elapsed = now.timeIntervalSince(lastTime)
        guard elapsed >= 0.25 else { return }
        let instant = Double(bytes - previous) / elapsed
        smoothed = smoothed == 0 ? instant : smoothed * 0.7 + instant * 0.3
        lastBytes = bytes
        lastTime = now
    }
}
