import CoreML
import Foundation
import Observation
import UIKit

/// ダウンロード中・一時停止中のモデルの状態。
struct DownloadState: Sendable {
    enum Phase: Sendable, Equatable {
        case downloading
        case paused
        case failed(String)
    }
    var phase: Phase
    var progress: DownloadProgress?
}

enum ModelLibraryError: Error, LocalizedError {
    case notEnoughStorage(needed: Int64, available: Int64)
    case incomplete
    case missingTokenizer
    case missingModelFolder

    var errorDescription: String? {
        switch self {
        case .notEnoughStorage(let needed, let available):
            return L("空き容量が足りません（必要: \(ByteFormat.string(needed)) / 空き: \(ByteFormat.string(available))）。",
                     "Not enough storage (needed: \(ByteFormat.string(needed)) / free: \(ByteFormat.string(available))).")
        case .incomplete:
            return L("ダウンロードが完了していません。", "The download has not finished.")
        case .missingTokenizer:
            return L("tokenizer.json が見つかりません。このリポジトリには対応するトークナイザが含まれていない可能性があります。",
                     "tokenizer.json was not found. This repository may not include a compatible tokenizer.")
        case .missingModelFolder:
            return L("モデルのファイルが見つかりません。削除してもう一度ダウンロードしてください。",
                     "The model files are missing. Delete the model and download it again.")
        }
    }
}

/// 端末に保存したモデルの一覧とダウンロードを管理する。
@MainActor
@Observable
final class ModelLibrary {
    private(set) var models: [InstalledModel] = []
    private(set) var downloads: [String: DownloadState] = [:]
    /// .mlpackage をコンパイル中のモデル ID
    private(set) var compilingModelIDs: Set<String> = []

    @ObservationIgnored private var tasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored private var backgroundTasks: [String: UIBackgroundTaskIdentifier] = [:]
    @ObservationIgnored private let fileManager = FileManager.default

    /// モデルを保存するフォルダ（Application Support/Models）
    let rootDirectory: URL

    init() {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        rootDirectory = base.appendingPathComponent("Models", isDirectory: true)
        try? fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        // 大きなファイルなので iCloud バックアップから除外する
        var root = rootDirectory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? root.setResourceValues(values)
        reload()
    }

    // MARK: - 一覧

    var completedModels: [InstalledModel] {
        models.filter(\.isComplete)
    }

    func model(withID id: String) -> InstalledModel? {
        models.first { $0.id == id }
    }

    /// manifest.json を読み直す。
    func reload() {
        let folders = (try? fileManager.contentsOfDirectory(at: rootDirectory, includingPropertiesForKeys: nil)) ?? []
        var loaded: [InstalledModel] = []
        let decoder = JSONDecoder()
        for folder in folders {
            let manifestURL = folder.appendingPathComponent("manifest.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let model = try? decoder.decode(InstalledModel.self, from: data) else { continue }
            loaded.append(model)
            if !model.isComplete, downloads[model.id] == nil {
                downloads[model.id] = DownloadState(phase: .paused, progress: nil)
            }
        }
        models = loaded.sorted { $0.createdAt > $1.createdAt }
    }

    func directory(for model: InstalledModel) -> URL {
        rootDirectory.appendingPathComponent(model.directoryName, isDirectory: true)
    }

    /// 保存済みモデルの合計サイズ（概算）
    var totalBytesOnDisk: Int64 {
        models.reduce(0) { $0 + $1.totalBytes }
    }

    /// 端末の空き容量
    var availableStorageBytes: Int64? {
        let values = try? rootDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    // MARK: - ダウンロード

    /// ダウンロードを始める（既に記録があれば続きから）。
    func startDownload(repoID: String, revision: String = "main", variant: ModelVariant, analysis: RepoAnalysis,
                       metadata: CoreMLMetadata?, client: HFClient) throws {
        let id = InstalledModel.makeID(repoID: repoID, variantPath: variant.path)
        if let existing = model(withID: id) {
            if !existing.isComplete { resumeDownload(id: id, client: client) }
            return
        }
        let files = analysis.filesToDownload(for: variant)
        let total = files.reduce(Int64(0)) { $0 + $1.size }
        if let available = availableStorageBytes, available < total + 200_000_000 {
            throw ModelLibraryError.notEnoughStorage(needed: total, available: available)
        }
        let weightBytes = variant.files.filter { $0.path.hasSuffix("weight.bin") }.reduce(Int64(0)) { $0 + $1.size }
        let model = InstalledModel(
            id: id,
            repoID: repoID,
            revision: revision,
            variantPath: variant.path,
            kind: variant.kind,
            displayName: variant.displayName,
            directoryName: InstalledModel.makeDirectoryName(repoID: repoID, variantPath: variant.path),
            files: files,
            totalBytes: total,
            createdAt: Date(),
            isComplete: false,
            compiledModelPath: nil,
            maxContext: metadata?.interface.maxContext,
            stateBytes: metadata?.stateBytes,
            weightBytes: weightBytes > 0 ? weightBytes : nil
        )
        try fileManager.createDirectory(at: directory(for: model), withIntermediateDirectories: true)
        try save(model)
        models.insert(model, at: 0)
        resumeDownload(id: id, client: client)
    }

    /// 一時停止中・失敗したダウンロードを続きから再開する。
    func resumeDownload(id: String, client: HFClient) {
        guard let model = model(withID: id), !model.isComplete, tasks[id] == nil else { return }
        let items = model.files.map { file in
            ModelDownloadJob.Item(url: client.resolveURL(repo: model.repoID, path: file.path, revision: model.revision),
                                  relativePath: file.path,
                                  expectedSize: file.size > 0 ? file.size : nil)
        }
        let job = ModelDownloadJob(items: items, destinationDirectory: directory(for: model),
                                   headers: client.authorizationHeaders)
        downloads[id] = DownloadState(phase: .downloading, progress: downloads[id]?.progress)
        beginBackgroundTask(for: id)
        updateIdleTimer()

        tasks[id] = Task { [weak self] in
            do {
                try await job.run { progress in
                    Task { @MainActor [weak self] in
                        guard let self, self.tasks[id] != nil else { return }
                        self.downloads[id] = DownloadState(phase: .downloading, progress: progress)
                    }
                }
                self?.finishDownload(id: id)
            } catch {
                self?.failDownload(id: id, error: error)
            }
        }
    }

    /// ダウンロードを一時停止する（途中までのファイルは残る）。
    func pauseDownload(id: String) {
        tasks[id]?.cancel()
        tasks[id] = nil
        downloads[id] = DownloadState(phase: .paused, progress: downloads[id]?.progress)
        endBackgroundTask(for: id)
        updateIdleTimer()
    }

    /// モデルを削除する（ダウンロード中なら止めてから）。
    func delete(id: String) {
        tasks[id]?.cancel()
        tasks[id] = nil
        downloads[id] = nil
        endBackgroundTask(for: id)
        updateIdleTimer()
        if let model = model(withID: id) {
            try? fileManager.removeItem(at: directory(for: model))
        }
        models.removeAll { $0.id == id }
    }

    var isDownloading: Bool { !tasks.isEmpty }

    private func finishDownload(id: String) {
        tasks[id] = nil
        endBackgroundTask(for: id)
        updateIdleTimer()
        guard var model = model(withID: id) else { return }
        model.isComplete = true
        if model.kind == .compiled {
            model.compiledModelPath = model.variantPath
            // ダウンロードした metadata.json から最大長などを読み直す
            let metadataURL = directory(for: model).appendingPathComponent(model.variantPath)
                .appendingPathComponent("metadata.json")
            if let data = try? Data(contentsOf: metadataURL), let metadata = try? CoreMLMetadata.parse(data) {
                model.maxContext = metadata.interface.maxContext ?? model.maxContext
                model.stateBytes = metadata.stateBytes
            }
        }
        try? save(model)
        replace(model)
        downloads[id] = nil
    }

    private func failDownload(id: String, error: Error) {
        tasks[id] = nil
        endBackgroundTask(for: id)
        updateIdleTimer()
        if error is CancellationError {
            downloads[id] = DownloadState(phase: .paused, progress: downloads[id]?.progress)
        } else {
            downloads[id] = DownloadState(phase: .failed(error.localizedDescription), progress: downloads[id]?.progress)
        }
    }

    // MARK: - 読み込みの準備

    /// 推論エンジンに渡すファイル一式を用意する。.mlpackage なら初回だけコンパイルする（数十秒〜数分）。
    func prepareFiles(for id: String) async throws -> ModelFiles {
        guard var model = model(withID: id) else { throw ModelLibraryError.missingModelFolder }
        guard model.isComplete else { throw ModelLibraryError.incomplete }
        let folder = directory(for: model)

        if model.compiledModelPath == nil || !fileManager.fileExists(atPath: folder.appendingPathComponent(model.compiledModelPath ?? "").path) {
            if model.kind == .compiled {
                model.compiledModelPath = model.variantPath
            } else {
                let packageURL = folder.appendingPathComponent(model.variantPath)
                guard fileManager.fileExists(atPath: packageURL.path) else { throw ModelLibraryError.missingModelFolder }
                compilingModelIDs.insert(id)
                defer { compilingModelIDs.remove(id) }
                let compiledTemp = try await MLModel.compileModel(at: packageURL)
                let compiledPath = (model.variantPath as NSString).deletingPathExtension + ".mlmodelc"
                let destination = folder.appendingPathComponent(compiledPath)
                try? fileManager.removeItem(at: destination)
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                do {
                    try fileManager.moveItem(at: compiledTemp, to: destination)
                } catch {
                    try fileManager.copyItem(at: compiledTemp, to: destination)
                    try? fileManager.removeItem(at: compiledTemp)
                }
                // 容量を節約するため、コンパイル前のパッケージは消す
                try? fileManager.removeItem(at: packageURL)
                model.compiledModelPath = compiledPath
            }
            try save(model)
            replace(model)
        }

        guard let compiledPath = model.compiledModelPath else { throw ModelLibraryError.missingModelFolder }
        guard let tokenizerPath = model.supportFilePath(named: "tokenizer.json") else { throw ModelLibraryError.missingTokenizer }

        func url(_ name: String) -> URL? {
            model.supportFilePath(named: name).map { folder.appendingPathComponent($0) }
                .flatMap { fileManager.fileExists(atPath: $0.path) ? $0 : nil }
        }

        // chat_template.jinja / chat_template.json があれば tokenizer_config.json の chat_template より優先する
        var templateOverride: String?
        if let jinjaURL = url("chat_template.jinja"), let text = try? String(contentsOf: jinjaURL, encoding: .utf8) {
            templateOverride = text
        } else if let jsonURL = url("chat_template.json"),
                  let data = try? Data(contentsOf: jsonURL),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let text = object["chat_template"] as? String {
            templateOverride = text
        }

        return ModelFiles(
            modelID: model.id,
            displayName: model.displayName,
            compiledModel: folder.appendingPathComponent(compiledPath),
            tokenizerJSON: folder.appendingPathComponent(tokenizerPath),
            tokenizerConfigJSON: url("tokenizer_config.json"),
            generationConfigJSON: url("generation_config.json"),
            configJSON: url("config.json"),
            chatTemplateOverride: templateOverride
        )
    }

    // MARK: - 補助

    private func save(_ model: InstalledModel) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(model)
        let folder = directory(for: model)
        try fileManager.createDirectory(at: folder, withIntermediateDirectories: true)
        try data.write(to: folder.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private func replace(_ model: InstalledModel) {
        if let index = models.firstIndex(where: { $0.id == model.id }) {
            models[index] = model
        } else {
            models.insert(model, at: 0)
        }
    }

    /// アプリがバックグラウンドに回っても、少しの間ダウンロードを続けられるようにする。
    private func beginBackgroundTask(for id: String) {
        guard backgroundTasks[id] == nil else { return }
        let identifier = UIApplication.shared.beginBackgroundTask(withName: "download-\(id)") { [weak self] in
            Task { @MainActor [weak self] in
                // 時間切れ: 一時停止して、次に開いたとき再開できるようにする
                self?.pauseDownload(id: id)
            }
        }
        if identifier != .invalid { backgroundTasks[id] = identifier }
    }

    private func endBackgroundTask(for id: String) {
        guard let identifier = backgroundTasks.removeValue(forKey: id) else { return }
        UIApplication.shared.endBackgroundTask(identifier)
    }

    /// ダウンロード中は画面が自動で消えないようにする。
    private func updateIdleTimer() {
        UIApplication.shared.isIdleTimerDisabled = !tasks.isEmpty
    }
}
