import Foundation
import Observation
import UIKit

/// 画像生成モデルの候補（Hugging Face のリポジトリ）。
struct ImageModelSpec: Identifiable, Hashable, Codable, Sendable {
    var id: String { repoID }
    var repoID: String
    var title: String
    var approximateBytes: Int64
    var note: String

    static var catalog: [ImageModelSpec] {[
        ImageModelSpec(repoID: "apple/coreml-stable-diffusion-v1-5-palettized", title: "Stable Diffusion 1.5 (6bit)",
                       approximateBytes: 890_000_000,
                       note: L("定番のモデル。512×512。Neural Engine 向けに小さくしてあります。",
                               "The classic model. 512×512, compressed for the Neural Engine.")),
        ImageModelSpec(repoID: "apple/coreml-stable-diffusion-2-1-base-palettized", title: "Stable Diffusion 2.1 base (6bit)",
                       approximateBytes: 1_070_000_000,
                       note: L("1.5 より新しい版。512×512。", "A newer version than 1.5. 512×512.")),
    ]}
}

/// 端末に保存した画像生成モデルの記録（モデルのフォルダの manifest.json）
struct InstalledImageModel: Codable, Identifiable, Hashable, Sendable {
    var id: String { repoID }
    var repoID: String
    var title: String
    var directoryName: String
    var files: [HFFileEntry]
    /// リポジトリ内の、モデルの入っているフォルダ
    var folder: String
    var totalBytes: Int64
    var isComplete: Bool
}

/// 画像生成モデルのダウンロードと一覧。
@MainActor
@Observable
final class ImageModelStore {
    private(set) var models: [InstalledImageModel] = []
    private(set) var downloads: [String: DownloadState] = [:]

    @ObservationIgnored private var tasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored private let fileManager = FileManager.default
    let rootDirectory: URL

    init() {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        rootDirectory = base.appendingPathComponent("ImageModels", isDirectory: true)
        try? fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        var root = rootDirectory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? root.setResourceValues(values)
        reload()
    }

    var completedModels: [InstalledImageModel] { models.filter(\.isComplete) }

    func model(repoID: String) -> InstalledImageModel? {
        models.first { $0.repoID == repoID }
    }

    func reload() {
        let folders = (try? fileManager.contentsOfDirectory(at: rootDirectory, includingPropertiesForKeys: nil)) ?? []
        var loaded: [InstalledImageModel] = []
        for folder in folders {
            guard let data = try? Data(contentsOf: folder.appendingPathComponent("manifest.json")),
                  let model = try? JSONDecoder().decode(InstalledImageModel.self, from: data) else { continue }
            loaded.append(model)
            if !model.isComplete, downloads[model.id] == nil {
                downloads[model.id] = DownloadState(phase: .paused, progress: nil)
            }
        }
        models = loaded.sorted { $0.title < $1.title }
    }

    func directory(for model: InstalledImageModel) -> URL {
        rootDirectory.appendingPathComponent(model.directoryName, isDirectory: true)
    }

    /// 生成に使うファイル（そろっていなければ nil）
    func files(for model: InstalledImageModel) -> ImageModelFiles? {
        guard model.isComplete else { return nil }
        let folder = directory(for: model)
        let files = ImageModelFiles(textEncoder: folder.appendingPathComponent("TextEncoder.mlmodelc"),
                                    unet: folder.appendingPathComponent("Unet.mlmodelc"),
                                    vaeDecoder: folder.appendingPathComponent("VAEDecoder.mlmodelc"),
                                    vocabulary: folder.appendingPathComponent("vocab.json"),
                                    merges: folder.appendingPathComponent("merges.txt"))
        let required = [files.textEncoder, files.unet, files.vaeDecoder, files.vocabulary, files.merges]
        return required.allSatisfy { fileManager.fileExists(atPath: $0.path) } ? files : nil
    }

    // MARK: - ダウンロード

    /// リポジトリを調べて、ダウンロードを始める（途中までのものがあれば続きから）
    func startDownload(repoID: String, title: String, client: HFClient) {
        guard tasks[repoID] == nil else { return }
        downloads[repoID] = DownloadState(phase: .downloading, progress: downloads[repoID]?.progress)
        tasks[repoID] = Task { [self] in
            do {
                var model: InstalledImageModel
                if let existing = self.model(repoID: repoID) {
                    model = existing
                } else {
                    let entries = try await client.listFiles(repo: repoID, revision: "main")
                    guard let layout = ImageRepoLayout.find(in: entries) else {
                        throw ImageModelError.unsupportedRepository
                    }
                    if let available = availableStorageBytes, available < layout.totalBytes + 200_000_000 {
                        throw ModelLibraryError.notEnoughStorage(needed: layout.totalBytes, available: available)
                    }
                    let name = InstalledModel.makeDirectoryName(repoID: repoID, variantPath: "image")
                    model = InstalledImageModel(repoID: repoID, title: title, directoryName: name, files: layout.files,
                                                folder: layout.folder, totalBytes: layout.totalBytes, isComplete: false)
                    try save(model)
                    models.append(model)
                }
                let layout = ImageRepoLayout(folder: model.folder, files: model.files)
                let items = model.files.map { file in
                    ModelDownloadJob.Item(url: client.resolveURL(repo: repoID, path: file.path, revision: "main"),
                                          relativePath: layout.localPath(for: file),
                                          expectedSize: file.size > 0 ? file.size : nil)
                }
                let job = ModelDownloadJob(items: items, destinationDirectory: directory(for: model),
                                           headers: client.authorizationHeaders)
                UIApplication.shared.isIdleTimerDisabled = true
                try await job.run { progress in
                    Task { @MainActor in
                        guard self.tasks[repoID] != nil else { return }
                        self.downloads[repoID] = DownloadState(phase: .downloading, progress: progress)
                    }
                }
                model.isComplete = true
                try save(model)
                replace(model)
                downloads[repoID] = nil
            } catch is CancellationError {
                downloads[repoID] = DownloadState(phase: .paused, progress: downloads[repoID]?.progress)
            } catch {
                downloads[repoID] = DownloadState(phase: .failed(error.localizedDescription), progress: downloads[repoID]?.progress)
            }
            tasks[repoID] = nil
            UIApplication.shared.isIdleTimerDisabled = !tasks.isEmpty
        }
    }

    func pauseDownload(repoID: String) {
        tasks[repoID]?.cancel()
    }

    func delete(repoID: String) {
        tasks[repoID]?.cancel()
        tasks[repoID] = nil
        downloads[repoID] = nil
        if let model = model(repoID: repoID) {
            try? fileManager.removeItem(at: directory(for: model))
        }
        models.removeAll { $0.repoID == repoID }
    }

    var availableStorageBytes: Int64? {
        let values = try? rootDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    private func save(_ model: InstalledImageModel) throws {
        let folder = directory(for: model)
        try fileManager.createDirectory(at: folder, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(model).write(to: folder.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private func replace(_ model: InstalledImageModel) {
        if let index = models.firstIndex(where: { $0.id == model.id }) {
            models[index] = model
        } else {
            models.append(model)
        }
    }
}

enum ImageModelError: Error, LocalizedError {
    case unsupportedRepository
    case notInstalled

    var errorDescription: String? {
        switch self {
        case .unsupportedRepository:
            return L("このリポジトリには、対応している画像生成モデル（TextEncoder / Unet / VAEDecoder の .mlmodelc と vocab.json・merges.txt）が見つかりませんでした。",
                     "No supported image model (TextEncoder / Unet / VAEDecoder .mlmodelc with vocab.json and merges.txt) was found in this repository.")
        case .notInstalled:
            return L("先に画像生成モデルをダウンロードしてください。", "Download an image model first.")
        }
    }
}
