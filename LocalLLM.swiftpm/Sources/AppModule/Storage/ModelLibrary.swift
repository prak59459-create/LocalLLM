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
    /// コンテキスト長を変えている途中のモデル ID
    private(set) var resizingModelIDs: Set<String> = []

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
            weightBytes: weightBytes > 0 ? weightBytes : nil,
            pipeline: variant.pipeline,
            renames: variant.renames
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
                                  relativePath: model.localPath(for: file.path),
                                  expectedSize: file.size > 0 ? file.size : nil)
        }
        let job = ModelDownloadJob(items: items, destinationDirectory: directory(for: model),
                                   headers: client.authorizationHeaders)
        downloads[id] = DownloadState(phase: .downloading, progress: downloads[id]?.progress)
        beginBackgroundTask(for: id)
        updateIdleTimer()

        // ModelLibrary はアプリが終わるまで生きているので、強く参照してよい
        tasks[id] = Task { [self] in
            do {
                try await job.run { progress in
                    Task { @MainActor in
                        guard self.tasks[id] != nil else { return }
                        self.downloads[id] = DownloadState(phase: .downloading, progress: progress)
                    }
                }
                finishDownload(id: id)
            } catch {
                failDownload(id: id, error: error)
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
        guard let model = model(withID: id) else { return }
        if model.kind == .archive {
            // ZIP を展開して、中のモデルを調べ直す（時間がかかるので別のスレッドで）
            downloads[id] = DownloadState(phase: .downloading, progress: downloads[id]?.progress)
            let folder = directory(for: model)
            tasks[id] = Task { [self] in
                do {
                    let adopted = try await Task.detached(priority: .userInitiated) {
                        try Self.expandArchive(of: model, in: folder)
                    }.value
                    tasks[id] = nil
                    completeDownload(adopted)
                } catch {
                    failDownload(id: id, error: error)
                }
            }
            return
        }
        completeDownload(model)
    }

    private func completeDownload(_ original: InstalledModel) {
        var model = original
        model.isComplete = true
        if model.kind == .compiled {
            model.compiledModelPath = model.localPath(for: model.variantPath)
            // ダウンロードした metadata.json から最大長などを読み直す
            let metadataURL = directory(for: model).appendingPathComponent(model.compiledModelPath ?? "")
                .appendingPathComponent("metadata.json")
            if let data = try? Data(contentsOf: metadataURL), let metadata = try? CoreMLMetadata.parse(data) {
                model.maxContext = metadata.interface.maxContext ?? model.maxContext
                model.stateBytes = metadata.stateBytes
            }
        }
        if model.kind == .pipeline, let context = model.pipeline?.contextLength {
            model.maxContext = context
        }
        try? save(model)
        replace(model)
        downloads[model.id] = nil
    }

    /// ZIP を展開して（ZIP は消す）、中のモデルの種類を決め直す
    nonisolated private static func expandArchive(of original: InstalledModel, in folder: URL) throws -> InstalledModel {
        let fileManager = FileManager.default
        let archiveURL = folder.appendingPathComponent(original.localPath(for: original.variantPath))
        let written = try ZipArchive(url: archiveURL).extract(to: folder, inflate: ZipArchive.systemInflater)
        try? fileManager.removeItem(at: archiveURL)
        let expanded = written.map { path in
            let size = (try? fileManager.attributesOfItem(atPath: folder.appendingPathComponent(path).path)[.size] as? NSNumber)?.int64Value ?? 0
            return HFFileEntry(path: path, size: size, lfsSHA256: nil)
        }

        var model = original
        let supportFiles = model.files.filter { $0.path != model.variantPath }
        let analysis = RepoAnalysis.analyze(files: expanded + supportFiles, repoName: model.shortRepoName)
        guard let variant = analysis.variants.first(where: { $0.kind != .archive }) else {
            throw ModelLibraryError.missingModelFolder
        }
        // 拡張子の無いモデルのフォルダは、ダウンロードしたときと同じように名前を変えておく
        let renames = variant.renames ?? [:]
        for (from, to) in renames {
            let destination = folder.appendingPathComponent(to)
            try? fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if from.isEmpty {
                try? fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
                for file in expanded where RepoAnalysis.isLooseModelFile(file.path) {
                    let target = folder.appendingPathComponent(ModelVariant.localPath(for: file.path, renames: renames))
                    try? fileManager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try? fileManager.moveItem(at: folder.appendingPathComponent(file.path), to: target)
                }
            } else {
                try? fileManager.moveItem(at: folder.appendingPathComponent(from), to: destination)
            }
        }
        model.kind = variant.kind
        model.variantPath = variant.path
        model.pipeline = variant.pipeline
        model.renames = variant.renames
        model.displayName = variant.displayName
        model.files = analysis.filesToDownload(for: variant)
        return model
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

    /// 初回の読み込みでコンパイルが必要か（.mlpackage が残っている）
    func needsCompilation(_ model: InstalledModel) -> Bool {
        switch model.kind {
        case .package: return model.compiledModelPath == nil
        case .pipeline: return (model.pipeline?.parts ?? []).contains { model.localPath(for: $0).hasSuffix(".mlpackage") }
        default: return false
        }
    }

    /// 推論エンジンに渡すファイル一式を用意する。.mlpackage なら初回だけコンパイルする（数十秒〜数分）。
    func prepareFiles(for id: String) async throws -> ModelFiles {
        guard var model = model(withID: id) else { throw ModelLibraryError.missingModelFolder }
        guard model.isComplete else { throw ModelLibraryError.incomplete }
        let folder = directory(for: model)

        let source: ModelFiles.Source
        switch model.kind {
        case .pipeline:
            guard var pipeline = model.pipeline else { throw ModelLibraryError.missingModelFolder }
            // .mlpackage の部品があればコンパイルして、パスを .mlmodelc に置き換える
            if needsCompilation(model) {
                compilingModelIDs.insert(id)
                defer { compilingModelIDs.remove(id) }
                func compiled(_ path: String?) async throws -> String? {
                    guard let path else { return nil }
                    let local = model.localPath(for: path)
                    guard local.hasSuffix(".mlpackage") else { return path }
                    return try await compilePackage(at: local, in: folder)
                }
                pipeline.embeddings = try await compiled(pipeline.embeddings)
                var chunks: [String] = []
                for chunk in pipeline.chunks { chunks.append(try await compiled(chunk) ?? chunk) }
                pipeline.chunks = chunks
                pipeline.lmHead = try await compiled(pipeline.lmHead)
                model.pipeline = pipeline
                try save(model)
                replace(model)
            }
            func partURL(_ path: String) throws -> URL {
                let url = folder.appendingPathComponent(model.localPath(for: path))
                guard fileManager.fileExists(atPath: url.path) else { throw ModelLibraryError.missingModelFolder }
                return url
            }
            source = .pipeline(PipelineFiles(embeddings: try pipeline.embeddings.map(partURL),
                                             chunks: try pipeline.chunks.map(partURL),
                                             lmHead: try pipeline.lmHead.map(partURL),
                                             descriptor: pipeline))
        case .archive:
            throw ModelLibraryError.incomplete
        case .compiled, .package:
            if model.compiledModelPath == nil || !fileManager.fileExists(atPath: folder.appendingPathComponent(model.compiledModelPath ?? "").path) {
                if model.kind == .compiled {
                    model.compiledModelPath = model.localPath(for: model.variantPath)
                } else {
                    compilingModelIDs.insert(id)
                    defer { compilingModelIDs.remove(id) }
                    model.compiledModelPath = try await compilePackage(at: model.localPath(for: model.variantPath), in: folder)
                }
                try save(model)
                replace(model)
            }
            guard let compiledPath = model.compiledModelPath else { throw ModelLibraryError.missingModelFolder }
            source = .single(folder.appendingPathComponent(compiledPath))
        }

        guard let tokenizerPath = model.supportFilePath(named: "tokenizer.json") else { throw ModelLibraryError.missingTokenizer }

        func url(_ name: String) -> URL? {
            model.supportFilePath(named: name).map { folder.appendingPathComponent(model.localPath(for: $0)) }
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
            source: source,
            tokenizerJSON: folder.appendingPathComponent(model.localPath(for: tokenizerPath)),
            tokenizerConfigJSON: url("tokenizer_config.json"),
            generationConfigJSON: url("generation_config.json"),
            configJSON: url("config.json"),
            chatTemplateOverride: templateOverride,
            stateBytes: model.stateBytes
        )
    }

    /// .mlpackage をコンパイルして .mlmodelc にし、元のパッケージは消す。戻り値は .mlmodelc のパス。
    private func compilePackage(at relativePath: String, in folder: URL) async throws -> String {
        let packageURL = folder.appendingPathComponent(relativePath)
        guard fileManager.fileExists(atPath: packageURL.path) else { throw ModelLibraryError.missingModelFolder }
        let compiledTemp = try await MLModel.compileModel(at: packageURL)
        let compiledPath = (relativePath as NSString).deletingPathExtension + ".mlmodelc"
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
        return compiledPath
    }

    // MARK: - コンテキスト長

    /// コンテキスト長の書き換えの対象になる .mlmodelc（分割モデルは本体のチャンクだけ）
    private func contextTargets(for model: InstalledModel) -> [URL] {
        let folder = directory(for: model)
        switch model.kind {
        case .pipeline:
            return (model.pipeline?.chunks ?? []).map { folder.appendingPathComponent(model.localPath(for: $0)) }
        case .compiled, .package:
            return model.compiledModelPath.map { [folder.appendingPathComponent($0)] } ?? []
        case .archive:
            return []
        }
    }

    private func contextBackup(for model: InstalledModel, index: Int) -> URL {
        directory(for: model).appendingPathComponent(".context-backup", isDirectory: true)
            .appendingPathComponent("part\(index)", isDirectory: true)
    }

    /// コンテキスト長を変えられるかを調べる（今の値・1 トークンあたりの KV キャッシュ）
    func contextAnalysis(for id: String) async throws -> ContextResizer.Analysis {
        // .mlpackage は先にコンパイルする
        if let model = model(withID: id), needsCompilation(model) {
            _ = try await prepareFiles(for: id)
        }
        guard let model = model(withID: id) else { throw ModelLibraryError.missingModelFolder }
        let targets = contextTargets(for: model)
        guard let first = targets.first else {
            throw ContextResizer.ResizeError.unsupported(L("コンパイル済みのモデルが見つかりません", "no compiled model was found"))
        }
        return try await Task.detached(priority: .userInitiated) {
            try ContextResizer.analyze(modelDirectory: first)
        }.value
    }

    /// コンテキスト長を変える。書き換えたあと Core ML で読み込めるか確かめ、だめなら元に戻す。
    func resizeContext(id: String, to newContext: Int) async throws {
        if let model = model(withID: id), needsCompilation(model) {
            _ = try await prepareFiles(for: id)
        }
        guard var model = model(withID: id) else { throw ModelLibraryError.missingModelFolder }
        let targets = contextTargets(for: model)
        guard !targets.isEmpty else {
            throw ContextResizer.ResizeError.unsupported(L("コンパイル済みのモデルが見つかりません", "no compiled model was found"))
        }
        let backups = targets.indices.map { contextBackup(for: model, index: $0) }
        resizingModelIDs.insert(id)
        defer { resizingModelIDs.remove(id) }

        let original = try await Task.detached(priority: .userInitiated) { () throws -> Int in
            var originalContext = 0
            do {
                for (index, target) in targets.enumerated() {
                    let analysis = try ContextResizer.resize(modelDirectory: target, to: newContext, backup: backups[index])
                    originalContext = analysis.context
                }
                // 書き換えたモデルが読み込めるか（型や形の食い違いがあればここで分かる）
                for target in targets {
                    try ModelLoadCheck.verify(target)
                }
            } catch {
                for (index, target) in targets.enumerated() {
                    try? ContextResizer.restore(modelDirectory: target, backup: backups[index])
                }
                throw error
            }
            return originalContext
        }.value

        if model.originalContext == nil { model.originalContext = original }
        if newContext == model.originalContext { model.originalContext = nil }
        model.maxContext = newContext
        if model.kind == .pipeline { model.pipeline?.contextLength = newContext }
        // KV キャッシュの大きさを読み直す
        if let first = targets.first, let data = try? Data(contentsOf: first.appendingPathComponent("metadata.json")),
           let metadata = try? CoreMLMetadata.parse(data) {
            model.stateBytes = metadata.stateBytes
        }
        try save(model)
        replace(model)
    }

    /// 元のコンテキスト長に戻す
    func restoreContext(id: String) async throws {
        guard var model = model(withID: id), let original = model.originalContext else { return }
        let targets = contextTargets(for: model)
        let backups = targets.indices.map { contextBackup(for: model, index: $0) }
        resizingModelIDs.insert(id)
        defer { resizingModelIDs.remove(id) }
        try await Task.detached(priority: .userInitiated) {
            for (index, target) in targets.enumerated() {
                try ContextResizer.restore(modelDirectory: target, backup: backups[index])
            }
        }.value
        model.maxContext = original
        if model.kind == .pipeline { model.pipeline?.contextLength = original }
        if let first = targets.first, let data = try? Data(contentsOf: first.appendingPathComponent("metadata.json")),
           let metadata = try? CoreMLMetadata.parse(data) {
            model.stateBytes = metadata.stateBytes
        }
        model.originalContext = nil
        try save(model)
        replace(model)
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
