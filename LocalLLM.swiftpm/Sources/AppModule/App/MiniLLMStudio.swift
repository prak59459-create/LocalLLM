import Foundation
import Observation
import UIKit

/// 学習曲線の 1 点。
struct LossPoint: Identifiable, Hashable, Sendable {
    var id: Int { step * 2 + (isValidation ? 1 : 0) }
    var step: Int
    var loss: Float
    var isValidation: Bool
}

/// 保存したミニモデルの情報（info.json）。
struct MiniModelInfo: Codable, Identifiable, Hashable, Sendable {
    var id: UUID
    var name: String
    var preset: MiniGPTPreset
    var config: MiniGPTConfig
    var trainedSteps: Int
    var trainLoss: Float?
    var validationLoss: Float?
    var datasetName: String
    var datasetCharacters: Int
    var createdAt: Date
}

/// 学習中のモデル一式。バックグラウンドの学習ループだけが触るので @unchecked Sendable で包む。
private final class TrainingBox: @unchecked Sendable {
    let model: MiniGPT
    let trainer: MiniGPTTrainer
    let vocabulary: CharVocabulary
    let hyperparameters: TrainingHyperparameters

    init(model: MiniGPT, trainer: MiniGPTTrainer, vocabulary: CharVocabulary, hyperparameters: TrainingHyperparameters) {
        self.model = model
        self.trainer = trainer
        self.vocabulary = vocabulary
        self.hyperparameters = hyperparameters
    }
}

/// 学習ループから画面へ送る報告。
private struct TrainingReport: Sendable {
    var step: Int
    var trainLoss: Float?
    var validationLoss: Float?
    var learningRate: Float
    var tokensPerSecond: Double
    var sample: String?
    var finished: Bool
}

/// 「ミニLLM工房」: iPad の中で小さな GPT を文字単位でゼロから学習させる。
@MainActor
@Observable
final class MiniLLMStudio {
    enum Status: Equatable {
        case idle
        case training
        case paused
        case finished
        case failed(String)
    }

    // データ
    var datasetText: String = SampleCorpora.wagahai
    var datasetName: String = L("吾輩は猫である（冒頭）", "Wagahai wa Neko de Aru (opening)")
    private(set) var isDownloadingDataset = false

    // 設定
    var preset: MiniGPTPreset = .tiny
    var totalSteps: Int = 600
    var learningRate: Float = 3e-3
    var batchSize: Int = 16
    var samplePrompt: String = ""

    // 学習の状態
    private(set) var status: Status = .idle
    private(set) var step: Int = 0
    private(set) var lossPoints: [LossPoint] = []
    private(set) var latestTrainLoss: Float?
    private(set) var latestValidationLoss: Float?
    private(set) var currentLearningRate: Float = 0
    private(set) var tokensPerSecond: Double = 0
    private(set) var sampleText: String = ""
    private(set) var trainingStartedAt: Date?
    private(set) var activeConfig: MiniGPTConfig?

    // 保存したモデル
    private(set) var savedModels: [MiniModelInfo] = []

    @ObservationIgnored private var box: TrainingBox?
    @ObservationIgnored private var trainingTask: Task<Void, Never>?
    @ObservationIgnored private let rootDirectory: URL

    init() {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        rootDirectory = base.appendingPathComponent("MiniModels", isDirectory: true)
        try? FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        reloadSavedModels()
    }

    // MARK: - データ

    var datasetCharacterCount: Int { datasetText.unicodeScalars.count }

    /// 文字の種類の数（語彙サイズの目安）
    var datasetUniqueCharacters: Int { Set(datasetText.unicodeScalars).count }

    func useSample(_ text: String, name: String) {
        guard status != .training else { return }
        datasetText = text
        datasetName = name
        resetTraining()
    }

    func downloadTinyShakespeare() async -> String? {
        isDownloadingDataset = true
        defer { isDownloadingDataset = false }
        do {
            let (data, response) = try await URLSession.shared.data(from: SampleCorpora.tinyShakespeareURL)
            guard (response as? HTTPURLResponse)?.statusCode == 200, let text = String(data: data, encoding: .utf8) else {
                return L("ダウンロードに失敗しました。", "Download failed.")
            }
            useSample(text, name: "Tiny Shakespeare")
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    /// ファイル・フォルダのテキストをつなげて学習データにする。
    func importDataset(from urls: [URL]) async -> String {
        let collected = await Task.detached(priority: .userInitiated) { () -> (String, Int) in
            var parts: [String] = []
            var count = 0
            for url in urls {
                let result = KnowledgeStore.collectTextFiles(at: url)
                for (_, text) in result.files {
                    parts.append(text)
                    count += 1
                }
            }
            return (parts.joined(separator: "\n\n"), count)
        }.value
        guard !collected.0.isEmpty else {
            return L("読み込めるテキストがありませんでした。", "No readable text was found.")
        }
        let name = urls.count == 1 ? urls[0].lastPathComponent : L("\(collected.1) ファイル", "\(collected.1) files")
        useSample(String(collected.0.prefix(5_000_000)), name: name)
        return L("\(collected.1) ファイル（\(datasetCharacterCount) 文字）を読み込みました。",
                 "Loaded \(collected.1) files (\(datasetCharacterCount) characters).")
    }

    // MARK: - 学習

    /// 現在の設定で作るモデルの構成（語彙はデータから決まる）
    var plannedConfig: MiniGPTConfig {
        MiniGPTConfig.preset(preset, vocabSize: min(datasetUniqueCharacters + 1, 4096))
    }

    /// 学習に使うおおよそのメモリ（重み・勾配・Adam の状態 = パラメータ数 × 16 バイト + 作業領域）
    var plannedMemoryBytes: Int64 {
        let config = plannedConfig
        let parameters = Int64(config.parameterCount) * 16
        let activations = Int64(batchSize * config.contextLength) * Int64(config.embeddingSize) * Int64(config.layerCount) * 4 * 24
        let logits = Int64(batchSize * config.contextLength) * Int64(config.vocabSize) * 4 * 3
        return parameters + activations + logits
    }

    var canStart: Bool {
        datasetCharacterCount >= 200 && status != .training
    }

    /// 学習を始める（一時停止中なら続きから）。
    func startOrResume() {
        guard status != .training else { return }
        if box == nil || status == .finished || status == .idle {
            guard prepareNewTraining() else { return }
        }
        guard let box else { return }
        status = .training
        UIApplication.shared.isIdleTimerDisabled = true
        if trainingStartedAt == nil { trainingStartedAt = Date() }
        let totalSteps = self.totalSteps
        let prompt = samplePrompt

        // MiniLLMStudio はアプリが終わるまで生きているので、強く参照してよい
        trainingTask = Task.detached(priority: .userInitiated) { [self] in
            await MiniLLMStudio.runTrainingLoop(box: box, totalSteps: totalSteps, samplePrompt: prompt) { report in
                await self.apply(report)
            }
            await self.trainingLoopEnded()
        }
    }

    /// 一時停止（モデルはメモリに残るので、続きから再開できる）。
    func pause() {
        trainingTask?.cancel()
    }

    /// 学習を破棄して最初からにする。
    func resetTraining() {
        trainingTask?.cancel()
        trainingTask = nil
        box = nil
        status = .idle
        step = 0
        lossPoints = []
        latestTrainLoss = nil
        latestValidationLoss = nil
        tokensPerSecond = 0
        sampleText = ""
        trainingStartedAt = nil
        activeConfig = nil
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func prepareNewTraining() -> Bool {
        let vocabulary = CharVocabulary(text: datasetText, maxSize: 4096)
        let tokens = vocabulary.encode(datasetText)
        let config = MiniGPTConfig.preset(preset, vocabSize: vocabulary.size)
        guard tokens.count > config.contextLength + 1 else {
            status = .failed(L("データが短すぎます。もっと長い文章を入れてください。", "The dataset is too short. Add more text."))
            return false
        }
        var hyperparameters = TrainingHyperparameters()
        hyperparameters.learningRate = learningRate
        hyperparameters.minLearningRate = learningRate / 10
        hyperparameters.batchSize = batchSize
        hyperparameters.totalSteps = totalSteps
        hyperparameters.warmupSteps = min(100, max(10, totalSteps / 20))
        let model = MiniGPT(config: config, seed: hyperparameters.seed)
        let trainer = MiniGPTTrainer(model: model, tokens: tokens, hyperparameters: hyperparameters, validationFraction: 0.1)
        box = TrainingBox(model: model, trainer: trainer, vocabulary: vocabulary, hyperparameters: hyperparameters)
        activeConfig = config
        step = 0
        lossPoints = []
        sampleText = ""
        trainingStartedAt = nil
        return true
    }

    /// バックグラウンドで回る学習ループ。
    nonisolated private static func runTrainingLoop(box: TrainingBox, totalSteps: Int, samplePrompt: String,
                                                    report: @escaping @Sendable (TrainingReport) async -> Void) async {
        let config = box.model.config
        let tokensPerStep = Double(box.hyperparameters.batchSize * config.contextLength)
        var windowStart = Date()
        var stepsInWindow = 0
        var lastLoss: Float = .nan
        while box.trainer.stepIndex < totalSteps {
            if Task.isCancelled { break }
            lastLoss = box.trainer.step()
            stepsInWindow += 1
            let current = box.trainer.stepIndex
            let isLast = current >= totalSteps
            // 10 ステップごと（と最後）に画面へ報告
            if current % 10 == 0 || isLast {
                let elapsed = Date().timeIntervalSince(windowStart)
                let speed = elapsed > 0 ? Double(stepsInWindow) * tokensPerStep / elapsed : 0
                windowStart = Date()
                stepsInWindow = 0
                var validation: Float?
                var sample: String?
                // 100 ステップごと（と最後）に検証損失と文章の例
                if current % 100 == 0 || isLast {
                    if box.trainer.hasValidationData {
                        validation = box.trainer.validationLoss(batches: 4)
                    }
                    sample = makeSample(box: box, prompt: samplePrompt, length: 120, seed: UInt64(current))
                }
                await report(TrainingReport(step: current, trainLoss: lastLoss, validationLoss: validation,
                                            learningRate: box.trainer.currentLearningRate,
                                            tokensPerSecond: speed, sample: sample, finished: isLast))
            }
            // 他の処理（UI）にも CPU を譲る
            await Task.yield()
        }
    }

    nonisolated private static func makeSample(box: TrainingBox, prompt: String, length: Int, seed: UInt64) -> String {
        var context = box.vocabulary.encode(prompt)
        if context.isEmpty { context = [box.vocabulary.encode("\n").first ?? 0] }
        let generated = box.model.generate(prompt: context, maxNewTokens: length, temperature: 0.8, topK: 20, seed: seed)
        return prompt + box.vocabulary.decode(generated)
    }

    private func apply(_ report: TrainingReport) {
        step = report.step
        currentLearningRate = report.learningRate
        if report.tokensPerSecond > 0 { tokensPerSecond = report.tokensPerSecond }
        if let loss = report.trainLoss, loss.isFinite {
            latestTrainLoss = loss
            lossPoints.append(LossPoint(step: report.step, loss: loss, isValidation: false))
        }
        if let loss = report.validationLoss, loss.isFinite {
            latestValidationLoss = loss
            lossPoints.append(LossPoint(step: report.step, loss: loss, isValidation: true))
        }
        if let sample = report.sample { sampleText = sample }
        // グラフの点が多くなりすぎたら間引く
        if lossPoints.count > 2000 {
            lossPoints = lossPoints.enumerated().filter { $0.offset % 2 == 0 || $0.element.isValidation }.map(\.element)
        }
    }

    private func trainingLoopEnded() {
        trainingTask = nil
        UIApplication.shared.isIdleTimerDisabled = false
        guard let box else { status = .idle; return }
        status = box.trainer.stepIndex >= totalSteps ? .finished : .paused
    }

    /// 学習の残り時間の目安（秒）
    var estimatedSecondsRemaining: Double? {
        guard status == .training, tokensPerSecond > 0, let config = activeConfig else { return nil }
        let remainingSteps = Double(max(totalSteps - step, 0))
        return remainingSteps * Double(batchSize * config.contextLength) / tokensPerSecond
    }

    var hasTrainedModel: Bool { box != nil && step > 0 }

    // MARK: - 保存と読み込み

    /// 今のモデルを保存する（学習中なら一時停止してから）。
    func saveCurrentModel(name: String) async -> String? {
        if status == .training {
            pause()
            while trainingTask != nil { try? await Task.sleep(for: .milliseconds(50)) }
        }
        guard let box else { return L("保存するモデルがありません。", "There is no model to save.") }
        let id = UUID()
        let folder = rootDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
        let info = MiniModelInfo(id: id,
                                 name: name.isEmpty ? L("ミニモデル", "Mini model") : name,
                                 preset: preset,
                                 config: box.model.config,
                                 trainedSteps: box.trainer.stepIndex,
                                 trainLoss: latestTrainLoss,
                                 validationLoss: latestValidationLoss,
                                 datasetName: datasetName,
                                 datasetCharacters: datasetCharacterCount,
                                 createdAt: Date())
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try box.model.save(to: folder.appendingPathComponent("model.bin"))
            try JSONEncoder().encode(box.vocabulary).write(to: folder.appendingPathComponent("vocab.json"))
            try JSONEncoder().encode(info).write(to: folder.appendingPathComponent("info.json"))
            reloadSavedModels()
            return nil
        } catch {
            try? FileManager.default.removeItem(at: folder)
            return error.localizedDescription
        }
    }

    func deleteSavedModel(_ id: UUID) {
        try? FileManager.default.removeItem(at: rootDirectory.appendingPathComponent(id.uuidString))
        reloadSavedModels()
    }

    func reloadSavedModels() {
        let folders = (try? FileManager.default.contentsOfDirectory(at: rootDirectory, includingPropertiesForKeys: nil)) ?? []
        savedModels = folders.compactMap { folder in
            guard let data = try? Data(contentsOf: folder.appendingPathComponent("info.json")) else { return nil }
            return try? JSONDecoder().decode(MiniModelInfo.self, from: data)
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    /// 保存したモデルを読み込む（生成用）。
    func loadSavedModel(_ id: UUID) async throws -> LoadedMiniModel {
        let folder = rootDirectory.appendingPathComponent(id.uuidString)
        return try await Task.detached(priority: .userInitiated) {
            let model = try MiniGPT.load(from: folder.appendingPathComponent("model.bin"))
            let vocabulary = try JSONDecoder().decode(CharVocabulary.self,
                                                      from: Data(contentsOf: folder.appendingPathComponent("vocab.json")))
            return LoadedMiniModel(model: model, vocabulary: vocabulary)
        }.value
    }
}

/// 読み込んだミニモデル（生成用）。生成はバックグラウンドで 1 つずつ行う。
final class LoadedMiniModel: @unchecked Sendable {
    let model: MiniGPT
    let vocabulary: CharVocabulary

    init(model: MiniGPT, vocabulary: CharVocabulary) {
        self.model = model
        self.vocabulary = vocabulary
    }

    /// 続きを少しずつ生成して返す（chunk 文字ごとに onChunk を呼ぶ）。
    func complete(prompt: String, length: Int, temperature: Float, topK: Int, seed: UInt64,
                  onChunk: @escaping @Sendable (String) async -> Void) async {
        var context = vocabulary.encode(prompt)
        if context.isEmpty { context = [vocabulary.encode("\n").first ?? 0] }
        var produced = 0
        var round: UInt64 = 0
        while produced < length {
            if Task.isCancelled { break }
            let count = min(8, length - produced)
            let generated = model.generate(prompt: context, maxNewTokens: count, temperature: temperature,
                                           topK: topK, seed: seed &+ round)
            guard !generated.isEmpty else { break }
            context.append(contentsOf: generated)
            produced += generated.count
            round += 1
            await onChunk(vocabulary.decode(generated))
            await Task.yield()
        }
    }
}
