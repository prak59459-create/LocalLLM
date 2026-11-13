import CoreML
import Foundation
import Observation
import UIKit

/// 作った画像 1 枚の記録（画像のフォルダの index.json に保存する）。
struct GeneratedImage: Codable, Identifiable, Hashable, Sendable {
    var id: UUID
    var fileName: String
    var prompt: String
    var negativePrompt: String
    var seed: UInt64
    var stepCount: Int
    var guidanceScale: Float
    var modelTitle: String
    var createdAt: Date
    var seconds: Double
}

/// 画像生成の画面の状態と操作。
@MainActor
@Observable
final class ImageStudio {
    var prompt = ""
    var negativePrompt = ""
    var stepCount = 20
    var guidanceScale: Float = 7.5
    /// 種を毎回ランダムにする
    var randomSeed = true
    var seed: UInt64 = 42
    var selectedRepoID: String?

    private(set) var isGenerating = false
    private(set) var status: String?
    /// ノイズ除去の進み具合
    private(set) var step: (done: Int, total: Int)?
    private(set) var errorMessage: String?
    private(set) var images: [GeneratedImage] = []
    /// いま表示している画像
    var currentID: UUID?
    /// 画像生成の画面を出しているか
    var isPresented = false

    @ObservationIgnored private let generator = ImageGenerator()
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let fileManager = FileManager.default
    let directory: URL

    init() {
        let base = (try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        directory = base.appendingPathComponent("Images", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        if let data = try? Data(contentsOf: indexURL), let saved = try? JSONDecoder().decode([GeneratedImage].self, from: data) {
            images = saved.filter { fileManager.fileExists(atPath: directory.appendingPathComponent($0.fileName).path) }
        }
        currentID = images.first?.id
    }

    private var indexURL: URL { directory.appendingPathComponent("index.json") }

    var current: GeneratedImage? {
        images.first { $0.id == currentID }
    }

    func url(for image: GeneratedImage) -> URL {
        directory.appendingPathComponent(image.fileName)
    }

    func uiImage(for image: GeneratedImage) -> UIImage? {
        UIImage(contentsOfFile: url(for: image).path)
    }

    var canGenerate: Bool {
        !isGenerating && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// 画像を作る。文章のモデルが読み込まれていたら、メモリを空けるために先に解放する。
    func generate(store: ImageModelStore, chat: ChatController) {
        guard canGenerate else { return }
        guard let model = store.completedModels.first(where: { $0.repoID == selectedRepoID }) ?? store.completedModels.first,
              let files = store.files(for: model) else {
            errorMessage = ImageModelError.notInstalled.localizedDescription
            return
        }
        selectedRepoID = model.repoID
        if randomSeed { seed = UInt64.random(in: 0..<4_294_967_296) }
        let request = ImageRequest(prompt: prompt, negativePrompt: negativePrompt, stepCount: stepCount,
                                   guidanceScale: guidanceScale, seed: seed)
        isGenerating = true
        errorMessage = nil
        step = nil
        status = L("準備しています…", "Preparing…")
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task { [self] in
            let started = Date()
            if chat.loadedInfo != nil {
                status = L("メモリを空けるため、文章のモデルを解放しています…", "Unloading the chat model to free memory…")
                await chat.unloadModelAndWait()
            }
            do {
                for try await event in generator.generate(files: files, request: request, computeUnits: .cpuAndNeuralEngine) {
                    switch event {
                    case .status(let text):
                        status = text
                    case .step(let done, let total):
                        step = (done, total)
                        if done > 0 { status = L("描いています… \(done)/\(total)", "Drawing… \(done)/\(total)") }
                    case .finished(let image):
                        try keep(image, request: request, model: model, seconds: Date().timeIntervalSince(started))
                    }
                }
            } catch is CancellationError {
                // 止めた
            } catch {
                errorMessage = error.localizedDescription
            }
            isGenerating = false
            status = nil
            step = nil
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    func cancel() {
        task?.cancel()
    }

    private func keep(_ image: CGImage, request: ImageRequest, model: InstalledImageModel, seconds: Double) throws {
        let id = UUID()
        let fileName = "\(id.uuidString).png"
        guard let data = UIImage(cgImage: image).pngData() else { return }
        try data.write(to: directory.appendingPathComponent(fileName), options: .atomic)
        let record = GeneratedImage(id: id, fileName: fileName, prompt: request.prompt, negativePrompt: request.negativePrompt,
                                    seed: request.seed, stepCount: request.stepCount, guidanceScale: request.guidanceScale,
                                    modelTitle: model.title, createdAt: Date(), seconds: seconds)
        images.insert(record, at: 0)
        currentID = id
        saveIndex()
    }

    func delete(_ image: GeneratedImage) {
        try? fileManager.removeItem(at: url(for: image))
        images.removeAll { $0.id == image.id }
        if currentID == image.id { currentID = images.first?.id }
        saveIndex()
    }

    /// その画像を作ったときの設定に戻す（同じ種でもう一度作れる）
    func reuseSettings(of image: GeneratedImage) {
        prompt = image.prompt
        negativePrompt = image.negativePrompt
        stepCount = image.stepCount
        guidanceScale = image.guidanceScale
        seed = image.seed
        randomSeed = false
    }

    private func saveIndex() {
        if let data = try? JSONEncoder().encode(images) { try? data.write(to: indexURL, options: .atomic) }
    }
}
