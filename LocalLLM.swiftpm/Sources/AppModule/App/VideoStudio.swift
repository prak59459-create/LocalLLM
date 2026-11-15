import CoreML
import Foundation
import UIKit

/// 動画生成の操作（画像生成の画面の中の「動画」）。
extension ImageStudio {
    private var videoIndexURL: URL { videoDirectory.appendingPathComponent("index.json") }

    func loadVideos() {
        guard let data = try? Data(contentsOf: videoIndexURL),
              let saved = try? JSONDecoder().decode([GeneratedVideo].self, from: data) else { return }
        videos = saved.filter { fileManager.fileExists(atPath: videoDirectory.appendingPathComponent($0.fileName).path) }
        currentVideoID = videos.first?.id
    }

    private func saveVideoIndex() {
        if let data = try? JSONEncoder().encode(videos) { try? data.write(to: videoIndexURL, options: .atomic) }
    }

    var currentVideo: GeneratedVideo? {
        videos.first { $0.id == currentVideoID }
    }

    func url(for video: GeneratedVideo) -> URL {
        videoDirectory.appendingPathComponent(video.fileName)
    }

    /// 動画を作る。文章のモデルが読み込まれていたら、メモリを空けるために先に解放する。
    func generateVideo(store: ImageModelStore, chat: ChatController) {
        guard canGenerate else { return }
        guard let model = store.completedModels.first(where: { $0.repoID == selectedRepoID }) ?? store.completedModels.first,
              let files = store.files(for: model) else {
            errorMessage = ImageModelError.notInstalled.localizedDescription
            return
        }
        selectedRepoID = model.repoID
        if randomSeed { seed = UInt64.random(in: 0..<4_294_967_296) }
        let request = VideoRequest(prompt: prompt, endPrompt: endPrompt, negativePrompt: negativePrompt, stepCount: stepCount,
                                   guidanceScale: guidanceScale, seed: seed, settings: videoSettings)
        let id = UUID()
        let output = videoDirectory.appendingPathComponent("\(id.uuidString).mp4")
        isGenerating = true
        errorMessage = nil
        step = nil
        keyframeProgress = nil
        videoPreview = nil
        status = L("準備しています…", "Preparing…")
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task { [self] in
            let started = Date()
            if chat.loadedInfo != nil {
                status = L("メモリを空けるため、文章のモデルを解放しています…", "Unloading the chat model to free memory…")
                await chat.unloadModelAndWait()
            }
            do {
                for try await event in videoGenerator.generate(files: files, request: request, computeUnits: .cpuAndNeuralEngine,
                                                               output: output) {
                    handle(event, request: request, model: model, id: id, started: started)
                }
            } catch is CancellationError {
                // 止めた
            } catch {
                errorMessage = error.localizedDescription
            }
            isGenerating = false
            status = nil
            step = nil
            keyframeProgress = nil
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    private func handle(_ event: VideoEvent, request: VideoRequest, model: InstalledImageModel, id: UUID, started: Date) {
        switch event {
        case .status(let text):
            status = text
        case .step(let done, let total):
            step = (done, total)
            if let frames = keyframeProgress {
                status = L("描いています… コマ \(frames.done + 1)/\(frames.total)", "Drawing… frame \(frames.done + 1)/\(frames.total)")
            } else if done > 0 {
                status = L("描いています…", "Drawing…")
            }
        case .keyframe(let done, let total):
            keyframeProgress = (done, total)
        case .preview(let image):
            videoPreview = UIImage(cgImage: image)
        case .finished(let url):
            let record = GeneratedVideo(id: id, fileName: url.lastPathComponent, prompt: request.prompt, endPrompt: request.endPrompt,
                                        seed: request.seed, settings: request.settings, modelTitle: model.title,
                                        createdAt: Date(), seconds: Date().timeIntervalSince(started))
            videos.insert(record, at: 0)
            currentVideoID = id
            saveVideoIndex()
        }
    }

    func delete(_ video: GeneratedVideo) {
        try? fileManager.removeItem(at: url(for: video))
        videos.removeAll { $0.id == video.id }
        if currentVideoID == video.id { currentVideoID = videos.first?.id }
        saveVideoIndex()
    }

    /// その動画を作ったときの設定に戻す
    func reuseSettings(of video: GeneratedVideo) {
        prompt = video.prompt
        endPrompt = video.endPrompt
        videoSettings = video.settings
        seed = video.seed
        randomSeed = false
    }
}
