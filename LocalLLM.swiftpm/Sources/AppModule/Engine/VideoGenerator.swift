import AVFoundation
import CoreGraphics
import CoreML
import CoreVideo
import Foundation

/// 動画生成の依頼。
struct VideoRequest: Sendable {
    var prompt: String
    /// 変化: 最後のコマのプロンプト（空なら prompt と同じ）
    var endPrompt: String = ""
    var negativePrompt: String = ""
    var stepCount: Int = 20
    var guidanceScale: Float = 7.5
    var seed: UInt64 = 0
    var settings = VideoSettings()
}

/// 動画生成の途中経過。
enum VideoEvent: @unchecked Sendable {
    case status(String)
    /// 描いたキーフレームの数と全体
    case keyframe(Int, Int)
    /// UNet の計算の進み具合（済んだ回数, 全体）
    case step(Int, Int)
    /// できたキーフレーム（画面に途中経過として出す）
    case preview(CGImage)
    case finished(URL)
}

/// 画像モデル（Core ML 版の Stable Diffusion）で、短い動画を作る。
///
/// 1. プロンプトを特徴にする（テキストエンコーダは使い終わったら手放す）
/// 2. UNet でキーフレームの潜在表現を全部描く（変化: ノイズとプロンプトの間を進む。カメラ: 前のコマを動かして描き直す）
/// 3. VAE デコーダで全部を画像にする
/// 4. キーフレームの間を混ぜてコマを足し、MP4 に書き出す
/// モデルは 1 つずつしか読み込まないので、画像 1 枚のときと同じくらいのメモリで動く。
actor VideoGenerator {
    private var currentTask: Task<Void, Never>?

    nonisolated func generate(files: ImageModelFiles, request: VideoRequest, computeUnits: MLComputeUnits,
                              output: URL) -> AsyncThrowingStream<VideoEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await self.start(files: files, request: request, computeUnits: computeUnits, output: output, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func start(files: ImageModelFiles, request: VideoRequest, computeUnits: MLComputeUnits, output: URL,
                       continuation: AsyncThrowingStream<VideoEvent, Error>.Continuation) async {
        let previous = currentTask
        previous?.cancel()
        let task = Task {
            await previous?.value
            do {
                let url = try await Self.run(files: files, request: request, computeUnits: computeUnits, output: output) { event in
                    continuation.yield(event)
                }
                continuation.yield(.finished(url))
                continuation.finish()
            } catch {
                try? FileManager.default.removeItem(at: output)
                continuation.finish(throwing: error)
            }
        }
        currentTask = task
        await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        if currentTask == task { currentTask = nil }
    }

    private static func run(files: ImageModelFiles, request: VideoRequest, computeUnits: MLComputeUnits, output: URL,
                            report: @escaping @Sendable (VideoEvent) -> Void) async throws -> URL {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { throw ImageGenerationError.emptyPrompt }
        let end = request.endPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let settings = request.settings

        report(.status(L("プロンプトを読み込んでいます…", "Encoding the prompt…")))
        let prompts = [request.negativePrompt, prompt, end.isEmpty ? prompt : end]
        let embeddings = try await DiffusionText.encode(files: files, prompts: prompts, computeUnits: computeUnits)
        try Task.checkCancellation()

        report(.status(L("画像モデルを読み込んでいます…（初回は数分かかることがあります）",
                         "Loading the image model… (the first time can take a few minutes)")))
        let (latents, shape) = try await drawKeyframes(files: files, request: request, embeddings: embeddings,
                                                       computeUnits: computeUnits, report: report)
        try Task.checkCancellation()

        report(.status(L("コマを画像に仕上げています…", "Decoding the frames…")))
        let decoder = try await DiffusionDecoder(url: files.vaeDecoder, computeUnits: computeUnits)
        var keyframes: [CGImage] = []
        for latent in latents {
            try Task.checkCancellation()
            let image = try await decoder.decode(latent, shape: shape)
            keyframes.append(image)
            report(.preview(image))
        }

        report(.status(L("動画に書き出しています…", "Writing the video…")))
        try await VideoWriter.write(keyframes: keyframes, timeline: VideoTimeline(keyframes: keyframes.count, inBetween: settings.inBetween),
                                    fps: settings.fps, to: output)
        return output
    }

    /// UNet でキーフレームの潜在表現を描く（UNet はこの関数を出るときに手放す）
    private static func drawKeyframes(files: ImageModelFiles, request: VideoRequest, embeddings: DiffusionEmbeddings,
                                      computeUnits: MLComputeUnits, report: @escaping @Sendable (VideoEvent) -> Void) async throws
        -> ([[Float]], [Int]) {
        let unet = try await DiffusionUNet(url: files.unet, computeUnits: computeUnits)
        let settings = request.settings
        let count = max(1, settings.keyframes)
        let negative = embeddings.arranged(0), first = embeddings.arranged(1), last = embeddings.arranged(2)
        let steps = max(1, request.stepCount)
        let strength = max(0.05, min(1, settings.strength))
        let template = DPMSolverScheduler(stepCount: steps)
        let partialSteps = template.timesteps.count - template.startIndex(strength: strength)
        let totalSteps = settings.motion == .camera ? steps + (count - 1) * partialSteps : steps * count
        var done = 0
        var latents: [[Float]] = []
        var noise = DiffusionNoise(seed: request.seed)
        let startNoise = noise.normals(unet.latentCount)
        var endSource = DiffusionNoise(seed: request.seed &+ 1)
        let endNoise = endSource.normals(unet.latentCount)
        let shape = unet.latentShape
        let camera = LatentCamera(zoom: settings.zoom, panX: settings.panX, panY: settings.panY, rotation: settings.rotation)
        report(.step(0, totalSteps))
        for index in 0..<count {
            try Task.checkCancellation()
            let t = count > 1 ? Float(index) / Float(count - 1) : 0
            var scheduler = DPMSolverScheduler(stepCount: steps)
            let positive: [Float]
            var start: [Float]
            switch settings.motion {
            case .morph:
                positive = DiffusionMath.lerp(first, last, t)
                start = DiffusionMath.slerp(startNoise, endNoise, t)
            case .camera:
                positive = DiffusionMath.lerp(first, last, t)
                if let previous = latents.last {
                    // 前のコマを動かし、ノイズを足して途中から描き直す
                    let moved = camera.apply(previous, channels: shape[1], height: shape[2], width: shape[3])
                    var frameNoise = DiffusionNoise(seed: request.seed &+ UInt64(index) &* 7919)
                    let begin = scheduler.startIndex(strength: strength)
                    start = scheduler.noised(moved, noise: frameNoise.normals(unet.latentCount), at: begin)
                    scheduler.start(at: begin)
                } else {
                    start = startNoise
                }
            }
            let hidden = try unet.hiddenStates(negative: negative, positive: positive)
            let base = done, skipped = scheduler.stepIndex
            let latent = try await unet.denoise(start, scheduler: &scheduler, hidden: hidden, guidance: request.guidanceScale) { step, _ in
                report(.step(min(totalSteps, base + step - skipped), totalSteps))
            }
            done += scheduler.timesteps.count - skipped
            latents.append(latent)
            report(.keyframe(index + 1, count))
        }
        return (latents, shape)
    }
}

/// コマを MP4（H.264）に書き出す。
enum VideoWriter {
    enum WriterError: Error, LocalizedError {
        case cannotStart(String)
        case noFrames

        var errorDescription: String? {
            switch self {
            case .cannotStart(let detail): return L("動画を書き出せません: \(detail)", "Cannot write the video: \(detail)")
            case .noFrames: return L("コマがありません。", "There are no frames.")
            }
        }
    }

    static func write(keyframes: [CGImage], timeline: VideoTimeline, fps: Int, to url: URL) async throws {
        guard let first = keyframes.first else { throw WriterError.noFrames }
        let width = first.width, height = first.height
        try? FileManager.default.removeItem(at: url)
        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            throw WriterError.cannotStart(error.localizedDescription)
        }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
        guard writer.canAdd(input) else { throw WriterError.cannotStart("input") }
        writer.add(input)
        guard writer.startWriting() else { throw WriterError.cannotStart(writer.error?.localizedDescription ?? "start") }
        writer.startSession(atSourceTime: .zero)

        let pixels = keyframes.map { bgra($0, width: width, height: height) }
        for (index, frame) in timeline.frames.enumerated() {
            try Task.checkCancellation()
            let mixed = FrameBlend.mix(pixels[min(frame.first, pixels.count - 1)], pixels[min(frame.second, pixels.count - 1)],
                                       weight: frame.weight)
            guard let buffer = pixelBuffer(mixed, width: width, height: height, pool: adaptor.pixelBufferPool) else {
                throw WriterError.cannotStart("pixel buffer")
            }
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 10_000_000)
            }
            let time = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(max(fps, 1)))
            guard adaptor.append(buffer, withPresentationTime: time) else {
                throw WriterError.cannotStart(writer.error?.localizedDescription ?? "append")
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed {
            throw WriterError.cannotStart(writer.error?.localizedDescription ?? "finish")
        }
    }

    /// 画像を BGRA の画素の並びにする
    static func bgra(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        pixels.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return pixels
    }

    private static func pixelBuffer(_ pixels: [UInt8], width: Int, height: Int, pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        if let pool {
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        } else {
            CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
        }
        guard let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        pixels.withUnsafeBytes { source in
            guard let start = source.baseAddress else { return }
            for row in 0..<height {
                (base + row * rowBytes).copyMemory(from: start + row * width * 4, byteCount: width * 4)
            }
        }
        return buffer
    }
}
