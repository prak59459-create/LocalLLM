import XCTest
@testable import LocalLLMCore

final class VideoFramesTests: XCTestCase {
    func testTimelineInterleavesBlendedFrames() {
        let timeline = VideoTimeline(keyframes: 3, inBetween: 2)
        XCTAssertEqual(timeline.frames.count, 7)
        XCTAssertEqual(timeline.frames[0], .init(first: 0, second: 0, weight: 0))
        XCTAssertEqual(timeline.frames[1].first, 0)
        XCTAssertEqual(timeline.frames[1].second, 1)
        XCTAssertEqual(timeline.frames[1].weight, 1 / 3, accuracy: 1e-6)
        XCTAssertEqual(timeline.frames[2].weight, 2 / 3, accuracy: 1e-6)
        XCTAssertEqual(timeline.frames[3], .init(first: 1, second: 1, weight: 0))
        XCTAssertEqual(timeline.frames.last, .init(first: 2, second: 2, weight: 0))
        XCTAssertEqual(VideoTimeline(keyframes: 4, inBetween: 0).frames.count, 4)
        XCTAssertEqual(VideoTimeline(keyframes: 1, inBetween: 3).frames.count, 1)
        var settings = VideoSettings()
        settings.keyframes = 8
        settings.inBetween = 2
        settings.fps = 11
        XCTAssertEqual(settings.frameCount, 22)
        XCTAssertEqual(settings.duration, 2, accuracy: 1e-9)
    }

    func testSlerpKeepsEndsAndMagnitude() {
        var noise = DiffusionNoise(seed: 1)
        let a = noise.normals(4096), b = noise.normals(4096)
        XCTAssertEqual(DiffusionMath.slerp(a, b, 0), a)
        let end = DiffusionMath.slerp(a, b, 1)
        for (x, y) in zip(end, b) { XCTAssertEqual(x, y, accuracy: 1e-4) }
        // 途中でも大きさ（ノイズの強さ）がほぼ同じ（線形補間だと小さくなる）
        func norm(_ v: [Float]) -> Float { v.map { $0 * $0 }.reduce(0, +).squareRoot() }
        let middle = DiffusionMath.slerp(a, b, 0.5)
        XCTAssertEqual(norm(middle) / norm(a), 1, accuracy: 0.03)
        XCTAssertLessThan(norm(DiffusionMath.lerp(a, b, 0.5)) / norm(a), 0.8)
    }

    func testCameraIdentityZoomAndPan() {
        let channels = 2, height = 8, width = 8
        let latent = (0..<(channels * height * width)).map { Float($0 % 64) }
        XCTAssertEqual(LatentCamera().apply(latent, channels: channels, height: height, width: width), latent)
        // ずらすと、右隣の値が来る（端はそのまま）
        let panned = LatentCamera(panX: 1 / 8).apply(latent, channels: channels, height: height, width: width)
        XCTAssertEqual(panned[0], latent[1], accuracy: 1e-5)
        XCTAssertEqual(panned[width - 1], latent[width - 1], accuracy: 1e-5)
        XCTAssertEqual(panned[height * width], latent[height * width + 1], accuracy: 1e-5)
        // 近づくと、端は中心に寄った点（2 倍なら (1.75, 1.75)）の値になる
        let zoomed = LatentCamera(zoom: 2).apply(latent, channels: channels, height: height, width: width)
        let expected = latent[width + 1] * 0.0625 + latent[width + 2] * 0.1875 + latent[2 * width + 1] * 0.1875
            + latent[2 * width + 2] * 0.5625
        XCTAssertEqual(zoomed[0], expected, accuracy: 1e-3)
        // 回転 360° は元どおり
        let turned = LatentCamera(rotation: 360).apply(latent, channels: channels, height: height, width: width)
        for (x, y) in zip(turned, latent) { XCTAssertEqual(x, y, accuracy: 1e-3) }
    }

    func testBlend() {
        XCTAssertEqual(FrameBlend.mix([0, 100, 200, 255], [255, 100, 0, 255], weight: 0.5), [128, 100, 100, 255])
        XCTAssertEqual(FrameBlend.mix([1, 2], [3, 4], weight: 0), [1, 2])
        XCTAssertEqual(FrameBlend.mix([1, 2], [3, 4], weight: 1), [3, 4])
    }

    /// 途中から描き直すとき: 強さに応じた段階から始め、その段階のノイズの量を足す
    func testSchedulerStartsMidway() {
        var scheduler = DPMSolverScheduler(stepCount: 20)
        XCTAssertEqual(scheduler.startIndex(strength: 1), 0)
        XCTAssertEqual(scheduler.startIndex(strength: 0.5), 10)
        XCTAssertEqual(scheduler.startIndex(strength: 0.01), 19)
        let original = [Float](repeating: 1, count: 4)
        let noise: [Float] = [0, 1, -1, 2]
        let index = scheduler.startIndex(strength: 0.45)
        let sigma = scheduler.noiseLevel(at: index)
        let alpha = Float(1 / (sigma * sigma + 1).squareRoot())
        let noised = scheduler.noised(original, noise: noise, at: index)
        for (value, n) in zip(noised, noise) {
            XCTAssertEqual(value, alpha + Float(sigma) * alpha * n, accuracy: 1e-5)
        }
        scheduler.start(at: index)
        XCTAssertEqual(scheduler.stepIndex, index)
        // 残りの段階を進めると終わる（最後は σ = 0）
        var sample = noised
        while !scheduler.isFinished {
            sample = scheduler.step(output: [Float](repeating: 0, count: 4), sample: sample)
        }
        XCTAssertEqual(scheduler.stepIndex, 20)
        // ノイズの予測が 0 なら、見積もりは sample / α のまま → 最後は元の画像 + 足したノイズの分
        XCTAssertTrue(sample.allSatisfy(\.isFinite))
    }
}
