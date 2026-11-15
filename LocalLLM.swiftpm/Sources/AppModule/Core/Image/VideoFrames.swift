import Foundation

/// 動画生成の計算（端末に依存しない部分）。
///
/// 動画は、画像モデル（Stable Diffusion）で「少しずつ違うコマ」を作り、その間をなめらかにつないで作る。
/// - 変化: 2 つのプロンプト（と 2 つのノイズ）の間を少しずつ進めながら、コマごとに描く
/// - カメラ: 1 コマ目を描いたあと、前のコマを少しズーム・移動・回転させてからノイズを足し、描き直す
enum VideoMotion: String, Codable, CaseIterable, Identifiable, Sendable {
    case morph
    case camera

    var id: String { rawValue }

    var label: String {
        switch self {
        case .morph: return L("変化（プロンプトの間）", "Morph (between prompts)")
        case .camera: return L("カメラ（ズーム・移動）", "Camera (zoom & pan)")
        }
    }
}

/// 動画の作り方の設定。
struct VideoSettings: Codable, Equatable, Sendable {
    var motion: VideoMotion = .camera
    /// 描くコマの数（キーフレーム）
    var keyframes = 8
    /// キーフレームの間に挟む、混ぜて作るコマの数
    var inBetween = 2
    /// 1 秒あたりのコマ数
    var fps = 8
    /// カメラ: 1 コマごとのズーム（1 より大きいと近づく）・移動（画面の幅に対する割合）・回転（度）
    var zoom: Float = 1.04
    var panX: Float = 0
    var panY: Float = 0
    var rotation: Float = 0
    /// カメラ: 前のコマをどれだけ描き直すか（0〜1。大きいほど変わる）
    var strength: Float = 0.45

    /// できあがる動画のコマ数
    var frameCount: Int { VideoTimeline(keyframes: keyframes, inBetween: inBetween).frames.count }
    /// できあがる動画の長さ（秒）
    var duration: Double { Double(frameCount) / Double(max(fps, 1)) }
}

/// キーフレームの間を混ぜてコマを並べる。
struct VideoTimeline: Equatable {
    struct Frame: Equatable {
        /// 混ぜる 2 つのキーフレームと、後ろのほうの重み（0 なら first だけ）
        var first: Int
        var second: Int
        var weight: Float
    }

    let frames: [Frame]

    init(keyframes: Int, inBetween: Int) {
        let keys = max(1, keyframes), between = max(0, inBetween)
        var frames: [Frame] = []
        for key in 0..<keys {
            frames.append(Frame(first: key, second: key, weight: 0))
            guard key + 1 < keys else { break }
            for index in 1...max(between, 1) where between > 0 {
                frames.append(Frame(first: key, second: key + 1, weight: Float(index) / Float(between + 1)))
            }
        }
        self.frames = frames
    }
}

enum DiffusionMath {
    /// 球面上の補間（2 つのノイズの間を、大きさを保ったまま進む）
    static func slerp(_ a: [Float], _ b: [Float], _ t: Float) -> [Float] {
        precondition(a.count == b.count)
        var dot: Double = 0, normA: Double = 0, normB: Double = 0
        for index in a.indices {
            dot += Double(a[index]) * Double(b[index])
            normA += Double(a[index]) * Double(a[index])
            normB += Double(b[index]) * Double(b[index])
        }
        let cosine = dot / max((normA * normB).squareRoot(), 1e-12)
        // ほとんど同じ向きなら、ふつうの線形補間
        guard abs(cosine) < 0.9995 else { return lerp(a, b, t) }
        let theta = Foundation.acos(max(-1, min(1, cosine)))
        let s = Foundation.sin(theta)
        let wa = Float(Foundation.sin((1 - Double(t)) * theta) / s), wb = Float(Foundation.sin(Double(t) * theta) / s)
        return zip(a, b).map { wa * $0 + wb * $1 }
    }

    static func lerp(_ a: [Float], _ b: [Float], _ t: Float) -> [Float] {
        zip(a, b).map { $0 + ($1 - $0) * t }
    }
}

/// 潜在表現（チャンネル × 高さ × 幅）を、カメラが動いたように変形する（双線形補間。外にはみ出したところは端の値）。
struct LatentCamera: Equatable {
    var zoom: Float = 1
    /// 移動（幅・高さに対する割合。正は右・下へ進む＝絵は左・上へ流れる）
    var panX: Float = 0
    var panY: Float = 0
    /// 回転（度）
    var rotation: Float = 0

    var isIdentity: Bool { zoom == 1 && panX == 0 && panY == 0 && rotation == 0 }

    func apply(_ latent: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        guard !isIdentity else { return latent }
        precondition(latent.count == channels * height * width)
        var output = [Float](repeating: 0, count: latent.count)
        let centerX = Float(width - 1) / 2, centerY = Float(height - 1) / 2
        let angle = rotation * .pi / 180
        let cosine = Foundation.cos(angle), sine = Foundation.sin(angle)
        let inverseZoom = 1 / max(zoom, 0.01)
        for y in 0..<height {
            for x in 0..<width {
                // 出力の点 → 元の点（ズームの逆・回転の逆・移動）
                let dx = (Float(x) - centerX) * inverseZoom, dy = (Float(y) - centerY) * inverseZoom
                let sourceX = cosine * dx + sine * dy + centerX + panX * Float(width)
                let sourceY = -sine * dx + cosine * dy + centerY + panY * Float(height)
                let clampedX = min(max(sourceX, 0), Float(width - 1)), clampedY = min(max(sourceY, 0), Float(height - 1))
                let x0 = Int(clampedX), y0 = Int(clampedY)
                let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
                let fx = clampedX - Float(x0), fy = clampedY - Float(y0)
                for channel in 0..<channels {
                    let base = channel * height * width
                    let top = latent[base + y0 * width + x0] * (1 - fx) + latent[base + y0 * width + x1] * fx
                    let bottom = latent[base + y1 * width + x0] * (1 - fx) + latent[base + y1 * width + x1] * fx
                    output[base + y * width + x] = top * (1 - fy) + bottom * fy
                }
            }
        }
        return output
    }
}

/// 画素の並び（RGBA 8 ビット）の 2 枚を混ぜる（キーフレームの間のコマ）
enum FrameBlend {
    static func mix(_ a: [UInt8], _ b: [UInt8], weight: Float) -> [UInt8] {
        precondition(a.count == b.count)
        guard weight > 0 else { return a }
        guard weight < 1 else { return b }
        let w = max(0, min(1, weight))
        var result = [UInt8](repeating: 0, count: a.count)
        for index in a.indices {
            let value = Float(a[index]) * (1 - w) + Float(b[index]) * w
            result[index] = UInt8(max(0, min(255, value.rounded())))
        }
        return result
    }
}

extension DPMSolverScheduler {
    /// 途中の段階（index 番目の時刻）のノイズの量で、ノイズの無い画像 original にノイズを足す（画像から画像を描き直すとき）
    func noised(_ original: [Float], noise: [Float], at index: Int) -> [Float] {
        let sigma = noiseLevel(at: index)
        let alpha = Float(1 / (sigma * sigma + 1).squareRoot())
        let scale = Float(sigma) * alpha
        return zip(original, noise).map { alpha * $0 + scale * $1 }
    }

    /// 描き直す強さ（0〜1）から、始める段階の番号（強さ 1 なら 0 番目＝最初から）
    func startIndex(strength: Float) -> Int {
        let count = timesteps.count
        let steps = Int((Float(count) * max(0, min(1, strength))).rounded())
        return max(0, min(count - 1, count - max(steps, 1)))
    }
}
