import Foundation

/// 画像生成（拡散モデル）のノイズ除去の進め方: DPM-Solver++（2M）。
///
/// diffusers の DPMSolverMultistepScheduler（既定の設定）と同じ計算。少ないステップ数（20 前後）でもきれいな画像になる。
/// ノイズの大きさ σ の並びを作り、モデルの出力（予測したノイズ）から「ノイズの無い画像」を見積もって、
/// 1 つ前の見積もりも使って次の σ まで進める。最後の 1 回は σ = 0（ノイズの無い画像そのもの）まで進める。
struct DPMSolverScheduler {
    enum Prediction: String, Sendable {
        /// モデルはノイズを予測する（Stable Diffusion 1.x・2.x base）
        case epsilon
        /// モデルは速度を予測する（Stable Diffusion 2.x の 768 版）
        case velocity
    }

    /// モデルに渡す時刻（大きい順）
    let timesteps: [Int]
    let prediction: Prediction
    /// 各ステップのノイズの大きさ（最後に 0 が付く）
    private let sigmas: [Double]
    /// これまでに見積もった「ノイズの無い画像」（新しいものが後ろ。2 つまで）
    private var estimates: [[Float]] = []
    private(set) var stepIndex = 0

    /// 最初のノイズに掛ける倍率
    let initialNoiseScale: Float = 1

    init(stepCount: Int, trainStepCount: Int = 1000, betaStart: Double = 0.00085, betaEnd: Double = 0.012,
         prediction: Prediction = .epsilon) {
        let steps = max(1, stepCount)
        self.prediction = prediction
        // β は √β が等間隔（scaled_linear）。ᾱ はその累積
        var cumulative: [Double] = []
        cumulative.reserveCapacity(trainStepCount)
        var product = 1.0
        let low = betaStart.squareRoot(), high = betaEnd.squareRoot()
        for index in 0..<trainStepCount {
            let root = low + (high - low) * Double(index) / Double(max(trainStepCount - 1, 1))
            product *= 1 - root * root
            cumulative.append(product)
        }
        // 時刻は 0 〜 最後 を等分して、大きい順に（0 は含めない）
        let last = Double(trainStepCount - 1)
        var times: [Int] = []
        for index in stride(from: steps, to: 0, by: -1) {
            times.append(Int((last * Double(index) / Double(steps)).rounded(.toNearestOrEven)))
        }
        timesteps = times
        var values = times.map { time -> Double in
            let alpha = cumulative[min(max(time, 0), trainStepCount - 1)]
            return ((1 - alpha) / alpha).squareRoot()
        }
        values.append(0)
        sigmas = values
    }

    var isFinished: Bool { stepIndex >= timesteps.count }

    /// σ から、画像に掛かる倍率 α とノイズに掛かる倍率
    private static func scales(_ sigma: Double) -> (alpha: Double, sigma: Double) {
        let alpha = 1 / (sigma * sigma + 1).squareRoot()
        return (alpha, sigma * alpha)
    }

    /// モデルの出力（今の時刻のもの）と今の画像から、次の画像を返す。
    mutating func step(output: [Float], sample: [Float]) -> [Float] {
        precondition(output.count == sample.count)
        guard stepIndex < timesteps.count else { return sample }
        let current = Self.scales(sigmas[stepIndex])
        let next = Self.scales(sigmas[stepIndex + 1])

        // ノイズの無い画像の見積もり
        var estimate = [Float](repeating: 0, count: sample.count)
        switch prediction {
        case .epsilon:
            let a = Float(1 / current.alpha), b = Float(-current.sigma / current.alpha)
            for index in 0..<sample.count { estimate[index] = a * sample[index] + b * output[index] }
        case .velocity:
            let a = Float(current.alpha), b = Float(-current.sigma)
            for index in 0..<sample.count { estimate[index] = a * sample[index] + b * output[index] }
        }
        estimates.append(estimate)
        if estimates.count > 2 { estimates.removeFirst() }

        let lambdaCurrent = log(current.alpha) - log(current.sigma)
        let lambdaNext = log(next.alpha) - log(next.sigma)
        let h = lambdaNext - lambdaCurrent
        let keep = Float(next.sigma / current.sigma)
        let gain = Float(-next.alpha * (exp(-h) - 1))

        let isLast = stepIndex == timesteps.count - 1
        var result = [Float](repeating: 0, count: sample.count)
        if estimates.count < 2 || isLast {
            // 1 次（最初と最後）
            for index in 0..<sample.count { result[index] = keep * sample[index] + gain * estimate[index] }
        } else {
            // 2 次: 1 つ前の見積もりとの差で補正する
            let previous = Self.scales(sigmas[stepIndex - 1])
            let lambdaPrevious = log(previous.alpha) - log(previous.sigma)
            let ratio = (lambdaCurrent - lambdaPrevious) / h
            let correction = Float(0.5 / ratio) * gain
            let older = estimates[0]
            for index in 0..<sample.count {
                result[index] = keep * sample[index] + gain * estimate[index] + correction * (estimate[index] - older[index])
            }
        }
        stepIndex += 1
        return result
    }
}

/// プロンプトに従わせる強さ（classifier-free guidance）: 無条件の予測から、条件付きの予測の方向へ scale 倍だけ進める。
enum DiffusionGuidance {
    static func combine(unconditional: [Float], conditional: [Float], scale: Float) -> [Float] {
        var result = [Float](repeating: 0, count: conditional.count)
        for index in 0..<min(unconditional.count, conditional.count) {
            result[index] = unconditional[index] + scale * (conditional[index] - unconditional[index])
        }
        return result
    }
}

/// 決まった種から、同じ正規乱数の列を作る（同じ種なら同じ画像になるように）。
struct DiffusionNoise {
    private var state: UInt64
    private var spare: Double?

    init(seed: UInt64) {
        state = seed &+ 0x9E37_79B9_7F4A_7C15
    }

    private mutating func nextBits() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    private mutating func uniform() -> Double {
        (Double(nextBits() >> 11) + 0.5) / Double(1 << 53)
    }

    mutating func normal() -> Float {
        if let value = spare {
            spare = nil
            return Float(value)
        }
        let radius = (-2 * log(uniform())).squareRoot()
        let angle = 2 * Double.pi * uniform()
        spare = radius * sin(angle)
        return Float(radius * cos(angle))
    }

    mutating func normals(_ count: Int) -> [Float] {
        (0..<count).map { _ in normal() }
    }
}
