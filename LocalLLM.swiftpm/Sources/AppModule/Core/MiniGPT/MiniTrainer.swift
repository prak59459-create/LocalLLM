import Foundation

/// PyTorch の torch.optim.AdamW と同じ式の最適化手法（勾配の全体ノルムでのクリップつき）。
struct AdamWOptimizer {
    let hyperparameters: TrainingHyperparameters
    private(set) var stepCount = 0
    private var firstMoments: [[Float]] = []
    private var secondMoments: [[Float]] = []

    init(hyperparameters: TrainingHyperparameters) {
        self.hyperparameters = hyperparameters
    }

    /// 戻り値はクリップ前の勾配ノルム。勾配に NaN / ∞ があるときは更新しない。
    mutating func step(parameters: [MiniGPTParameter], learningRate: Float) -> Float {
        if firstMoments.count != parameters.count {
            firstMoments = parameters.map { [Float](repeating: 0, count: $0.count) }
            secondMoments = firstMoments
        }
        var squares: Double = 0
        for parameter in parameters {
            squares += parameter.gradient.withUnsafeBufferPointer { Double(MiniMath.sumOfSquares($0.baseAddress!, count: $0.count)) }
        }
        let norm = Float(squares.squareRoot())
        guard norm.isFinite else { return norm }

        // torch.nn.utils.clip_grad_norm_ と同じ: 係数 = min(1, max / (norm + 1e-6))
        var clip: Float = 1
        if hyperparameters.gradientClip > 0 {
            clip = min(1, hyperparameters.gradientClip / (norm + 1e-6))
        }

        stepCount += 1
        let h = hyperparameters
        let correction1 = 1 - Float(pow(Double(h.beta1), Double(stepCount)))
        let correction2 = 1 - Float(pow(Double(h.beta2), Double(stepCount)))
        let stepSize = learningRate / correction1
        let correction2Root = correction2.squareRoot()

        for (index, parameter) in parameters.enumerated() {
            let decay = parameter.shape.count >= 2 ? h.weightDecay : 0
            update(parameter, m: &firstMoments[index], v: &secondMoments[index], clip: clip, decay: decay,
                   learningRate: learningRate, stepSize: stepSize, correction2Root: correction2Root)
        }
        return norm
    }

    private func update(_ parameter: MiniGPTParameter, m: inout [Float], v: inout [Float], clip: Float, decay: Float,
                        learningRate: Float, stepSize: Float, correction2Root: Float) {
        let n = parameter.count
        let h = hyperparameters
        var scratch = [Float](repeating: 0, count: n)
        parameter.gradient.withUnsafeMutableBufferPointer { g in
            parameter.value.withUnsafeMutableBufferPointer { p in
                m.withUnsafeMutableBufferPointer { m in
                    v.withUnsafeMutableBufferPointer { v in
                        scratch.withUnsafeMutableBufferPointer { s in
                            let g = g.baseAddress!, p = p.baseAddress!, m = m.baseAddress!, v = v.baseAddress!, s = s.baseAddress!
                            if clip != 1 { MiniVector.scale(g, clip, g, n) }
                            // p *= 1 - lr * wd
                            if decay != 0 { MiniVector.scale(p, 1 - learningRate * decay, p, n) }
                            // m = β1 m + (1 − β1) g
                            MiniVector.scale(m, h.beta1, m, n)
                            MiniVector.scaleAdd(g, 1 - h.beta1, m, m, n)
                            // v = β2 v + (1 − β2) g²
                            MiniVector.multiply(g, g, s, n)
                            MiniVector.scale(v, h.beta2, v, n)
                            MiniVector.scaleAdd(s, 1 - h.beta2, v, v, n)
                            // p -= lr / bc1 · m / (sqrt(v) / sqrt(bc2) + eps)
                            MiniVector.sqrt(v, s, n)
                            MiniVector.scaleShift(s, 1 / correction2Root, h.epsilon, s, n)
                            MiniVector.divide(m, s, s, n)
                            MiniVector.scaleAdd(s, -stepSize, p, p, n)
                        }
                    }
                }
            }
        }
    }
}

/// 学習ループ（データの切り出し、学習率スケジュール、評価）。
final class MiniGPTTrainer {
    let model: MiniGPT
    let hyperparameters: TrainingHyperparameters
    private(set) var stepIndex = 0
    private(set) var lastGradientNorm: Float = 0
    /// 損失や勾配が NaN / ∞ になって更新を飛ばした回数
    private(set) var skippedSteps = 0

    private var optimizer: AdamWOptimizer
    private let trainTokens: [Int]
    private let validationTokens: [Int]
    private var random: MiniRandom

    init(model: MiniGPT, tokens: [Int], hyperparameters: TrainingHyperparameters, validationFraction: Double = 0.1) {
        self.model = model
        self.hyperparameters = hyperparameters
        self.optimizer = AdamWOptimizer(hyperparameters: hyperparameters)
        self.random = MiniRandom(seed: hyperparameters.seed &+ 0x9E37_79B9)
        // 後ろの一部を検証用にする（どちらも 1 窓ぶん以上あるときだけ）
        let minimum = model.config.contextLength + 2
        let validationCount = Int(Double(tokens.count) * min(max(validationFraction, 0), 0.5))
        if validationCount >= minimum && tokens.count - validationCount >= minimum {
            trainTokens = Array(tokens[0..<(tokens.count - validationCount)])
            validationTokens = Array(tokens[(tokens.count - validationCount)...])
        } else {
            trainTokens = tokens
            validationTokens = []
        }
    }

    var hasValidationData: Bool { !validationTokens.isEmpty }

    /// 次のステップで使う学習率（線形ウォームアップ → コサインで minLearningRate まで）
    var currentLearningRate: Float { learningRate(at: stepIndex) }

    func learningRate(at step: Int) -> Float {
        let h = hyperparameters
        if h.warmupSteps > 0 && step < h.warmupSteps {
            return h.learningRate * Float(step + 1) / Float(h.warmupSteps)
        }
        let span = max(h.totalSteps - h.warmupSteps, 1)
        let progress = min(max(Double(step - h.warmupSteps) / Double(span), 0), 1)
        let cosine = 0.5 * (1 + cos(Double.pi * progress))
        return h.minLearningRate + Float(cosine) * (h.learningRate - h.minLearningRate)
    }

    /// ランダムな位置から batchSize 個の窓を切り出して 1 ステップ学習し、学習損失を返す。
    @discardableResult
    func step() -> Float {
        let batch = Self.makeBatch(tokens: trainTokens, count: max(hyperparameters.batchSize, 1),
                                   contextLength: model.config.contextLength, random: &random)
        model.zeroGradients()
        let loss = model.lossAndGradients(inputs: batch.inputs, targets: batch.targets)
        if loss.isFinite {
            lastGradientNorm = optimizer.step(parameters: model.parameters, learningRate: currentLearningRate)
            if !lastGradientNorm.isFinite { skippedSteps += 1 }
        } else {
            skippedSteps += 1
        }
        stepIndex += 1
        return loss
    }

    /// 検証データでの平均損失（batches 回ぶん）。毎回同じ位置を使うので、ステップ間で比べられる。
    func validationLoss(batches: Int) -> Float {
        guard hasValidationData, batches > 0 else { return .nan }
        var random = MiniRandom(seed: hyperparameters.seed &+ 0x5EED)
        var total: Float = 0
        for _ in 0..<batches {
            let batch = Self.makeBatch(tokens: validationTokens, count: max(hyperparameters.batchSize, 1),
                                       contextLength: model.config.contextLength, random: &random)
            total += model.loss(inputs: batch.inputs, targets: batch.targets)
        }
        return total / Float(batches)
    }

    static func makeBatch(tokens: [Int], count: Int, contextLength: Int,
                          random: inout MiniRandom) -> (inputs: [[Int]], targets: [[Int]]) {
        let length = min(contextLength, tokens.count - 1)
        guard length > 0 else { return ([[0]], [[0]]) }
        var inputs: [[Int]] = []
        var targets: [[Int]] = []
        inputs.reserveCapacity(count)
        targets.reserveCapacity(count)
        for _ in 0..<count {
            let start = random.nextInt(tokens.count - length)
            inputs.append(Array(tokens[start..<(start + length)]))
            targets.append(Array(tokens[(start + 1)...(start + length)]))
        }
        return (inputs, targets)
    }
}
