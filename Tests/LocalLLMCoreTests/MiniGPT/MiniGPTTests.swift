import Foundation
import XCTest
@testable import LocalLLMCore

/// MiniGPT を PyTorch（自動微分）の結果と比べる。
/// 事前に Tools/fixtures/minigpt_expectations.py を実行しておく（無ければスキップ）。
final class MiniGPTParityTests: XCTestCase {
    struct Reference: Decodable {
        struct Parameter: Decodable {
            var name: String
            var shape: [Int]
            var value: [Float]
        }
        struct Hyper: Decodable {
            var learningRate: Float
            var weightDecay: Float
            var beta1: Float
            var beta2: Float
            var epsilon: Float
            var gradientClip: Float
        }
        var config: MiniGPTConfig
        var hyperparameters: Hyper
        var parameters: [Parameter]
        var inputs: [[Int]]
        var targets: [[Int]]
        var losses: [Float]
        var gradients: [String: [Float]]
        var gradientNorms: [Float]
        var afterSteps: [[String: [Float]]]
        var fullInputs: [[Int]]
        var fullTargets: [[Int]]
        var fullLoss: Float
        var context: [Int]
        var logits: [Float]
        var shortContext: [Int]
        var shortLogits: [Float]
    }

    func loadReference() throws -> Reference {
        let url = TestFixtures.root.appendingPathComponent("expectations/minigpt/reference.json")
        guard let data = try? Data(contentsOf: url) else { throw XCTSkip("MiniGPT expectations not generated") }
        return try JSONDecoder().decode(Reference.self, from: data)
    }

    func makeModel(_ reference: Reference) throws -> MiniGPT {
        let model = MiniGPT(config: reference.config, seed: 0)
        XCTAssertEqual(model.parameters.map(\.name), reference.parameters.map(\.name))
        for entry in reference.parameters {
            let parameter = try XCTUnwrap(model.parameter(named: entry.name))
            XCTAssertEqual(parameter.shape, entry.shape, entry.name)
            parameter.value = entry.value
        }
        return model
    }

    func assertClose(_ actual: [Float], _ expected: [Float], tolerance: Float, _ label: String,
                     file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
        let scale = max(expected.map { abs($0) }.max() ?? 1, 1e-3)
        var worst: Float = 0
        var worstIndex = 0
        for index in 0..<min(actual.count, expected.count) {
            let difference = abs(actual[index] - expected[index])
            if difference > worst {
                worst = difference
                worstIndex = index
            }
        }
        XCTAssertLessThanOrEqual(worst / scale, tolerance,
                                 "\(label): index \(worstIndex) got \(actual[worstIndex]) expected \(expected[worstIndex])",
                                 file: file, line: line)
    }

    func testLossAndLogitsMatchPyTorch() throws {
        let reference = try loadReference()
        let model = try makeModel(reference)
        XCTAssertEqual(model.loss(inputs: reference.inputs, targets: reference.targets), reference.losses[0], accuracy: 1e-5)
        XCTAssertEqual(model.loss(inputs: reference.fullInputs, targets: reference.fullTargets), reference.fullLoss, accuracy: 1e-5)
        assertClose(model.logits(context: reference.context), reference.logits, tolerance: 1e-5, "logits")
        assertClose(model.logits(context: reference.shortContext), reference.shortLogits, tolerance: 1e-5, "short logits")
    }

    func testGradientsMatchPyTorch() throws {
        let reference = try loadReference()
        let model = try makeModel(reference)
        model.zeroGradients()
        let loss = model.lossAndGradients(inputs: reference.inputs, targets: reference.targets)
        XCTAssertEqual(loss, reference.losses[0], accuracy: 1e-5)
        for parameter in model.parameters {
            let expected = try XCTUnwrap(reference.gradients[parameter.name])
            assertClose(parameter.gradient, expected, tolerance: 1e-4, "grad \(parameter.name)")
        }
        // 加算されること
        _ = model.lossAndGradients(inputs: reference.inputs, targets: reference.targets)
        let wte = try XCTUnwrap(model.parameter(named: "wte"))
        assertClose(wte.gradient, reference.gradients["wte"]!.map { $0 * 2 }, tolerance: 1e-4, "accumulated grad")
    }

    func testAdamWStepsMatchPyTorch() throws {
        let reference = try loadReference()
        let model = try makeModel(reference)
        var hyper = TrainingHyperparameters()
        hyper.learningRate = reference.hyperparameters.learningRate
        hyper.weightDecay = reference.hyperparameters.weightDecay
        hyper.beta1 = reference.hyperparameters.beta1
        hyper.beta2 = reference.hyperparameters.beta2
        hyper.epsilon = reference.hyperparameters.epsilon
        hyper.gradientClip = reference.hyperparameters.gradientClip
        var optimizer = AdamWOptimizer(hyperparameters: hyper)
        for step in 0..<2 {
            model.zeroGradients()
            let loss = model.lossAndGradients(inputs: reference.inputs, targets: reference.targets)
            XCTAssertEqual(loss, reference.losses[step], accuracy: 1e-4, "loss at step \(step)")
            let norm = optimizer.step(parameters: model.parameters, learningRate: hyper.learningRate)
            XCTAssertEqual(norm, reference.gradientNorms[step], accuracy: 1e-4, "norm at step \(step)")
            for parameter in model.parameters {
                let expected = try XCTUnwrap(reference.afterSteps[step][parameter.name])
                // 本来の勾配がほぼ 0 の要素（k のバイアスは softmax に効かないので勾配が厳密に 0）は、
                // Adam が丸め誤差を ±lr に拡大するので比べない
                let gradient = try XCTUnwrap(reference.gradients[parameter.name])
                let compared = gradient.indices.filter { abs(gradient[$0]) > 1e-6 }
                assertClose(compared.map { parameter.value[$0] }, compared.map { expected[$0] }, tolerance: 1e-4,
                            "step \(step) \(parameter.name)")
            }
        }
    }
}

final class MiniGPTUnitTests: XCTestCase {
    static let tinyConfig = MiniGPTConfig(vocabSize: 7, contextLength: 6, embeddingSize: 8, layerCount: 2, headCount: 2)

    func testParameterLayoutAndCount() {
        let config = MiniGPTConfig.preset(.small, vocabSize: 100)
        let model = MiniGPT(config: config, seed: 1)
        XCTAssertEqual(model.parameters.reduce(0) { $0 + $1.count }, config.parameterCount)
        XCTAssertEqual(model.parameters.first?.name, "wte")
        XCTAssertEqual(model.parameters.last?.name, "ln_f.bias")
        XCTAssertEqual(model.parameter(named: "h.3.mlp.fc.weight")?.shape, [128, 512])
        XCTAssertEqual(model.parameter(named: "h.0.attn.qkv.weight")?.shape, [128, 384])
        XCTAssertEqual(MiniGPTConfig.preset(.tiny, vocabSize: 5).layerCount, 2)
        XCTAssertEqual(MiniGPTConfig.preset(.medium, vocabSize: 5).embeddingSize, 192)
    }

    func testInitializationStatistics() throws {
        let config = MiniGPTConfig.preset(.small, vocabSize: 300)
        let model = MiniGPT(config: config, seed: 7)
        func std(_ values: [Float]) -> Double {
            let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
            return (values.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / Double(values.count)).squareRoot()
        }
        XCTAssertEqual(std(try XCTUnwrap(model.parameter(named: "wte")).value), 0.02, accuracy: 0.001)
        XCTAssertEqual(std(try XCTUnwrap(model.parameter(named: "h.0.mlp.fc.weight")).value), 0.02, accuracy: 0.001)
        XCTAssertEqual(std(try XCTUnwrap(model.parameter(named: "h.1.attn.proj.weight")).value),
                       0.02 / (2.0 * 4).squareRoot(), accuracy: 0.0005)
        XCTAssertTrue(try XCTUnwrap(model.parameter(named: "h.0.ln1.weight")).value.allSatisfy { $0 == 1 })
        XCTAssertTrue(try XCTUnwrap(model.parameter(named: "h.0.attn.qkv.bias")).value.allSatisfy { $0 == 0 })
        // 同じ seed なら同じ重み、違う seed なら違う重み
        XCTAssertEqual(MiniGPT(config: config, seed: 7).parameters[0].value, model.parameters[0].value)
        XCTAssertNotEqual(MiniGPT(config: config, seed: 8).parameters[0].value, model.parameters[0].value)
    }

    /// PyTorch が無くても確かめられるように、差分近似でも勾配を確かめる
    func testGradientsMatchFiniteDifferences() {
        let model = MiniGPT(config: Self.tinyConfig, seed: 3)
        // 勾配が小さすぎないように重みを大きめにする
        var random = MiniRandom(seed: 99)
        for parameter in model.parameters where parameter.shape.count >= 2 {
            for index in parameter.value.indices { parameter.value[index] = Float(random.nextNormal() * 0.3) }
        }
        let inputs = [[1, 2, 3, 4, 5], [6, 0, 2, 2, 1]]
        let targets = [[2, 3, 4, 5, 6], [0, 2, 2, 1, 3]]
        model.zeroGradients()
        _ = model.lossAndGradients(inputs: inputs, targets: targets)
        let epsilon: Float = 1e-2
        for parameter in model.parameters {
            // 勾配の大きい要素をいくつか選んで調べる
            let indices = parameter.gradient.indices.sorted { abs(parameter.gradient[$0]) > abs(parameter.gradient[$1]) }.prefix(3)
            for index in indices where abs(parameter.gradient[index]) > 1e-3 {
                let original = parameter.value[index]
                parameter.value[index] = original + epsilon
                let plus = model.loss(inputs: inputs, targets: targets)
                parameter.value[index] = original - epsilon
                let minus = model.loss(inputs: inputs, targets: targets)
                parameter.value[index] = original
                let numeric = (plus - minus) / (2 * epsilon)
                let analytic = parameter.gradient[index]
                XCTAssertEqual(numeric, analytic, accuracy: max(abs(analytic) * 0.05, 2e-4), "\(parameter.name)[\(index)]")
            }
        }
    }

    func testLearnsRepeatingPattern() {
        let config = MiniGPTConfig(vocabSize: 5, contextLength: 8, embeddingSize: 16, layerCount: 1, headCount: 2)
        let model = MiniGPT(config: config, seed: 1)
        let tokens = (0..<400).map { [1, 2, 3, 4][$0 % 4] }
        var hyper = TrainingHyperparameters()
        hyper.learningRate = 1e-2
        hyper.minLearningRate = 1e-3
        hyper.batchSize = 8
        hyper.warmupSteps = 10
        hyper.totalSteps = 150
        let trainer = MiniGPTTrainer(model: model, tokens: tokens, hyperparameters: hyper)
        XCTAssertTrue(trainer.hasValidationData)
        let first = trainer.step()
        var last = first
        for _ in 1..<150 { last = trainer.step() }
        XCTAssertEqual(trainer.stepIndex, 150)
        XCTAssertGreaterThan(first, 1.2)
        XCTAssertLessThan(last, 0.1)
        XCTAssertLessThan(trainer.validationLoss(batches: 2), 0.1)
        XCTAssertEqual(trainer.skippedSteps, 0)
        // 覚えたパターンを続けられる
        XCTAssertEqual(model.generate(prompt: [1, 2, 3], maxNewTokens: 6, temperature: 0, topK: 0, seed: 0), [4, 1, 2, 3, 4, 1])
    }

    func testTrainingIsDeterministic() {
        func run() -> ([Float], [Float]) {
            let model = MiniGPT(config: Self.tinyConfig, seed: 5)
            var hyper = TrainingHyperparameters()
            hyper.batchSize = 4
            hyper.seed = 11
            let tokens = (0..<200).map { ($0 * 7 + $0 / 3) % 7 }
            let trainer = MiniGPTTrainer(model: model, tokens: tokens, hyperparameters: hyper)
            let losses = (0..<5).map { _ in trainer.step() }
            return (losses, model.parameters[0].value)
        }
        let a = run(), b = run()
        XCTAssertEqual(a.0, b.0)
        XCTAssertEqual(a.1, b.1)
    }

    func testLearningRateSchedule() {
        var hyper = TrainingHyperparameters()
        hyper.learningRate = 1
        hyper.minLearningRate = 0.1
        hyper.warmupSteps = 10
        hyper.totalSteps = 110
        let trainer = MiniGPTTrainer(model: MiniGPT(config: Self.tinyConfig, seed: 1), tokens: Array(repeating: 1, count: 50),
                                     hyperparameters: hyper)
        XCTAssertEqual(trainer.learningRate(at: 0), 0.1, accuracy: 1e-6)
        XCTAssertEqual(trainer.learningRate(at: 9), 1, accuracy: 1e-6)
        XCTAssertEqual(trainer.learningRate(at: 10), 1, accuracy: 1e-6)
        XCTAssertEqual(trainer.learningRate(at: 60), 0.55, accuracy: 1e-6)
        XCTAssertEqual(trainer.learningRate(at: 110), 0.1, accuracy: 1e-6)
        XCTAssertEqual(trainer.learningRate(at: 500), 0.1, accuracy: 1e-6)
        XCTAssertEqual(trainer.currentLearningRate, trainer.learningRate(at: 0))
    }

    func testValidationSplitNeedsEnoughData() {
        let model = MiniGPT(config: Self.tinyConfig, seed: 1)
        // 10% が 1 窓（文脈長 + 2）に届かないときは検証データなし
        let small = MiniGPTTrainer(model: model, tokens: Array(repeating: 1, count: 40), hyperparameters: TrainingHyperparameters())
        XCTAssertFalse(small.hasValidationData)
        XCTAssertTrue(small.validationLoss(batches: 1).isNaN)
        let large = MiniGPTTrainer(model: model, tokens: Array(repeating: 1, count: 200), hyperparameters: TrainingHyperparameters())
        XCTAssertTrue(large.hasValidationData)
        XCTAssertTrue(large.validationLoss(batches: 1).isFinite)
        // 窓より短いデータでも落ちない
        let tiny = MiniGPTTrainer(model: model, tokens: [1, 2, 3], hyperparameters: TrainingHyperparameters())
        XCTAssertTrue(tiny.step().isFinite)
    }

    func testGenerateReturnsOnlyNewTokensAndIsDeterministic() {
        let model = MiniGPT(config: Self.tinyConfig, seed: 2)
        let prompt = [1, 2, 3, 4, 5, 6, 1, 2]
        let greedy = model.generate(prompt: prompt, maxNewTokens: 5, temperature: 0, topK: 0, seed: 0)
        XCTAssertEqual(greedy.count, 5)
        let logits = model.logits(context: prompt)
        XCTAssertEqual(greedy.first, logits.indices.max { logits[$0] < logits[$1] })
        let sampledA = model.generate(prompt: prompt, maxNewTokens: 20, temperature: 1, topK: 3, seed: 42)
        let sampledB = model.generate(prompt: prompt, maxNewTokens: 20, temperature: 1, topK: 3, seed: 42)
        XCTAssertEqual(sampledA, sampledB)
        XCTAssertTrue(sampledA.allSatisfy { (0..<7).contains($0) })
        XCTAssertEqual(model.generate(prompt: [], maxNewTokens: 3, temperature: 0.8, topK: 0, seed: 1).count, 3)
        XCTAssertEqual(model.generate(prompt: prompt, maxNewTokens: 0, temperature: 0.8, topK: 0, seed: 1), [])
    }

    func testTopKOneIsGreedy() {
        var random = MiniRandom(seed: 1)
        let logits: [Float] = [0.1, 2, -1, 1.9]
        for _ in 0..<20 {
            XCTAssertEqual(MiniGPT.sample(logits, temperature: 1.5, topK: 1, random: &random), 1)
        }
        XCTAssertEqual(MiniGPT.sample(logits, temperature: 0, topK: 0, random: &random), 1)
    }

    func testSaveAndLoadRoundTrip() throws {
        let model = MiniGPT(config: Self.tinyConfig, seed: 9)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("minigpt-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.save(to: url)
        let loaded = try MiniGPT.load(from: url)
        XCTAssertEqual(loaded.config, model.config)
        for (a, b) in zip(loaded.parameters, model.parameters) {
            XCTAssertEqual(a.name, b.name)
            XCTAssertEqual(a.value, b.value)
        }
        XCTAssertEqual(loaded.logits(context: [1, 2, 3]), model.logits(context: [1, 2, 3]))

        // 壊れたファイル
        try Data("hello".utf8).write(to: url)
        XCTAssertThrowsError(try MiniGPT.load(from: url))
        var data = try Data(contentsOf: url)
        data = Data("MINIGPT1".utf8) + Data([255, 255, 0, 0])
        try data.write(to: url)
        XCTAssertThrowsError(try MiniGPT.load(from: url))
    }

    func testInvalidBatchesReturnNaN() {
        let model = MiniGPT(config: Self.tinyConfig, seed: 1)
        XCTAssertTrue(model.loss(inputs: [], targets: []).isNaN)
        XCTAssertTrue(model.loss(inputs: [[1, 2]], targets: [[1]]).isNaN)
        XCTAssertTrue(model.loss(inputs: [Array(repeating: 1, count: 7)], targets: [Array(repeating: 1, count: 7)]).isNaN)
        // 範囲外のトークンは丸める
        XCTAssertTrue(model.loss(inputs: [[100, -3]], targets: [[1, 2]]).isFinite)
    }
}

final class CharVocabularyTests: XCTestCase {
    func testOrderingAndRoundTrip() throws {
        let vocabulary = CharVocabulary(text: "ねこねこ犬a")
        XCTAssertEqual(vocabulary.size, 5)
        // 多い順、同じ回数なら文字コード順（こ < ね、a < 犬）
        XCTAssertEqual(vocabulary.encode("こ"), [1])
        XCTAssertEqual(vocabulary.encode("ね"), [2])
        XCTAssertEqual(vocabulary.encode("a犬"), [3, 4])
        XCTAssertEqual(vocabulary.encode("ねZ"), [2, 0])
        XCTAssertEqual(vocabulary.decode([1, 2, 0, 99]), "こね\u{FFFD}\u{FFFD}")
        let data = try JSONEncoder().encode(vocabulary)
        let decoded = try JSONDecoder().decode(CharVocabulary.self, from: data)
        XCTAssertEqual(decoded, vocabulary)
        XCTAssertEqual(decoded.encode("こねa"), [1, 2, 3])
    }

    func testMaxSizeAndScalars() {
        let vocabulary = CharVocabulary(text: "aaabbc", maxSize: 3)
        XCTAssertEqual(vocabulary.size, 3)
        XCTAssertEqual(vocabulary.encode("abc"), [1, 2, 0])
        // 結合文字もスカラー単位
        let combining = CharVocabulary(text: "e\u{301}")
        XCTAssertEqual(combining.size, 3)
        XCTAssertEqual(combining.decode(combining.encode("e\u{301}")), "e\u{301}")
        XCTAssertEqual(CharVocabulary(text: "").size, 1)
    }
}

final class MiniMathTests: XCTestCase {
    func testGemmAllTransposes() {
        // A: 2x3, B: 3x2（行の長さに余白をつけて lda / ldb を確かめる）
        let a: [[Float]] = [[1, 2, 3], [4, 5, 6]]
        let b: [[Float]] = [[7, 8], [9, 10], [11, 12]]
        let expected: [[Float]] = [[58, 64], [139, 154]]
        func padded(_ matrix: [[Float]], width: Int) -> [Float] {
            matrix.flatMap { $0 + Array(repeating: -99, count: width - $0.count) }
        }
        func transpose(_ m: [[Float]]) -> [[Float]] {
            (0..<m[0].count).map { j in m.map { $0[j] } }
        }
        for transA in [false, true] {
            for transB in [false, true] {
                let storedA = transA ? transpose(a) : a
                let storedB = transB ? transpose(b) : b
                let lda = storedA[0].count + 1, ldb = storedB[0].count + 2
                let pa = padded(storedA, width: lda), pb = padded(storedB, width: ldb)
                var c: [Float] = [1, 1, 0, 1, 1, 0]
                pa.withUnsafeBufferPointer { pa in
                    pb.withUnsafeBufferPointer { pb in
                        c.withUnsafeMutableBufferPointer { pc in
                            MiniMath.gemm(transA: transA, transB: transB, m: 2, n: 2, k: 3, alpha: 2,
                                          a: pa.baseAddress!, lda: lda, b: pb.baseAddress!, ldb: ldb,
                                          beta: 0.5, c: pc.baseAddress!, ldc: 3)
                        }
                    }
                }
                let result = [[c[0], c[1]], [c[3], c[4]]]
                let want = expected.map { $0.map { $0 * 2 + 0.5 } }
                XCTAssertEqual(result, want, "transA \(transA) transB \(transB)")
                XCTAssertEqual(c[2], 0)
            }
        }
    }

    func testVectorHelpers() {
        let a: [Float] = [1, -2, 3, 0.5]
        let b: [Float] = [2, 4, -1, 2]
        var out = [Float](repeating: 0, count: 4)
        a.withUnsafeBufferPointer { a in
            b.withUnsafeBufferPointer { b in
                out.withUnsafeMutableBufferPointer { o in
                    let a = a.baseAddress!, b = b.baseAddress!, o = o.baseAddress!
                    MiniVector.multiplyAdd(a, b, b, o, 4)
                    XCTAssertEqual(Array(UnsafeBufferPointer(start: o, count: 4)), [4, -4, -4, 3])
                    MiniVector.divide(a, b, o, 4)
                    XCTAssertEqual(Array(UnsafeBufferPointer(start: o, count: 4)), [0.5, -0.5, -3, 0.25])
                    MiniVector.scaleShift(a, 2, 1, o, 4)
                    XCTAssertEqual(Array(UnsafeBufferPointer(start: o, count: 4)), [3, -3, 7, 2])
                    XCTAssertEqual(MiniVector.maximum(a, 4), 3)
                    XCTAssertEqual(MiniVector.sum(a, 4), 2.5)
                    XCTAssertEqual(MiniMath.dot(a, b, count: 4), -8)
                }
            }
        }
    }
}
