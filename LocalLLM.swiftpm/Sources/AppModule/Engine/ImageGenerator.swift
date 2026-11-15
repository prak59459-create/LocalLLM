import CoreGraphics
import CoreML
import Foundation

/// 画像生成モデル（Core ML 版の Stable Diffusion）のファイル。
struct ImageModelFiles: Sendable {
    var textEncoder: URL
    var unet: URL
    var vaeDecoder: URL
    var vocabulary: URL
    var merges: URL
}

/// 画像生成の依頼。
struct ImageRequest: Sendable {
    var prompt: String
    /// 描いてほしくないもの
    var negativePrompt: String = ""
    var stepCount: Int = 20
    /// プロンプトに従わせる強さ
    var guidanceScale: Float = 7.5
    var seed: UInt64 = 0
}

/// 画像生成の途中経過。
enum ImageEvent: @unchecked Sendable {
    case status(String)
    /// ノイズ除去の進み具合（済んだ回数, 全体）
    case step(Int, Int)
    case finished(CGImage)
}

enum ImageGenerationError: Error, LocalizedError {
    case missingInput(String)
    case missingOutput(String)
    case unexpectedShape(String, [Int])
    case emptyPrompt

    var errorDescription: String? {
        switch self {
        case .missingInput(let name):
            return L("画像モデルの入力「\(name)」が見つかりません（対応していない形式のモデルです）。",
                     "The image model has no input \"\(name)\" (unsupported model format).")
        case .missingOutput(let name):
            return L("画像モデルの出力「\(name)」が見つかりません。", "The image model has no output \"\(name)\".")
        case .unexpectedShape(let name, let shape):
            return L("画像モデルの「\(name)」の形が想定外です: \(shape)", "Unexpected shape for \"\(name)\": \(shape)")
        case .emptyPrompt:
            return L("プロンプトを入力してください。", "Enter a prompt.")
        }
    }
}

/// Core ML 版の Stable Diffusion で画像を作る。
///
/// メモリの少ない iPad でも動くように、3 つのモデル（テキストエンコーダ・UNet・VAE デコーダ）を
/// 同時には読み込まず、使うものだけを読み込んで、使い終わったらすぐ手放す。
/// 1. プロンプトを CLIP のトークンにして、テキストエンコーダで特徴に変える（プロンプトと、否定プロンプトの 2 つ）
/// 2. 乱数のノイズから始めて、UNet でノイズを少しずつ取り除く（DPM-Solver++）
/// 3. VAE デコーダで、できあがった潜在表現を画像にする
actor ImageGenerator {
    private var currentTask: Task<Void, Never>?

    /// 画像を作る。返ってくるストリームを途中でやめると、生成も止まる。
    nonisolated func generate(files: ImageModelFiles, request: ImageRequest,
                              computeUnits: MLComputeUnits) -> AsyncThrowingStream<ImageEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await self.start(files: files, request: request, computeUnits: computeUnits, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func start(files: ImageModelFiles, request: ImageRequest, computeUnits: MLComputeUnits,
                       continuation: AsyncThrowingStream<ImageEvent, Error>.Continuation) async {
        let previous = currentTask
        previous?.cancel()
        let task = Task {
            await previous?.value
            do {
                let image = try await Self.run(files: files, request: request, computeUnits: computeUnits) { event in
                    continuation.yield(event)
                }
                continuation.yield(.finished(image))
                continuation.finish()
            } catch {
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

    // MARK: - 生成

    private static func run(files: ImageModelFiles, request: ImageRequest, computeUnits: MLComputeUnits,
                            report: @escaping @Sendable (ImageEvent) -> Void) async throws -> CGImage {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { throw ImageGenerationError.emptyPrompt }

        // 1. プロンプト → 特徴
        report(.status(L("プロンプトを読み込んでいます…", "Encoding the prompt…")))
        let embeddings = try await DiffusionText.encode(files: files, prompts: [request.negativePrompt, prompt], computeUnits: computeUnits)
        try Task.checkCancellation()

        // 2. ノイズ除去
        report(.status(L("画像モデルを読み込んでいます…（初回は数分かかることがあります）",
                         "Loading the image model… (the first time can take a few minutes)")))
        let unet = try await DiffusionUNet(url: files.unet, computeUnits: computeUnits)
        var scheduler = DPMSolverScheduler(stepCount: request.stepCount)
        var noise = DiffusionNoise(seed: request.seed)
        let start = noise.normals(unet.latentCount).map { $0 * scheduler.initialNoiseScale }
        let hidden = try unet.hiddenStates(negative: embeddings.arranged(0), positive: embeddings.arranged(1))
        report(.step(0, scheduler.timesteps.count))
        let latents = try await unet.denoise(start, scheduler: &scheduler, hidden: hidden, guidance: request.guidanceScale) { done, total in
            report(.step(done, total))
        }
        try Task.checkCancellation()

        // 3. 潜在表現 → 画像
        report(.status(L("画像に仕上げています…", "Decoding the image…")))
        let decoder = try await DiffusionDecoder(url: files.vaeDecoder, computeUnits: computeUnits)
        return try await decoder.decode(latents, shape: unet.latentShape)
    }
}

/// 画像モデルを読み込む
enum DiffusionModels {
    static func load(_ url: URL, computeUnits: MLComputeUnits) async throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        return try await MLModel.load(contentsOf: url, configuration: configuration)
    }
}

/// テキストエンコーダの出力（プロンプトごとに 長さ × 次元）
struct DiffusionEmbeddings: Sendable {
    var values: [Float]
    var length: Int
    var dimension: Int

    var count: Int { values.count / max(length * dimension, 1) }

    /// index 番目のプロンプトの特徴を、UNet が求める [次元][長さ] の並びにする
    func arranged(_ index: Int) -> [Float] {
        let perPrompt = length * dimension
        var result = [Float](repeating: 0, count: perPrompt)
        let base = index * perPrompt
        for position in 0..<length {
            for channel in 0..<dimension {
                result[channel * length + position] = values[base + position * dimension + channel]
            }
        }
        return result
    }
}

/// プロンプトを、CLIP のテキストエンコーダで特徴に変える（使い終わったらモデルを手放す）
enum DiffusionText {
    static func encode(files: ImageModelFiles, prompts: [String], computeUnits: MLComputeUnits) async throws -> DiffusionEmbeddings {
        let tokenizer = try CLIPTextTokenizer(vocabJSON: try Data(contentsOf: files.vocabulary),
                                              mergesText: try String(contentsOf: files.merges, encoding: .utf8))
        let model = try await DiffusionModels.load(files.textEncoder, computeUnits: computeUnits)
        guard let input = model.modelDescription.inputDescriptionsByName["input_ids"]?.multiArrayConstraint else {
            throw ImageGenerationError.missingInput("input_ids")
        }
        let length = input.shape.last?.intValue ?? CLIPTextTokenizer.sequenceLength
        var combined: [Float] = []
        var dimension = 0
        // 埋め草は、まず「終わりの印」で試す（Stable Diffusion 2 系は 0 で埋めるので、次元を見て決め直す）
        for (index, text) in prompts.enumerated() {
            var hidden = try await encode(text, tokenizer: tokenizer, padID: nil, length: length, dataType: input.dataType, model: model)
            if hidden.dimension == 1024 {
                hidden = try await encode(text, tokenizer: tokenizer, padID: 0, length: length, dataType: input.dataType, model: model)
            }
            if index == 0 { dimension = hidden.dimension }
            combined.append(contentsOf: hidden.values)
        }
        return DiffusionEmbeddings(values: combined, length: length, dimension: dimension)
    }

    private static func encode(_ text: String, tokenizer: CLIPTextTokenizer, padID: Int?, length: Int,
                               dataType: MLMultiArrayDataType, model: MLModel) async throws -> DiffusionEmbeddings {
        let ids = tokenizer.encode(text, length: length, padID: padID).map { Float($0) }
        let array = try ImageTensors.array(shape: [1, length], dataType: dataType, values: ids)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: array)])
        let box = UncheckedSendable(model)
        return try await InferenceQueue.run {
            let output = try box.value.prediction(from: provider, options: MLPredictionOptions())
            guard let hidden = output.featureValue(for: "last_hidden_state")?.multiArrayValue else {
                throw ImageGenerationError.missingOutput("last_hidden_state")
            }
            let shape = hidden.shape.map(\.intValue)
            guard shape.count == 3, shape[1] == length else {
                throw ImageGenerationError.unexpectedShape("last_hidden_state", shape)
            }
            return DiffusionEmbeddings(values: ImageTensors.floats(hidden), length: length, dimension: shape[2])
        }
    }
}

/// UNet を読み込んだもの。何回でもノイズ除去ができる（動画では、同じ UNet で何コマも描く）。
final class DiffusionUNet: @unchecked Sendable {
    /// テキストの特徴（UNet の入力の形にしたもの）
    struct HiddenStates: @unchecked Sendable {
        var both: MLMultiArray
        var negative: MLMultiArray
        var positive: MLMultiArray
    }

    private let model: MLModel
    private let sampleType: MLMultiArrayDataType
    private let timeType: MLMultiArrayDataType
    private let hiddenType: MLMultiArrayDataType
    private let hiddenShape: [Int]
    /// 否定とプロンプトを 1 回でまとめて計算できるか（バッチ 2）
    private let batch: Int
    /// 潜在表現の形（1 × チャンネル × 高さ × 幅）
    let latentShape: [Int]
    var latentCount: Int { latentShape.reduce(1, *) }

    init(url: URL, computeUnits: MLComputeUnits) async throws {
        let model = try await DiffusionModels.load(url, computeUnits: computeUnits)
        let inputs = model.modelDescription.inputDescriptionsByName
        guard let sample = inputs["sample"]?.multiArrayConstraint else { throw ImageGenerationError.missingInput("sample") }
        guard let time = inputs["timestep"]?.multiArrayConstraint else { throw ImageGenerationError.missingInput("timestep") }
        guard let hidden = inputs["encoder_hidden_states"]?.multiArrayConstraint else {
            throw ImageGenerationError.missingInput("encoder_hidden_states")
        }
        let sampleShape = sample.shape.map(\.intValue)
        guard sampleShape.count == 4, sampleShape[0] == 1 || sampleShape[0] == 2 else {
            throw ImageGenerationError.unexpectedShape("sample", sampleShape)
        }
        self.model = model
        sampleType = sample.dataType
        timeType = time.dataType
        hiddenType = hidden.dataType
        hiddenShape = hidden.shape.map(\.intValue)
        batch = sampleShape[0]
        latentShape = [1] + Array(sampleShape[1...])
    }

    /// 否定とプロンプトの特徴（[次元][長さ] の並び）を、UNet の入力にする
    func hiddenStates(negative: [Float], positive: [Float]) throws -> HiddenStates {
        guard hiddenShape.count == 4 else { throw ImageGenerationError.unexpectedShape("encoder_hidden_states", hiddenShape) }
        let dimension = hiddenShape[1], length = hiddenShape[3]
        guard negative.count == dimension * length, positive.count == dimension * length else {
            throw ImageGenerationError.unexpectedShape("encoder_hidden_states", hiddenShape)
        }
        return HiddenStates(
            both: try ImageTensors.array(shape: [2, dimension, 1, length], dataType: hiddenType, values: negative + positive),
            negative: try ImageTensors.array(shape: [1, dimension, 1, length], dataType: hiddenType, values: negative),
            positive: try ImageTensors.array(shape: [1, dimension, 1, length], dataType: hiddenType, values: positive))
    }

    /// scheduler の今の段階から最後まで、ノイズを取り除く
    func denoise(_ start: [Float], scheduler: inout DPMSolverScheduler, hidden: HiddenStates, guidance: Float,
                 onStep: (Int, Int) -> Void) async throws -> [Float] {
        var latents = start
        let total = scheduler.timesteps.count
        while !scheduler.isFinished {
            try Task.checkCancellation()
            let timestep = scheduler.timesteps[scheduler.stepIndex]
            let unconditional: [Float], conditional: [Float]
            if batch == 2 {
                // 否定とプロンプトを 1 回でまとめて計算する
                let both = try await predict(latents, timestep, hidden.both, copies: 2)
                guard both.count == latentCount * 2 else { throw ImageGenerationError.unexpectedShape("noise_pred", [both.count]) }
                unconditional = Array(both[0..<latentCount])
                conditional = Array(both[latentCount...])
            } else {
                unconditional = try await predict(latents, timestep, hidden.negative, copies: 1)
                conditional = try await predict(latents, timestep, hidden.positive, copies: 1)
            }
            let guided = DiffusionGuidance.combine(unconditional: unconditional, conditional: conditional, scale: guidance)
            latents = scheduler.step(output: guided, sample: latents)
            onStep(scheduler.stepIndex, total)
        }
        return latents
    }

    private func predict(_ latentValues: [Float], _ timestep: Int, _ hiddenStates: MLMultiArray, copies: Int) async throws -> [Float] {
        var values = latentValues
        if copies == 2 { values.append(contentsOf: latentValues) }
        let sampleArray = try ImageTensors.array(shape: [copies] + Array(latentShape[1...]), dataType: sampleType, values: values)
        let timeArray = try ImageTensors.array(shape: [copies], dataType: timeType,
                                               values: [Float](repeating: Float(timestep), count: copies))
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "sample": MLFeatureValue(multiArray: sampleArray),
            "timestep": MLFeatureValue(multiArray: timeArray),
            "encoder_hidden_states": MLFeatureValue(multiArray: hiddenStates),
        ])
        let model = self.model
        return try await InferenceQueue.run {
            let output = try model.prediction(from: provider, options: MLPredictionOptions())
            guard let prediction = output.featureValue(for: "noise_pred")?.multiArrayValue else {
                throw ImageGenerationError.missingOutput("noise_pred")
            }
            return ImageTensors.floats(prediction)
        }
    }
}

/// VAE デコーダを読み込んだもの（潜在表現 → 画像）
final class DiffusionDecoder: @unchecked Sendable {
    /// 潜在表現の倍率（Stable Diffusion 1.x / 2.x の VAE）
    static let latentScale: Float = 0.18215

    private let model: MLModel
    private let inputType: MLMultiArrayDataType

    init(url: URL, computeUnits: MLComputeUnits) async throws {
        let model = try await DiffusionModels.load(url, computeUnits: computeUnits)
        guard let input = model.modelDescription.inputDescriptionsByName["z"]?.multiArrayConstraint else {
            throw ImageGenerationError.missingInput("z")
        }
        self.model = model
        inputType = input.dataType
    }

    func decode(_ latents: [Float], shape: [Int]) async throws -> CGImage {
        let scaled = latents.map { $0 / Self.latentScale }
        let array = try ImageTensors.array(shape: shape, dataType: inputType, values: scaled)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["z": MLFeatureValue(multiArray: array)])
        let model = self.model
        return try await InferenceQueue.run {
            let output = try model.prediction(from: provider, options: MLPredictionOptions())
            guard let image = output.featureValue(for: "image")?.multiArrayValue else {
                throw ImageGenerationError.missingOutput("image")
            }
            let shape = image.shape.map(\.intValue)
            guard shape.count == 4, shape[1] == 3 else { throw ImageGenerationError.unexpectedShape("image", shape) }
            return try ImageTensors.image(ImageTensors.floats(image), width: shape[3], height: shape[2])
        }
    }
}

/// 画像生成で使う MLMultiArray の読み書き。
enum ImageTensors {
    /// values を、指定の型（Float16 / Float32 / Double / Int32）の配列にする
    static func array(shape: [Int], dataType: MLMultiArrayDataType, values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
        let count = min(values.count, array.count)
        switch dataType {
        case .float16:
            array.withUnsafeMutableBufferPointer(ofType: Float16.self) { pointer, _ in
                for index in 0..<count { pointer[index] = Float16(values[index]) }
            }
        case .float32:
            array.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, _ in
                for index in 0..<count { pointer[index] = values[index] }
            }
        case .double:
            array.withUnsafeMutableBufferPointer(ofType: Double.self) { pointer, _ in
                for index in 0..<count { pointer[index] = Double(values[index]) }
            }
        case .int32:
            array.withUnsafeMutableBufferPointer(ofType: Int32.self) { pointer, _ in
                for index in 0..<count { pointer[index] = Int32(values[index]) }
            }
        default:
            for index in 0..<count { array[index] = NSNumber(value: values[index]) }
        }
        return array
    }

    /// 配列の中身を、並びの順（行優先）の Float にして取り出す
    static func floats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        let shape = array.shape.map(\.intValue)
        // 隙間なく並んでいるか（最後の次元から掛け合わせた幅と strides が同じ）
        var expected = 1
        var contiguous = true
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            if array.strides[axis].intValue != expected { contiguous = false }
            expected *= shape[axis]
        }
        guard contiguous else {
            for index in 0..<count { result[index] = array[index].floatValue }
            return result
        }
        switch array.dataType {
        case .float16:
            array.withUnsafeBufferPointer(ofType: Float16.self) { pointer in
                for index in 0..<min(count, pointer.count) { result[index] = Float(pointer[index]) }
            }
        case .float32:
            array.withUnsafeBufferPointer(ofType: Float.self) { pointer in
                for index in 0..<min(count, pointer.count) { result[index] = pointer[index] }
            }
        case .double:
            array.withUnsafeBufferPointer(ofType: Double.self) { pointer in
                for index in 0..<min(count, pointer.count) { result[index] = Float(pointer[index]) }
            }
        default:
            for index in 0..<count { result[index] = array[index].floatValue }
        }
        return result
    }

    /// [3 × 高さ × 幅]（値は −1〜1）を画像にする
    static func image(_ values: [Float], width: Int, height: Int) throws -> CGImage {
        let plane = width * height
        guard values.count >= plane * 3 else { throw ImageGenerationError.unexpectedShape("image", [values.count]) }
        var pixels = [UInt8](repeating: 255, count: plane * 4)
        for index in 0..<plane {
            for channel in 0..<3 {
                let value = (values[channel * plane + index] + 1) * 127.5
                pixels[index * 4 + channel] = UInt8(max(0, min(255, value.rounded())))
            }
        }
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
            throw ImageGenerationError.unexpectedShape("image", [width, height])
        }
        return image
    }
}
