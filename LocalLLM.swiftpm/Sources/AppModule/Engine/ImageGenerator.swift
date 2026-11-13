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

    private static func loadModel(_ url: URL, computeUnits: MLComputeUnits) async throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        return try await MLModel.load(contentsOf: url, configuration: configuration)
    }

    private static func run(files: ImageModelFiles, request: ImageRequest, computeUnits: MLComputeUnits,
                            report: @escaping @Sendable (ImageEvent) -> Void) async throws -> CGImage {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { throw ImageGenerationError.emptyPrompt }

        // 1. プロンプト → 特徴
        report(.status(L("プロンプトを読み込んでいます…", "Encoding the prompt…")))
        let tokenizer = try CLIPTextTokenizer(vocabJSON: try Data(contentsOf: files.vocabulary),
                                              mergesText: try String(contentsOf: files.merges, encoding: .utf8))
        let embeddings = try await encodePrompts(files: files, tokenizer: tokenizer, prompt: prompt,
                                                 negative: request.negativePrompt, computeUnits: computeUnits)
        try Task.checkCancellation()

        // 2. ノイズ除去
        report(.status(L("画像モデルを読み込んでいます…（初回は数分かかることがあります）",
                         "Loading the image model… (the first time can take a few minutes)")))
        let latents = try await denoise(files: files, embeddings: embeddings, request: request,
                                        computeUnits: computeUnits, report: report)
        try Task.checkCancellation()

        // 3. 潜在表現 → 画像
        report(.status(L("画像に仕上げています…", "Decoding the image…")))
        return try await decode(files: files, latents: latents, computeUnits: computeUnits)
    }

    /// テキストエンコーダの出力（[否定, プロンプト] の順。それぞれ 長さ × 次元）
    private struct PromptEmbeddings {
        var values: [Float]
        var length: Int
        var dimension: Int
    }

    private static func encodePrompts(files: ImageModelFiles, tokenizer: CLIPTextTokenizer, prompt: String, negative: String,
                                      computeUnits: MLComputeUnits) async throws -> PromptEmbeddings {
        let model = try await loadModel(files.textEncoder, computeUnits: computeUnits)
        guard let input = model.modelDescription.inputDescriptionsByName["input_ids"]?.multiArrayConstraint else {
            throw ImageGenerationError.missingInput("input_ids")
        }
        let length = input.shape.last?.intValue ?? CLIPTextTokenizer.sequenceLength
        var combined: [Float] = []
        var dimension = 0
        // 埋め草は、まず「終わりの印」で試す（Stable Diffusion 2 系は 0 で埋めるので、次元を見て決め直す）
        for (index, text) in [negative, prompt].enumerated() {
            var hidden = try await encode(text, tokenizer: tokenizer, padID: nil, length: length, dataType: input.dataType, model: model)
            if hidden.dimension == 1024 {
                hidden = try await encode(text, tokenizer: tokenizer, padID: 0, length: length, dataType: input.dataType, model: model)
            }
            if index == 0 { dimension = hidden.dimension }
            combined.append(contentsOf: hidden.values)
        }
        return PromptEmbeddings(values: combined, length: length, dimension: dimension)
    }

    private static func encode(_ text: String, tokenizer: CLIPTextTokenizer, padID: Int?, length: Int,
                               dataType: MLMultiArrayDataType, model: MLModel) async throws -> PromptEmbeddings {
        let ids = tokenizer.encode(text, length: length, padID: padID).map { Float($0) }
        let array = try ImageTensors.array(shape: [1, length], dataType: dataType, values: ids)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: array)])
        return try await InferenceQueue.run {
            let output = try model.prediction(from: provider, options: MLPredictionOptions())
            guard let hidden = output.featureValue(for: "last_hidden_state")?.multiArrayValue else {
                throw ImageGenerationError.missingOutput("last_hidden_state")
            }
            let shape = hidden.shape.map(\.intValue)
            guard shape.count == 3, shape[1] == length else {
                throw ImageGenerationError.unexpectedShape("last_hidden_state", shape)
            }
            return PromptEmbeddings(values: ImageTensors.floats(hidden), length: length, dimension: shape[2])
        }
    }

    /// 潜在表現（1 × チャンネル × 高さ × 幅）
    private struct Latents {
        var values: [Float]
        var shape: [Int]
    }

    private static func denoise(files: ImageModelFiles, embeddings: PromptEmbeddings, request: ImageRequest,
                                computeUnits: MLComputeUnits, report: @escaping @Sendable (ImageEvent) -> Void) async throws -> Latents {
        let model = try await loadModel(files.unet, computeUnits: computeUnits)
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
        let batch = sampleShape[0]
        let latentShape = [1] + Array(sampleShape[1...])
        let latentCount = latentShape.reduce(1, *)

        // テキストの特徴を [バッチ, 次元, 1, 長さ] に並べ替える（モデルが求める並び）
        let hiddenShape = hidden.shape.map(\.intValue)
        guard hiddenShape.count == 4, hiddenShape[1] == embeddings.dimension, hiddenShape[3] == embeddings.length else {
            throw ImageGenerationError.unexpectedShape("encoder_hidden_states", hiddenShape)
        }
        let perPrompt = embeddings.length * embeddings.dimension
        func arranged(_ promptIndex: Int) -> [Float] {
            var result = [Float](repeating: 0, count: perPrompt)
            let base = promptIndex * perPrompt
            for position in 0..<embeddings.length {
                for channel in 0..<embeddings.dimension {
                    result[channel * embeddings.length + position] = embeddings.values[base + position * embeddings.dimension + channel]
                }
            }
            return result
        }
        let negative = arranged(0), positive = arranged(1)
        let hiddenBoth = try ImageTensors.array(shape: [2, embeddings.dimension, 1, embeddings.length], dataType: hidden.dataType,
                                                values: negative + positive)
        let hiddenNegative = try ImageTensors.array(shape: [1, embeddings.dimension, 1, embeddings.length], dataType: hidden.dataType,
                                                    values: negative)
        let hiddenPositive = try ImageTensors.array(shape: [1, embeddings.dimension, 1, embeddings.length], dataType: hidden.dataType,
                                                    values: positive)

        var scheduler = DPMSolverScheduler(stepCount: request.stepCount)
        var noise = DiffusionNoise(seed: request.seed)
        var latents = noise.normals(latentCount).map { $0 * scheduler.initialNoiseScale }
        let total = scheduler.timesteps.count
        report(.step(0, total))

        func predict(_ latentValues: [Float], _ timestep: Int, _ hiddenStates: MLMultiArray, copies: Int) async throws -> [Float] {
            var values = latentValues
            if copies == 2 { values.append(contentsOf: latentValues) }
            let sampleArray = try ImageTensors.array(shape: [copies] + Array(latentShape[1...]), dataType: sample.dataType, values: values)
            let timeArray = try ImageTensors.array(shape: [copies], dataType: time.dataType,
                                                   values: [Float](repeating: Float(timestep), count: copies))
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "sample": MLFeatureValue(multiArray: sampleArray),
                "timestep": MLFeatureValue(multiArray: timeArray),
                "encoder_hidden_states": MLFeatureValue(multiArray: hiddenStates),
            ])
            return try await InferenceQueue.run {
                let output = try model.prediction(from: provider, options: MLPredictionOptions())
                guard let prediction = output.featureValue(for: "noise_pred")?.multiArrayValue else {
                    throw ImageGenerationError.missingOutput("noise_pred")
                }
                return ImageTensors.floats(prediction)
            }
        }

        for (index, timestep) in scheduler.timesteps.enumerated() {
            try Task.checkCancellation()
            let unconditional: [Float], conditional: [Float]
            if batch == 2 {
                // 否定とプロンプトを 1 回でまとめて計算する
                let both = try await predict(latents, timestep, hiddenBoth, copies: 2)
                guard both.count == latentCount * 2 else { throw ImageGenerationError.unexpectedShape("noise_pred", [both.count]) }
                unconditional = Array(both[0..<latentCount])
                conditional = Array(both[latentCount...])
            } else {
                unconditional = try await predict(latents, timestep, hiddenNegative, copies: 1)
                conditional = try await predict(latents, timestep, hiddenPositive, copies: 1)
            }
            let guided = DiffusionGuidance.combine(unconditional: unconditional, conditional: conditional, scale: request.guidanceScale)
            latents = scheduler.step(output: guided, sample: latents)
            report(.step(index + 1, total))
        }
        return Latents(values: latents, shape: latentShape)
    }

    /// 潜在表現の倍率（Stable Diffusion 1.x / 2.x の VAE）
    private static let latentScale: Float = 0.18215

    private static func decode(files: ImageModelFiles, latents: Latents, computeUnits: MLComputeUnits) async throws -> CGImage {
        let model = try await loadModel(files.vaeDecoder, computeUnits: computeUnits)
        guard let input = model.modelDescription.inputDescriptionsByName["z"]?.multiArrayConstraint else {
            throw ImageGenerationError.missingInput("z")
        }
        let scaled = latents.values.map { $0 / latentScale }
        let array = try ImageTensors.array(shape: latents.shape, dataType: input.dataType, values: scaled)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["z": MLFeatureValue(multiArray: array)])
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
