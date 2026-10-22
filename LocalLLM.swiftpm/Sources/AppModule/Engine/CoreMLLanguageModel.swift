import CoreML
import Foundation

enum EngineError: Error, LocalizedError {
    case unsupportedModel(String)
    case emptyPrompt
    case contextOverflow(Int, Int)
    case missingOutput(String)
    case unexpectedOutputShape([Int])
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let reason):
            return L("このモデルの形式には対応していません: \(reason)",
                     "This model format is not supported: \(reason)")
        case .emptyPrompt:
            return L("プロンプトが空です。", "The prompt is empty.")
        case .contextOverflow(let count, let limit):
            return L("プロンプトが長すぎます（\(count) トークン / 上限 \(limit)）。",
                     "The prompt is too long (\(count) tokens / limit \(limit)).")
        case .missingOutput(let name):
            return L("モデルの出力「\(name)」が見つかりません。", "Model output \"\(name)\" was not found.")
        case .unexpectedOutputShape(let shape):
            return L("logits の形状が想定外です: \(shape)", "Unexpected logits shape: \(shape)")
        case .modelNotLoaded:
            return L("モデルが読み込まれていません。", "No model is loaded.")
        }
    }
}

/// Core ML の言語モデルを 1 つ包み、「このトークン列の次の logits をください」という操作だけを提供する。
///
/// ステートフル形式（KV キャッシュを MLState に持つ）の場合:
/// - state に入っているトークン列（cachedTokens）を覚えておき、新しい入力と共通の先頭部分は計算し直さない。
/// - このモデル形式では「causal_mask の最後の次元 − 入力の長さ」がキャッシュ済みの長さとして扱われ、
///   その位置から先が上書きされる。なので共通部分の長さまで「巻き戻す」のは数値を変えるだけでよい。
/// - 長いプロンプトはチャンクに分けて流し込む（logits が [1, n, 語彙数] と大きくなるのを防ぐ）。
///
/// ステートレス形式の場合は、毎回トークン列全体を入力する（遅いが単純）。
///
/// このクラスはスレッドセーフではない。InferenceEngine（actor）の中からだけ使う。
final class CoreMLLanguageModel {
    let model: MLModel
    let interface: ModelInterface
    let maxContext: Int
    /// プロンプトを流し込むときの 1 回あたりのトークン数。
    var prefillChunkSize: Int = 128

    /// state（または直近の入力）に入っているトークン列。
    private(set) var cachedTokens: [Int] = []
    private var state: MLState?
    /// ステートレス形式で attention_mask が要求する型。
    private var attentionMaskDataType: MLMultiArrayDataType = .int32

    init(model: MLModel) throws {
        self.model = model
        let specs = model.modelDescription.tensorSpecs
        let interface = ModelInterface.classify(inputs: specs.inputs, outputs: specs.outputs, states: specs.states)
        switch interface {
        case .unsupported(let reason):
            throw EngineError.unsupportedModel(reason)
        case .statefulCausalLM(_, _, _, _, let maxContext):
            self.maxContext = maxContext
        case .statelessCausalLM(_, let maskName, _, _, let maxContext):
            self.maxContext = maxContext
            if let maskName,
               let constraint = model.modelDescription.inputDescriptionsByName[maskName]?.multiArrayConstraint {
                attentionMaskDataType = constraint.dataType
            }
        }
        self.interface = interface
    }

    /// モデルファイル（.mlmodelc）を読み込む。
    static func load(compiledModelURL: URL, computeUnits: MLComputeUnits) async throws -> CoreMLLanguageModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: compiledModelURL, configuration: configuration)
        return try CoreMLLanguageModel(model: model)
    }

    /// キャッシュを捨てる（次の呼び出しは最初から計算する）。
    func resetCache() {
        cachedTokens = []
        state = nil
    }

    /// モデル内部の系列を `tokens` にそろえ、最後のトークンの次に来る logits を返す。
    /// - Parameter progress: (処理済みトークン数, 全体) を報告する。
    /// - Returns: logits と、キャッシュを再利用できたトークン数。
    func evaluate(tokens: [Int], progress: ((Int, Int) -> Void)? = nil) async throws -> (logits: [Float], reusedTokens: Int) {
        guard !tokens.isEmpty else { throw EngineError.emptyPrompt }
        guard tokens.count <= maxContext else { throw EngineError.contextOverflow(tokens.count, maxContext) }

        switch interface {
        case .statefulCausalLM(let idsName, let maskName, let logitsName, let maskType, _):
            return try await evaluateStateful(tokens: tokens, idsName: idsName, maskName: maskName,
                                              logitsName: logitsName, maskType: maskType, progress: progress)
        case .statelessCausalLM(let idsName, let maskName, let logitsName, let fixedLength, _):
            let logits = try await evaluateStateless(tokens: tokens, idsName: idsName, maskName: maskName,
                                                     logitsName: logitsName, fixedLength: fixedLength)
            progress?(tokens.count, tokens.count)
            cachedTokens = tokens
            return (logits, 0)
        case .unsupported(let reason):
            throw EngineError.unsupportedModel(reason)
        }
    }

    /// 1 トークン追加して次の logits を返す。
    func append(token: Int) async throws -> [Float] {
        try await evaluate(tokens: cachedTokens + [token]).logits
    }

    // MARK: - ステートフル

    private func evaluateStateful(tokens: [Int], idsName: String, maskName: String, logitsName: String,
                                  maskType: TensorSpec.DataType,
                                  progress: ((Int, Int) -> Void)?) async throws -> (logits: [Float], reusedTokens: Int) {
        if state == nil {
            state = model.makeState()
            cachedTokens = []
        }
        guard let state else { throw EngineError.modelNotLoaded }

        // 共通の先頭部分は再計算しない。ただし logits を得るため最低 1 トークンは入力する。
        var reused = commonPrefixLength(cachedTokens, tokens)
        if reused >= tokens.count { reused = tokens.count - 1 }
        // ここから先の位置は上書きされるので、キャッシュの記録も先に縮めておく（途中で失敗しても整合性を保つ）。
        cachedTokens = Array(tokens[0..<reused])

        var position = reused
        var lastLogits: [Float] = []
        progress?(position, tokens.count)
        while position < tokens.count {
            try Task.checkCancellation()
            let end = min(position + max(prefillChunkSize, 1), tokens.count)
            let chunk = Array(tokens[position..<end])
            lastLogits = try await predictStateful(chunk: chunk, pastLength: position, state: state,
                                                   idsName: idsName, maskName: maskName,
                                                   logitsName: logitsName, maskType: maskType)
            cachedTokens.append(contentsOf: chunk)
            position = end
            progress?(position, tokens.count)
        }
        return (lastLogits, reused)
    }

    private func predictStateful(chunk: [Int], pastLength: Int, state: MLState, idsName: String, maskName: String,
                                 logitsName: String, maskType: TensorSpec.DataType) async throws -> [Float] {
        let n = chunk.count
        let total = pastLength + n
        let ids = try makeInt32Array(chunk, shape: [1, n])
        let mask = try makeCausalMask(queryLength: n, totalLength: total, pastLength: pastLength, dataType: maskType)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            idsName: MLFeatureValue(multiArray: ids),
            maskName: MLFeatureValue(multiArray: mask),
        ])
        let output = try await model.prediction(from: input, using: state, options: MLPredictionOptions())
        guard let logits = output.featureValue(for: logitsName)?.multiArrayValue else {
            throw EngineError.missingOutput(logitsName)
        }
        return try lastRow(of: logits, sequenceLength: n)
    }

    /// 因果マスク [1, 1, n, past+n]。見てよい位置は 0、見てはいけない位置は大きな負の値。
    private func makeCausalMask(queryLength n: Int, totalLength: Int, pastLength: Int,
                                dataType: TensorSpec.DataType) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: n), NSNumber(value: totalLength)]
        if dataType == .float32 {
            let array = try MLMultiArray(shape: shape, dataType: .float32)
            array.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, strides in
                let rowStride = strides[2], colStride = strides[3]
                for row in 0..<n {
                    let visible = pastLength + row
                    for col in 0..<totalLength {
                        pointer[row * rowStride + col * colStride] = col <= visible ? 0 : -Float.greatestFiniteMagnitude
                    }
                }
            }
            return array
        } else {
            let array = try MLMultiArray(shape: shape, dataType: .float16)
            array.withUnsafeMutableBufferPointer(ofType: Float16.self) { pointer, strides in
                let rowStride = strides[2], colStride = strides[3]
                for row in 0..<n {
                    let visible = pastLength + row
                    for col in 0..<totalLength {
                        pointer[row * rowStride + col * colStride] = col <= visible ? 0 : -Float16.greatestFiniteMagnitude
                    }
                }
            }
            return array
        }
    }

    // MARK: - ステートレス

    private func evaluateStateless(tokens: [Int], idsName: String, maskName: String?, logitsName: String,
                                   fixedLength: Int?) async throws -> [Float] {
        let length = fixedLength ?? tokens.count
        guard tokens.count <= length else { throw EngineError.contextOverflow(tokens.count, length) }
        // 右側をパディングする（因果モデルなので、後ろのパディングは前の位置の logits に影響しない）。
        var padded = tokens
        padded.append(contentsOf: repeatElement(0, count: length - tokens.count))
        var features: [String: MLFeatureValue] = [
            idsName: MLFeatureValue(multiArray: try makeInt32Array(padded, shape: [1, length])),
        ]
        if let maskName {
            let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: attentionMaskDataType)
            for index in 0..<length {
                mask[[0, NSNumber(value: index)]] = NSNumber(value: index < tokens.count ? 1 : 0)
            }
            features[maskName] = MLFeatureValue(multiArray: mask)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await model.prediction(from: input, options: MLPredictionOptions())
        guard let logits = output.featureValue(for: logitsName)?.multiArrayValue else {
            throw EngineError.missingOutput(logitsName)
        }
        return try row(of: logits, position: tokens.count - 1, sequenceLength: length)
    }

    // MARK: - 補助

    private func makeInt32Array(_ values: [Int], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { pointer, strides in
            let stride = strides.last ?? 1
            for (index, value) in values.enumerated() {
                pointer[index * stride] = Int32(truncatingIfNeeded: value)
            }
        }
        return array
    }

    private func lastRow(of logits: MLMultiArray, sequenceLength: Int) throws -> [Float] {
        let shape = logits.shape.map(\.intValue)
        // [1, n, V] / [n, V] / [1, V] / [V] のどれでも、最後の位置を取り出す。
        let positions: Int
        switch shape.count {
        case 3: positions = shape[1]
        case 2: positions = shape[0]
        case 1: positions = 1
        default: throw EngineError.unexpectedOutputShape(shape)
        }
        return try row(of: logits, position: max(positions - 1, 0), sequenceLength: sequenceLength)
    }

    /// logits から指定位置の 1 行（語彙数ぶん）を Float 配列として取り出す。
    private func row(of logits: MLMultiArray, position: Int, sequenceLength: Int) throws -> [Float] {
        let shape = logits.shape.map(\.intValue)
        let strides = logits.strides.map(\.intValue)
        guard let vocab = shape.last, vocab > 0 else { throw EngineError.unexpectedOutputShape(shape) }
        let offset: Int
        switch shape.count {
        case 3:
            guard position < shape[1] else { throw EngineError.unexpectedOutputShape(shape) }
            offset = position * strides[1]
        case 2:
            // [1, V] なら位置は 1 つしかない
            offset = shape[0] == 1 ? 0 : position * strides[0]
        case 1:
            offset = 0
        default:
            throw EngineError.unexpectedOutputShape(shape)
        }
        let vocabStride = strides.last ?? 1
        var result = [Float](repeating: 0, count: vocab)
        switch logits.dataType {
        case .float16:
            logits.withUnsafeBufferPointer(ofType: Float16.self) { pointer in
                for index in 0..<vocab { result[index] = Float(pointer[offset + index * vocabStride]) }
            }
        case .float32:
            logits.withUnsafeBufferPointer(ofType: Float.self) { pointer in
                for index in 0..<vocab { result[index] = pointer[offset + index * vocabStride] }
            }
        case .double:
            logits.withUnsafeBufferPointer(ofType: Double.self) { pointer in
                for index in 0..<vocab { result[index] = Float(pointer[offset + index * vocabStride]) }
            }
        default:
            for index in 0..<vocab { result[index] = logits[offset + index * vocabStride].floatValue }
        }
        return result
    }

    private func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let limit = min(a.count, b.count)
        var index = 0
        while index < limit && a[index] == b[index] { index += 1 }
        return index
    }
}
