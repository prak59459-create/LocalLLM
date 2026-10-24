import CoreML
import Foundation

enum EngineError: Error, LocalizedError {
    case unsupportedModel(String)
    case emptyPrompt
    case contextOverflow(Int, Int)
    case missingOutput(String)
    case unexpectedOutputShape([Int])
    case modelNotLoaded
    case lowMemory(available: Int64)
    case stateTooLarge(needed: Int64, available: Int64)

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
        case .lowMemory(let available):
            return L("メモリが残り \(ByteFormat.string(available)) まで減ったので、アプリが落ちないよう途中で止めました。もう一度送ると、古いやり取りを短くして続けます。",
                     "Stopped early because only \(ByteFormat.string(available)) of memory was left, to keep the app from crashing. Send again to continue with a shorter history.")
        case .stateTooLarge(let needed, let available):
            return L("このコンテキスト長では KV キャッシュに \(ByteFormat.string(needed)) 必要ですが、空きが \(ByteFormat.string(available)) しかありません。モデルの詳細でコンテキスト長を短くしてください。",
                     "The KV cache needs \(ByteFormat.string(needed)) at this context length but only \(ByteFormat.string(available)) is free. Shorten the context length in the model details.")
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
final class CoreMLLanguageModel: LanguageModelRunner {
    let model: MLModel
    let interface: ModelInterface
    let maxContext: Int
    /// 1 回の予測で入力できるトークン数の上限（input_ids と causal_mask の形状から決まる）
    let maxQueryLength: Int
    let computeUnits: MLComputeUnits

    /// state（または直近の入力）に入っているトークン列。
    private(set) var cachedTokens: [Int] = []
    private var state: MLState?
    /// ステートレス形式で attention_mask が要求する型。
    private var attentionMaskDataType: MLMultiArrayDataType = .int32
    let logitsElementBytes: Int
    var gate: MemoryGate?

    var isStateless: Bool {
        if case .statelessCausalLM = interface { return true }
        return false
    }

    var needsNewState: Bool { state == nil && !isStateless }

    init(model: MLModel, computeUnits: MLComputeUnits) throws {
        self.model = model
        self.computeUnits = computeUnits
        let specs = model.modelDescription.tensorSpecs
        let interface = ModelInterface.classify(inputs: specs.inputs, outputs: specs.outputs, states: specs.states)
        switch interface {
        case .unsupported(let reason):
            throw EngineError.unsupportedModel(reason)
        case .statefulCausalLM(let idsName, let maskName, _, _, let maxContext):
            self.maxContext = maxContext
            let idsLimit = specs.inputs.first { $0.name == idsName }?.upperBound(axis: 1) ?? 1
            let maskLimit = specs.inputs.first { $0.name == maskName }?.upperBound(axis: 2) ?? idsLimit
            self.maxQueryLength = max(1, min(idsLimit, maskLimit, maxContext))
        case .statelessCausalLM(_, let maskName, _, _, let maxContext):
            self.maxContext = maxContext
            self.maxQueryLength = maxContext
            if let maskName,
               let constraint = model.modelDescription.inputDescriptionsByName[maskName]?.multiArrayConstraint {
                attentionMaskDataType = constraint.dataType
            }
        }
        self.interface = interface
        var logitsName = ""
        switch interface {
        case .statefulCausalLM(_, _, let name, _, _), .statelessCausalLM(_, _, let name, _, _): logitsName = name
        case .unsupported: break
        }
        let logitsType = model.modelDescription.outputDescriptionsByName[logitsName]?.multiArrayConstraint?.dataType
        logitsElementBytes = logitsType?.byteCount ?? 4
    }

    /// モデルファイル（.mlmodelc）を読み込む。
    static func load(compiledModelURL: URL, computeUnits: MLComputeUnits) async throws -> CoreMLLanguageModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: compiledModelURL, configuration: configuration)
        return try CoreMLLanguageModel(model: model, computeUnits: computeUnits)
    }

    var interfaceDescription: String {
        switch interface {
        case .statefulCausalLM(_, _, _, _, let maxContext):
            return L("ステートフル（KV キャッシュあり）・最大 \(maxContext) トークン",
                     "Stateful (KV cache) · up to \(maxContext) tokens")
        case .statelessCausalLM(_, _, _, let fixedLength, let maxContext):
            if let fixedLength {
                return L("ステートレス（固定長 \(fixedLength)）", "Stateless (fixed length \(fixedLength))")
            }
            return L("ステートレス・最大 \(maxContext) トークン", "Stateless · up to \(maxContext) tokens")
        case .unsupported(let reason):
            return L("非対応: \(reason)", "Unsupported: \(reason)")
        }
    }

    var computeDescription: String { computeUnits.label }

    /// キャッシュを捨てる（次の呼び出しは最初から計算する）。
    func resetCache() {
        cachedTokens = []
        state = nil
    }

    /// モデル内部の系列を `tokens` にそろえ、最後のトークンの次に来る logits を返す。
    /// - Parameters:
    ///   - chunkSize: プロンプトを流し込むときの 1 回あたりのトークン数（maxQueryLength まで）
    ///   - progress: (処理済みトークン数, 全体) を報告する。
    /// - Returns: logits と、キャッシュを再利用できたトークン数。
    func evaluate(tokens: [Int], chunkSize: Int = 128,
                  progress: ((Int, Int) -> Void)? = nil) async throws -> (scores: NextTokenScores, reusedTokens: Int) {
        guard !tokens.isEmpty else { throw EngineError.emptyPrompt }
        guard tokens.count <= maxContext else { throw EngineError.contextOverflow(tokens.count, maxContext) }

        switch interface {
        case .statefulCausalLM(let idsName, let maskName, let logitsName, let maskType, _):
            let chunk = min(max(chunkSize, 1), maxQueryLength)
            let result = try await evaluateStateful(tokens: tokens, chunkSize: chunk, idsName: idsName, maskName: maskName,
                                                    logitsName: logitsName, maskType: maskType, progress: progress)
            return (.logits(result.logits), result.reusedTokens)
        case .statelessCausalLM(let idsName, let maskName, let logitsName, let fixedLength, _):
            let logits = try await evaluateStateless(tokens: tokens, idsName: idsName, maskName: maskName,
                                                     logitsName: logitsName, fixedLength: fixedLength)
            progress?(tokens.count, tokens.count)
            cachedTokens = tokens
            return (.logits(logits), 0)
        case .unsupported(let reason):
            throw EngineError.unsupportedModel(reason)
        }
    }

    /// 1 トークン追加して次の logits を返す。
    func append(token: Int) async throws -> NextTokenScores {
        try await evaluate(tokens: cachedTokens + [token], chunkSize: 1, progress: nil).scores
    }

    // MARK: - ステートフル

    private func evaluateStateful(tokens: [Int], chunkSize: Int, idsName: String, maskName: String, logitsName: String,
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
            // 1 回ごとにメモリを確かめ、少なければ小さく区切って読む
            try gate?.check()
            let size = gate?.chunk(chunkSize) ?? chunkSize
            let end = min(position + size, tokens.count)
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
        let ids = try CoreMLTensors.int32Array(chunk, shape: [1, n])
        let mask = try CoreMLTensors.causalMask(rows: n, columns: pastLength + n, firstRowPosition: pastLength,
                                                dataType: maskType)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            idsName: MLFeatureValue(multiArray: ids),
            maskName: MLFeatureValue(multiArray: mask),
        ])
        let output = try await model.prediction(from: input, using: state, options: MLPredictionOptions())
        guard let logits = output.featureValue(for: logitsName)?.multiArrayValue else {
            throw EngineError.missingOutput(logitsName)
        }
        return try row(of: logits, position: max(CoreMLTensors.positions(of: logits) - 1, 0))
    }

    // MARK: - ステートレス

    private func evaluateStateless(tokens: [Int], idsName: String, maskName: String?, logitsName: String,
                                   fixedLength: Int?) async throws -> [Float] {
        let length = fixedLength ?? tokens.count
        guard tokens.count <= length else { throw EngineError.contextOverflow(tokens.count, length) }
        try gate?.check()
        // 右側をパディングする（因果モデルなので、後ろのパディングは前の位置の logits に影響しない）。
        var padded = tokens
        padded.append(contentsOf: repeatElement(0, count: length - tokens.count))
        var features: [String: MLFeatureValue] = [
            idsName: MLFeatureValue(multiArray: try CoreMLTensors.int32Array(padded, shape: [1, length])),
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
        return try row(of: logits, position: tokens.count - 1)
    }

    // MARK: - 補助

    /// logits から指定位置の 1 行（語彙数ぶん）を Float 配列として取り出す。
    private func row(of logits: MLMultiArray, position: Int) throws -> [Float] {
        let layout = try CoreMLTensors.rowLayout(of: logits, position: position)
        var result = [Float](repeating: 0, count: layout.count)
        CoreMLTensors.copyRow(of: logits, elementOffset: layout.offset, count: layout.count, stride: layout.stride,
                              into: &result, at: 0)
        return result
    }

    private func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let limit = min(a.count, b.count)
        var index = 0
        while index < limit && a[index] == b[index] { index += 1 }
        return index
    }
}
