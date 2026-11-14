import CoreML
import Foundation

/// 分割モデルの部品のファイル（端末上の .mlmodelc）
struct PipelineFiles: Sendable {
    var embeddings: URL?
    var chunks: [URL]
    var lmHead: URL?
    var descriptor: PipelineDescriptor
}

/// 分割された Core ML の LLM（ANEMLL 形式）を順番につないで動かす。
///
/// - 埋め込み: input_ids [1, n] → hidden_states [1, n, H]
/// - 本体のチャンク（infer / prefill 関数。KV キャッシュの state は全チャンクで 1 つを共有）:
///   hidden_states, position_ids, causal_mask [1, 1, n, 文脈長], current_pos → output_hidden_states
/// - 出力層: hidden_states → logits1…logitsN（語彙を N 分割）、または argmax_idx / argmax_val
/// - 一体型（monolithic）は 1 つのモデルが input_ids から logits までを計算する。
///
/// プロンプトは prefill 関数で batchSize（多くは 64）トークンずつまとめて読み込み、最後のトークンだけ infer で計算して
/// 次のトークンのスコアを得る。KV キャッシュの位置は current_pos で明示するので、共通の先頭部分は再計算しない。
final class PipelineLanguageModel: LanguageModelRunner {
    private struct Chunk {
        var infer: MLModel
        var prefill: MLModel?
        var inferRotate: MLModel?
        var prefillRotate: MLModel?
    }

    let descriptor: PipelineDescriptor
    let maxContext: Int
    let batchSize: Int
    let computeUnits: MLComputeUnits

    private let embeddings: MLModel?
    private let chunks: [Chunk]
    private let lmHead: MLModel?
    private let maskType: TensorSpec.DataType
    private var state: MLState?
    private var cachedTokens: [Int] = []
    var gate: MemoryGate?
    let isStateless = false
    /// 出力層は最後のトークンの分しか計算しないので、logits の大きさは気にしなくてよい
    let logitsElementBytes = 2
    var needsNewState: Bool { state == nil }

    private init(descriptor: PipelineDescriptor, embeddings: MLModel?, chunks: [Chunk], lmHead: MLModel?,
                 computeUnits: MLComputeUnits) throws {
        guard let first = chunks.first else { throw EngineError.unsupportedModel(L("本体のモデルがありません", "no body model")) }
        self.descriptor = descriptor
        self.embeddings = embeddings
        self.chunks = chunks
        self.lmHead = lmHead
        self.computeUnits = computeUnits
        let inputs = first.infer.modelDescription.inputDescriptionsByName
        guard let mask = inputs["causal_mask"]?.multiArrayConstraint else {
            throw EngineError.unsupportedModel(L("causal_mask の入力がありません", "no causal_mask input"))
        }
        maskType = TensorSpec.DataType(mask.dataType)
        let maskLength = mask.shape.last?.intValue ?? 0
        maxContext = descriptor.contextLength ?? maskLength
        guard maxContext >= 2 else { throw EngineError.unsupportedModel(L("最大の長さが分かりません", "unknown maximum length")) }
        // prefill 関数の 1 回あたりのトークン数
        var batch = 1
        if let prefill = first.prefill {
            let prefillInputs = prefill.modelDescription.inputDescriptionsByName
            if let positions = prefillInputs["position_ids"]?.multiArrayConstraint?.shape.first?.intValue {
                batch = positions
            } else if let hidden = prefillInputs["hidden_states"]?.multiArrayConstraint?.shape, hidden.count >= 2 {
                batch = hidden[1].intValue
            }
        }
        // 埋め込みが一度に受け取れる長さも確かめる
        if let embeddings, let description = embeddings.modelDescription.inputDescriptionsByName["input_ids"],
           let spec = TensorSpec(feature: description), let upper = spec.upperBound(axis: 1), upper < batch {
            batch = 1
        }
        batchSize = max(1, batch)
    }

    static func load(files: PipelineFiles, computeUnits: MLComputeUnits) async throws -> PipelineLanguageModel {
        func load(_ url: URL, function: String?) async throws -> MLModel {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            if let function { configuration.functionName = function }
            return try await MLModel.load(contentsOf: url, configuration: configuration)
        }
        var embeddings: MLModel?
        if let url = files.embeddings { embeddings = try await load(url, function: nil) }
        var chunks: [Chunk] = []
        for url in files.chunks {
            try Task.checkCancellation()
            let functions = Self.functionNames(of: url)
            if functions.contains("infer") {
                let infer = try await load(url, function: "infer")
                let prefill = functions.contains("prefill") ? try await load(url, function: "prefill") : nil
                let inferRotate = functions.contains("infer_rotate") ? try await load(url, function: "infer_rotate") : nil
                let prefillRotate = functions.contains("prefill_rotate") ? try await load(url, function: "prefill_rotate") : nil
                chunks.append(Chunk(infer: infer, prefill: prefill, inferRotate: inferRotate, prefillRotate: prefillRotate))
            } else {
                chunks.append(Chunk(infer: try await load(url, function: nil)))
            }
        }
        var lmHead: MLModel?
        if let url = files.lmHead { lmHead = try await load(url, function: nil) }
        return try PipelineLanguageModel(descriptor: files.descriptor, embeddings: embeddings, chunks: chunks,
                                         lmHead: lmHead, computeUnits: computeUnits)
    }

    /// .mlmodelc/metadata.json に書かれた関数の名前（複数の関数を持つモデル）
    static func functionNames(of url: URL) -> [String] {
        guard let data = try? Data(contentsOf: url.appendingPathComponent("metadata.json")),
              let root = try? JSONSerialization.jsonObject(with: data) else { return [] }
        let object = (root as? [[String: Any]])?.first ?? (root as? [String: Any]) ?? [:]
        return (object["functions"] as? [[String: Any]] ?? []).compactMap { $0["name"] as? String }
    }

    var interfaceDescription: String {
        let kind = descriptor.isMonolithic
            ? L("一体型", "monolithic")
            : L("埋め込み + \(chunks.count) チャンク + 出力層", "embeddings + \(chunks.count) chunks + head")
        var text = L("分割モデル（ANEMLL・\(kind)）・最大 \(maxContext) トークン",
                     "Split model (ANEMLL, \(kind)) · up to \(maxContext) tokens")
        if descriptor.argmaxInModel {
            text += L("・常に最も確率の高いトークンを選ぶ形式", " · greedy only")
        }
        return text
    }

    var computeDescription: String { computeUnits.label }

    func resetCache() {
        state = nil
        cachedTokens = []
    }

    // MARK: - 評価

    func evaluate(tokens: [Int], chunkSize: Int, progress: ((Int, Int) -> Void)?) async throws
        -> (scores: NextTokenScores, reusedTokens: Int) {
        guard !tokens.isEmpty else { throw EngineError.emptyPrompt }
        guard tokens.count <= maxContext else { throw EngineError.contextOverflow(tokens.count, maxContext) }

        var reused = 0
        let limit = min(cachedTokens.count, tokens.count)
        while reused < limit && cachedTokens[reused] == tokens[reused] { reused += 1 }
        if reused >= tokens.count { reused = tokens.count - 1 }
        // スライディングウィンドウ（Gemma 3）は、窓を回したあとに巻き戻せないので最初からやり直す
        if let window = descriptor.slidingWindow, reused < cachedTokens.count, cachedTokens.count > window {
            reused = 0
            state = nil
        }
        if state == nil {
            state = (chunks[0].prefill ?? chunks[0].infer).makeState()
            reused = 0
        }
        guard let state else { throw EngineError.modelNotLoaded }

        // prefill はバッチの境目からやり直す（KV の位置がそろうように）
        let lastIndex = tokens.count - 1
        var position = batchSize > 1 ? (reused / batchSize) * batchSize : reused
        let start = position
        cachedTokens = Array(tokens[0..<position])
        progress?(position, tokens.count)
        while position < lastIndex {
            try Task.checkCancellation()
            try gate?.check()
            if batchSize > 1, chunks[0].prefill != nil, position + batchSize <= maxContext {
                let count = min(batchSize, lastIndex - position)
                try await prefill(Array(tokens[position..<(position + count)]), at: position, state: state)
                cachedTokens.append(contentsOf: tokens[position..<(position + count)])
                position += count
            } else {
                _ = try await infer(token: tokens[position], at: position, state: state, needScores: false)
                cachedTokens.append(tokens[position])
                position += 1
            }
            progress?(position, tokens.count)
        }
        let scores = try await infer(token: tokens[lastIndex], at: lastIndex, state: state, needScores: true)
        cachedTokens.append(tokens[lastIndex])
        progress?(tokens.count, tokens.count)
        return (scores ?? .logits([]), start)
    }

    func append(token: Int) async throws -> NextTokenScores {
        guard let state else { throw EngineError.modelNotLoaded }
        let position = cachedTokens.count
        guard position < maxContext else { throw EngineError.contextOverflow(position + 1, maxContext) }
        let scores = try await infer(token: token, at: position, state: state, needScores: true)
        cachedTokens.append(token)
        return scores ?? .logits([])
    }

    // MARK: - 1 回分の計算

    /// batchSize 個にそろえて（足りなければ 0 で埋めて）まとめて読み込む。出力は使わない。
    private func prefill(_ tokens: [Int], at position: Int, state: MLState) async throws {
        try await InferenceQueue.run { try self.prefillNow(tokens, at: position, state: state) }
    }

    /// 1 トークンを計算する。needScores なら次のトークンのスコアを返す。
    private func infer(token: Int, at position: Int, state: MLState, needScores: Bool) async throws -> NextTokenScores? {
        try await InferenceQueue.run { try self.inferNow(token: token, at: position, state: state, needScores: needScores) }
    }

    // InferenceQueue の中で動かす（予測ごとに一時的な出力が解放される）

    private func prefillNow(_ tokens: [Int], at position: Int, state: MLState) throws {
        var padded = tokens
        padded.append(contentsOf: repeatElement(0, count: batchSize - tokens.count))
        let ids = try CoreMLTensors.int32Array(padded, shape: [1, batchSize])
        let positions = try CoreMLTensors.int32Array(Array(position..<(position + batchSize)), shape: [batchSize])
        let mask = try CoreMLTensors.causalMask(rows: batchSize, columns: maxContext, firstRowPosition: position,
                                                dataType: maskType)
        let current = try CoreMLTensors.int32Array([position], shape: [1])
        let rotate = (descriptor.slidingWindow.map { position >= $0 }) ?? false
        var hidden = try embedNow(ids)
        for chunk in chunks {
            guard let base = chunk.prefill else { throw EngineError.unsupportedModel("prefill") }
            let model = rotate ? (chunk.prefillRotate ?? base) : base
            let input = try inputs(for: model, hidden: hidden, ids: ids, positions: positions, mask: mask,
                                   current: current, position: position)
            try autoreleasepool {
                let output = try model.prediction(from: input, using: state, options: MLPredictionOptions())
                if let next = output.featureValue(for: "output_hidden_states")?.multiArrayValue { hidden = next }
            }
        }
    }

    private func inferNow(token: Int, at position: Int, state: MLState, needScores: Bool) throws -> NextTokenScores? {
        let ids = try CoreMLTensors.int32Array([token], shape: [1, 1])
        let positions = try CoreMLTensors.int32Array([position], shape: [1])
        let mask = try CoreMLTensors.causalMask(rows: 1, columns: maxContext, firstRowPosition: position, dataType: maskType)
        let current = try CoreMLTensors.int32Array([position], shape: [1])
        let rotate = (descriptor.slidingWindow.map { position >= $0 }) ?? false
        var hidden = try embedNow(ids)
        var lastOutput: MLFeatureProvider?
        for chunk in chunks {
            let model = rotate ? (chunk.inferRotate ?? chunk.infer) : chunk.infer
            let input = try inputs(for: model, hidden: hidden, ids: ids, positions: positions, mask: mask,
                                   current: current, position: position)
            let output = try model.prediction(from: input, using: state, options: MLPredictionOptions())
            lastOutput = output
            if let next = output.featureValue(for: "output_hidden_states")?.multiArrayValue { hidden = next }
        }
        guard needScores else { return nil }
        if let lmHead {
            guard let hidden else { throw EngineError.missingOutput("output_hidden_states") }
            let input = try MLDictionaryFeatureProvider(dictionary: ["hidden_states": MLFeatureValue(multiArray: hidden)])
            lastOutput = try lmHead.prediction(from: input, options: MLPredictionOptions())
        }
        guard let lastOutput else { throw EngineError.missingOutput("logits") }
        return try scores(from: lastOutput)
    }

    /// 埋め込み（一体型なら nil）
    private func embedNow(_ ids: MLMultiArray) throws -> MLMultiArray? {
        guard let embeddings else { return nil }
        let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: ids)])
        let output = try embeddings.prediction(from: input, options: MLPredictionOptions())
        if let hidden = output.featureValue(for: "hidden_states")?.multiArrayValue { return hidden }
        for name in output.featureNames {
            if let value = output.featureValue(for: name)?.multiArrayValue { return value }
        }
        throw EngineError.missingOutput("hidden_states")
    }

    /// モデルが求める入力だけを渡す
    private func inputs(for model: MLModel, hidden: MLMultiArray?, ids: MLMultiArray, positions: MLMultiArray,
                        mask: MLMultiArray, current: MLMultiArray, position: Int) throws -> MLDictionaryFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        for name in model.modelDescription.inputDescriptionsByName.keys {
            switch name {
            case "hidden_states":
                guard let hidden else { throw EngineError.missingOutput("hidden_states") }
                features[name] = MLFeatureValue(multiArray: hidden)
            case "input_ids":
                features[name] = MLFeatureValue(multiArray: ids)
            case "position_ids":
                features[name] = MLFeatureValue(multiArray: positions)
            case "causal_mask":
                features[name] = MLFeatureValue(multiArray: mask)
            case "current_pos":
                features[name] = MLFeatureValue(multiArray: current)
            case "update_mask":
                // 古い形式: 書き込む位置だけ 1 の [1, 1, 文脈長, 1]
                let update = try MLMultiArray(shape: [1, 1, NSNumber(value: maxContext), 1], dataType: .float16)
                update.withUnsafeMutableBufferPointer(ofType: Float16.self) { pointer, strides in
                    pointer.baseAddress?.update(repeating: 0, count: pointer.count)
                    if position < maxContext { pointer[position * strides[2]] = 1 }
                }
                features[name] = MLFeatureValue(multiArray: update)
            default:
                throw EngineError.unsupportedModel(L("未対応の入力 \(name)", "unsupported input \(name)"))
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// 出力層の出力から次のトークンのスコアを作る
    private func scores(from output: MLFeatureProvider) throws -> NextTokenScores {
        if let indices = output.featureValue(for: "argmax_idx")?.multiArrayValue,
           let values = output.featureValue(for: "argmax_val")?.multiArrayValue {
            let count = indices.count
            let offsets = descriptor.argmaxOffsets(count: count)
            var best = 0
            var bestValue = -Float.infinity
            for index in 0..<count {
                let value = values[index].floatValue
                if value > bestValue {
                    bestValue = value
                    best = index
                }
            }
            return .token(indices[best].intValue + (best < offsets.count ? offsets[best] : 0))
        }
        // logits1, logits2, … を順番につなぐ（無ければ logits / output_logits）
        let numbered = output.featureNames.compactMap { name -> (Int, String)? in
            guard name.hasPrefix("logits"), let number = Int(name.dropFirst("logits".count)) else { return nil }
            return (number, name)
        }.sorted { $0.0 < $1.0 }.map(\.1)
        let names = numbered.isEmpty ? ["logits", "output_logits"].filter { output.featureNames.contains($0) } : numbered
        let arrays = names.compactMap { output.featureValue(for: $0)?.multiArrayValue }
        guard !arrays.isEmpty else { throw EngineError.missingOutput("logits") }
        var layouts: [(offset: Int, count: Int, stride: Int)] = []
        for array in arrays {
            layouts.append(try CoreMLTensors.rowLayout(of: array, position: max(CoreMLTensors.positions(of: array) - 1, 0)))
        }
        var result = [Float](repeating: 0, count: layouts.reduce(0) { $0 + $1.count })
        var offset = 0
        for (array, layout) in zip(arrays, layouts) {
            CoreMLTensors.copyRow(of: array, elementOffset: layout.offset, count: layout.count, stride: layout.stride,
                                  into: &result, at: offset)
            offset += layout.count
        }
        return .logits(result)
    }
}
