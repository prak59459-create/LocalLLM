import Foundation

/// GGUF から変換したモデルを動かすのに要るファイル。
struct GGUFModelFiles: Sendable {
    /// 変換後の重み（.llw）
    var store: URL
    /// KV キャッシュを用意するトークン数
    var context: Int
}

/// 途中で止めるための旗（計算は専用のスレッドで動くので、Task の取り消しを自分で伝える）
private final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

/// GGUF（llama.cpp 形式）のモデルを CPU（Accelerate）で動かす。Core ML は使わない。
final class GGUFLanguageModel: LanguageModelRunner {
    private let model: GGUFModel
    private var cachedTokens: [Int] = []
    var gate: MemoryGate?

    let isStateless = false
    /// KV キャッシュは使った分だけ増えていくので、先にまとめて確保するものは無い
    let needsNewState = false
    let logitsElementBytes = MemoryLayout<Float>.size

    private init(model: GGUFModel) {
        self.model = model
    }

    static func load(files: GGUFModelFiles) async throws -> GGUFLanguageModel {
        let store = files.store
        let context = files.context
        let model = try await InferenceQueue.run {
            try GGUFModel(store: try GGUFWeightStore(url: store), maxContext: context)
        }
        return GGUFLanguageModel(model: model)
    }

    var maxContext: Int { model.maxContext }

    var interfaceDescription: String {
        let config = model.config
        let quantization = config.quantization.map { "\($0) → " } ?? ""
        return L("GGUF（\(config.architecture)・\(quantization)8 ビット）・最大 \(maxContext) トークン",
                 "GGUF (\(config.architecture), \(quantization)8-bit) · up to \(maxContext) tokens")
    }

    var computeDescription: String { "CPU (Accelerate)" }

    func resetCache() {
        cachedTokens = []
        model.truncate(to: 0)
    }

    func evaluate(tokens: [Int], chunkSize: Int, progress: ((Int, Int) -> Void)?) async throws
        -> (scores: NextTokenScores, reusedTokens: Int) {
        guard !tokens.isEmpty else { throw EngineError.emptyPrompt }
        guard tokens.count <= maxContext else { throw EngineError.contextOverflow(tokens.count, maxContext) }
        // 共通の先頭部分は計算し直さない。ただし logits を得るため最低 1 トークンは入力する
        var reused = 0
        let limit = min(cachedTokens.count, tokens.count, model.length)
        while reused < limit && cachedTokens[reused] == tokens[reused] { reused += 1 }
        if reused >= tokens.count { reused = tokens.count - 1 }
        let fresh = Array(tokens[reused...])
        cachedTokens = Array(tokens[0..<reused])
        progress?(reused, tokens.count)

        let model = self.model
        let gate = self.gate
        let total = tokens.count
        let batch = max(1, min(chunkSize, GGUFModel.maxBatch))
        let flag = CancellationFlag()
        // 途中で止まっても、読めたところまでは KV キャッシュに残っているので、それを覚えておく
        defer { cachedTokens = Array(tokens[0..<min(model.length, tokens.count)]) }
        let logits = try await withTaskCancellationHandler {
            try await InferenceQueue.run { () throws -> [Float] in
                model.truncate(to: reused)
                return try model.evaluate(fresh, batch: batch) { done in
                    progress?(reused + done, total)
                    if flag.isCancelled { throw CancellationError() }
                    // 読み終わっていなければ、次のまとまりの前にメモリを確かめる
                    if done < fresh.count { try gate?.check() }
                }
            }
        } onCancel: {
            flag.cancel()
        }
        return (.logits(logits), reused)
    }

    func append(token: Int) async throws -> NextTokenScores {
        guard cachedTokens.count < maxContext else { throw EngineError.contextOverflow(cachedTokens.count + 1, maxContext) }
        let model = self.model
        let logits = try await InferenceQueue.run { try model.evaluate([token]) }
        cachedTokens.append(token)
        return .logits(logits)
    }
}
