import Foundation

/// GGUF から変換したモデルを動かすのに要るファイル。
struct GGUFModelFiles: Sendable {
    /// 変換後の重み（.llw）
    var store: URL
    /// KV キャッシュを用意するトークン数
    var context: Int
    /// 重みのメモリへの置き方
    var memory = GGUFMemoryOptions()
    /// 先読み（MTP）用のモデルを変換したファイル（あるときだけ）
    var drafter: URL? = nil
    /// 先読みで 1 回に予想するトークンの数の上限（0 なら使わない。当たり方を見て、この中で増やしたり減らしたりする）
    var drafts = 0

    static let draftsKey = "gguf.mtpDrafts"
    /// 既定は使わない（今の GPU の作りでは、先読みを使うとかえって遅くなる。速くできたら既定で使う）
    static let defaultDrafts = 0

    /// 保存してある「先読みの数」（無ければ既定）
    static var storedDrafts: Int {
        guard let value = UserDefaults.standard.object(forKey: draftsKey) as? Int else { return defaultDrafts }
        return max(0, min(value, 4))
    }
}

/// 先読みの数を、当たり方から決める。位置ごとの当たる割合（直前まで当たったときの）を覚えておき、
/// 「1 回の往復で進めるトークンの数 ÷ その手間」がいちばん大きくなる数を選ぶ
struct DraftPlanner {
    /// 予想 1 つの手間（本体の 1 回を 1 とする。先読みは小さく、まとめての確かめも重みを読むのは 1 回）
    static let draftCost = 0.08
    static let verifyCost = 0.04

    let maximum: Int
    /// i 番目の予想が当たる割合（i − 1 番目まで当たったとき）
    private(set) var rates: [Double]

    init(maximum: Int) {
        self.maximum = max(0, maximum)
        rates = Array(repeating: 0.5, count: self.maximum)
    }

    /// 今使う予想の数
    var count: Int {
        guard maximum > 0 else { return 0 }
        var best = 0
        var bestValue = 1.0
        var expected = 1.0, chance = 1.0
        for k in 1...maximum {
            chance *= rates[k - 1]
            expected += chance
            let value = expected / (1 + Double(k) * (Self.draftCost + Self.verifyCost))
            if value > bestValue {
                bestValue = value
                best = k
            }
        }
        // 当たらなくなっても、ときどきは 1 つ予想して割合を確かめ直す
        return max(best, 1)
    }

    /// drafted 個予想して、accepted 個当たった
    mutating func record(drafted: Int, accepted: Int) {
        for index in 0..<min(drafted, maximum) {
            guard index <= accepted else { break }
            let hit: Double = index < accepted ? 1 : 0
            rates[index] = rates[index] * 0.8 + hit * 0.2
        }
    }
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

/// GGUF（llama.cpp 形式）のモデルを動かす。Core ML は使わない。
///
/// GPU（Metal）で計算できるときは GPU、できないときは CPU（Accelerate）で計算する。
/// GPU の計算が途中で失敗したとき（アプリが裏に回った など）は、その回は CPU で計算し直す。
final class GGUFLanguageModel: LanguageModelRunner, SpeculativeRunner {
    private var model: any GGUFEvaluator
    /// GPU 用のモデル（GPU を使わないときは nil）
    private let metal: (any GGUFEvaluator)?
    private let files: GGUFModelFiles
    /// GPU を使わなかった・使えなくなった理由（画面に出す）
    private var note: String?
    private var cachedTokens: [Int] = []
    var gate: MemoryGate?

    let isStateless = false
    /// KV キャッシュは使った分だけ増えていくので、先にまとめて確保するものは無い
    let needsNewState = false
    let logitsElementBytes = MemoryLayout<Float>.size

    private init(model: any GGUFEvaluator, isMetal: Bool, files: GGUFModelFiles, note: String?) {
        self.model = model
        metal = isMetal ? model : nil
        self.files = files
        self.note = note
        planner = DraftPlanner(maximum: files.drafts)
    }

    /// 先読み: 予想の数の決め方と、確かめているトークン（[最後に選んだもの, 予想 …]）
    private var planner: DraftPlanner
    private var proposed: [Int] = []
    private var proposedDrafts = 0
    /// 先読みで予想した数と当たった数（画面に出す）
    private(set) var draftedTotal = 0
    private(set) var acceptedTotal = 0

    static func load(files: GGUFModelFiles, useGPU: Bool = true) async throws -> GGUFLanguageModel {
        let available = DeviceMemory.availableBytes
        let made = try await InferenceQueue.run {
            try GGUFModelFactory.make(store: files.store, context: files.context, useGPU: useGPU, memory: files.memory,
                                      availableBytes: available, drafter: files.drafts > 0 ? files.drafter : nil)
        }
        return GGUFLanguageModel(model: made.model, isMetal: made.isMetal, files: files, note: made.note)
    }

    var maxContext: Int { model.maxContext }

    var interfaceDescription: String {
        let config = model.config
        let quantization = config.quantization.map { "\($0) → " } ?? ""
        return L("GGUF（\(config.architecture)・\(quantization)8 ビット）・最大 \(maxContext) トークン",
                 "GGUF (\(config.architecture), \(quantization)8-bit) · up to \(maxContext) tokens")
    }

    var computeDescription: String {
        guard let note else { return model.computeDescription }
        return model.computeDescription + "（" + note + "）"
    }

    // MARK: - 先読み（MTP）

    var speculates: Bool {
        planner.maximum > 0 && (model as? GGUFSpeculativeEvaluator)?.canDraft == true
    }

    func propose(after token: Int) async throws -> SpeculativeProposal {
        guard cachedTokens.count < maxContext else { throw EngineError.contextOverflow(cachedTokens.count + 1, maxContext) }
        guard let target = model as? GGUFSpeculativeEvaluator, target.canDraft else {
            let scores = try await append(token: token)
            return SpeculativeProposal(drafts: []) { _ in
                if case .logits(let logits) = scores { return logits }
                return []
            }
        }
        // 予想の数（当たり方と、コンテキストの残りから）
        let room = max(0, maxContext - cachedTokens.count - 1)
        let count = min(planner.count, room)
        do {
            let drafts = try await InferenceQueue.run { () throws -> [Int] in
                let drafts = try target.draft(after: token, hidden: target.lastHidden, count: count)
                try target.verify([token] + drafts)
                return drafts
            }
            proposed = [token] + drafts
            proposedDrafts = drafts.count
            return SpeculativeProposal(drafts: drafts) { index in
                try await InferenceQueue.run { target.verifiedLogits(index) }
            }
        } catch let error as GGUFMetalError {
            // GPU が使えなくなった: ここまでの系列ごと CPU で計算し直して、予想はやめる
            proposed = []
            let tokens = cachedTokens + [token]
            try await switchToCPU(error)
            let scores = try await read(tokens, chunkSize: GGUFModel.maxBatch, progress: nil).scores
            return SpeculativeProposal(drafts: []) { _ in
                if case .logits(let logits) = scores { return logits }
                return []
            }
        }
    }

    func commit(accepted: Int) async throws {
        guard !proposed.isEmpty, let target = model as? GGUFSpeculativeEvaluator else { return }
        let kept = min(accepted, proposedDrafts)
        try await InferenceQueue.run { target.accept(kept) }
        cachedTokens.append(contentsOf: proposed.prefix(kept + 1))
        planner.record(drafted: proposedDrafts, accepted: kept)
        draftedTotal += proposedDrafts
        acceptedTotal += kept
        proposed = []
    }

    func resetCache() {
        cachedTokens = []
        // 巻き戻すだけでなく、KV キャッシュのメモリも手放す（メモリが足りなくなったときに空けられるように）
        model.releaseCache()
    }

    func evaluate(tokens: [Int], chunkSize: Int, progress: ((Int, Int) -> Void)?) async throws
        -> (scores: NextTokenScores, reusedTokens: Int) {
        guard !tokens.isEmpty else { throw EngineError.emptyPrompt }
        guard tokens.count <= maxContext else { throw EngineError.contextOverflow(tokens.count, maxContext) }
        returnToGPU()
        do {
            return try await read(tokens, chunkSize: chunkSize, progress: progress)
        } catch let error as GGUFMetalError {
            try await switchToCPU(error)
            return try await read(tokens, chunkSize: chunkSize, progress: progress)
        }
    }

    func append(token: Int) async throws -> NextTokenScores {
        guard cachedTokens.count < maxContext else { throw EngineError.contextOverflow(cachedTokens.count + 1, maxContext) }
        let model = self.model
        do {
            let logits = try await InferenceQueue.run { try model.evaluate([token]) }
            cachedTokens.append(token)
            return .logits(logits)
        } catch let error as GGUFMetalError {
            // GPU が使えなくなった: ここまでの系列ごと CPU で計算し直す
            let tokens = cachedTokens + [token]
            try await switchToCPU(error)
            return try await read(tokens, chunkSize: GGUFModel.maxBatch, progress: nil).scores
        }
    }

    /// モデル内部の系列を tokens にそろえて、次のスコアを返す
    private func read(_ tokens: [Int], chunkSize: Int, progress: ((Int, Int) -> Void)?) async throws
        -> (scores: NextTokenScores, reusedTokens: Int) {
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

    /// GPU の計算が失敗したので、CPU 用のモデルに切り替える（系列は最初から読み直す）
    private func switchToCPU(_ error: GGUFMetalError) async throws {
        // 用意の段階のエラー（.unavailable）はここには来ない。計算の失敗だけを CPU でやり直す
        guard case .failed = error else { throw error }
        model.releaseCache()
        cachedTokens = []
        let files = self.files
        let available = DeviceMemory.availableBytes
        let made = try await InferenceQueue.run {
            try GGUFModelFactory.make(store: files.store, context: files.context, useGPU: false, memory: files.memory,
                                      availableBytes: available)
        }
        model = made.model
        note = L("GPU が使えなくなったので CPU で計算中", "GPU became unavailable; computing on the CPU")
    }

    /// 前に GPU が失敗して CPU に切り替えていたら、新しい生成の始めに GPU に戻す（また失敗すれば CPU に戻る）
    private func returnToGPU() {
        guard let metal, model !== metal else { return }
        model.releaseCache()
        cachedTokens = []
        model = metal
        note = nil
    }
}
