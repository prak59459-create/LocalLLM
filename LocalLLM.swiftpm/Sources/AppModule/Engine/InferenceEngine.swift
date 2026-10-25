import CoreML
import Foundation

/// 生成のリクエスト（UI から渡す内容）。
struct GenerationRequest: Sendable {
    var systemPrompt: String?
    var history: [ChatMessage]          // 最後が user（または tool）のメッセージ
    var parameters: GenerationParameters
    var enableThinking: Bool?
    /// モデルに使わせるツール（空なら使わない）
    var tools: [ToolDefinition] = []
}

/// 生成中に UI へ流すイベント。
enum GenerationEvent: Sendable {
    /// プロンプトを組み立てた（トークン数、捨てた古いメッセージ数）
    case promptReady(promptTokens: Int, droppedMessages: Int)
    /// プロンプトの読み込み（prefill）の進み具合
    case prefill(done: Int, total: Int)
    /// 新しく確定した文字列
    case text(String)
    /// 画面に出す途中経過（メモリを空けている、など）
    case status(String)
    /// 終了（統計付き）
    case finished(GenerationStats)
}

/// 読み込み済みモデルの情報（UI 表示用）。
struct LoadedModelInfo: Sendable, Hashable {
    var modelID: String
    var displayName: String
    var maxContext: Int
    var interfaceDescription: String
    var computeDescription: String
    var usesJinjaTemplate: Bool
    var vocabSize: Int
    var loadSeconds: Double
}

/// モデルの読み込みと文章生成をまとめて受け持つ actor。
/// Core ML のモデルと state は 1 つの生成ループからしか触らないよう、ここに閉じ込める。
actor InferenceEngine {
    private struct Loaded {
        var info: LoadedModelInfo
        var model: LanguageModelRunner
        var tokenizer: HFTokenizer
        var renderer: ChatTemplateRenderer
        var stopTokenIDs: Set<Int>
        var stopStrings: [String]
    }

    private var loaded: Loaded?
    private var currentTask: Task<Void, Never>?
    private var prefillChunkSize = 128
    private var contextLimit = 0
    /// 計算の前にメモリを確かめるもの（読み込んだモデルと共有する）
    private let gate = MemoryGate()
    /// 読み直しに使う、今のモデルのファイルと計算ユニット
    private var reloadSource: (files: ModelFiles, units: ComputeUnitsOption)?
    /// 読み込んだ直後のメモリ使用量（あとで増えすぎていないかを見る）
    private var baselineFootprint: Int64?
    /// 次の生成の前にモデルを読み直してメモリを空ける
    private var needsReload = false
    /// メモリが足りなくなったときに縮めた、コンテキストの上限（別のモデルを読み込むと元に戻す）
    private var memoryContextCap: Int?
    /// 読み込みのたびに増やす。読み込み中に別の読み込み・解放が来たら、古いほうの結果は捨てる。
    private var loadGeneration = 0

    var loadedInfo: LoadedModelInfo? { loaded?.info }

    /// モデルを読み込む（既に別のモデルが読み込まれていれば解放してから）。
    func load(files: ModelFiles, computeUnits: ComputeUnitsOption, prefillChunkSize: Int) async throws -> LoadedModelInfo {
        loadGeneration += 1
        let generation = loadGeneration
        await cancelCurrentGeneration()
        loaded = nil   // 先に解放してメモリを空ける
        self.prefillChunkSize = max(1, prefillChunkSize)

        let start = Date()
        let tokenizer = try HFTokenizer(tokenizerJSON: try Data(contentsOf: files.tokenizerJSON),
                                        tokenizerConfigJSON: files.tokenizerConfigJSON.flatMap { try? Data(contentsOf: $0) })
        let template = files.chatTemplateOverride ?? tokenizer.chatTemplate
        let renderer = ChatTemplateRenderer(template: template, bosToken: tokenizer.bosToken, eosToken: tokenizer.eosToken)
        let stopIDs = StopTokenResolver.resolve(
            tokenizer: tokenizer,
            eosToken: tokenizer.eosToken,
            generationConfigJSON: files.generationConfigJSON.flatMap { try? Data(contentsOf: $0) },
            configJSON: files.configJSON.flatMap { try? Data(contentsOf: $0) },
            extraTokens: renderer.endOfTurnTokens
        )
        var stopStrings = renderer.endOfTurnTokens
        if let eos = tokenizer.eosToken, !stopStrings.contains(eos) { stopStrings.append(eos) }

        let model = try await Self.makeRunner(files: files, computeUnits: computeUnits)
        // 待っている間に別の読み込みや解放が始まっていたら、この結果は使わない
        guard generation == loadGeneration else { throw CancellationError() }
        gate.vocabSize = tokenizer.vocabSize
        gate.bytesPerLogit = model.logitsElementBytes
        model.gate = gate
        reloadSource = (files, computeUnits)
        needsReload = false
        memoryContextCap = nil

        let info = LoadedModelInfo(
            modelID: files.modelID,
            displayName: files.displayName,
            maxContext: model.maxContext,
            interfaceDescription: model.interfaceDescription,
            computeDescription: model.computeDescription,
            usesJinjaTemplate: renderer.usesJinja,
            vocabSize: tokenizer.vocabSize,
            loadSeconds: Date().timeIntervalSince(start)
        )
        loaded = Loaded(info: info, model: model, tokenizer: tokenizer, renderer: renderer,
                        stopTokenIDs: stopIDs, stopStrings: stopStrings.filter { !$0.isEmpty })
        baselineFootprint = DeviceMemory.footprintBytes
        return info
    }

    private static func makeRunner(files: ModelFiles, computeUnits: ComputeUnitsOption) async throws -> LanguageModelRunner {
        switch files.source {
        case .single(let url):
            return try await CoreMLLanguageModel.load(compiledModelURL: url, computeUnits: computeUnits.resolved(isPipeline: false))
        case .pipeline(let pipeline):
            return try await PipelineLanguageModel.load(files: pipeline, computeUnits: computeUnits.resolved(isPipeline: true))
        }
    }

    /// iOS からメモリ不足の警告が来た。しばらく控えめに計算し、生成していなければ KV キャッシュを捨てて空ける。
    func noteMemoryWarning() {
        gate.noteWarning()
        if currentTask == nil { loaded?.model.resetCache() }
    }

    /// Core ML の中に溜まったメモリを空けるため、モデルだけを読み直す（トークナイザなどはそのまま）
    private func reloadRunner() async throws {
        guard let source = reloadSource else { return }
        var kept: (info: LoadedModelInfo, tokenizer: HFTokenizer, renderer: ChatTemplateRenderer, ids: Set<Int>, strings: [String])?
        do {
            if let current = loaded {
                kept = (current.info, current.tokenizer, current.renderer, current.stopTokenIDs, current.stopStrings)
            }
        }
        loaded = nil   // 先に古いモデルを手放す
        guard let kept else { return }
        let model = try await Self.makeRunner(files: source.files, computeUnits: source.units)
        model.gate = gate
        loaded = Loaded(info: kept.info, model: model, tokenizer: kept.tokenizer, renderer: kept.renderer,
                        stopTokenIDs: kept.ids, stopStrings: kept.strings)
        baselineFootprint = DeviceMemory.footprintBytes
        needsReload = false
    }

    /// メモリ不足で止めたあと: まだ危なければ KV キャッシュを捨て、読み込んだときより大きく増えていれば次に読み直す
    private func recoverFromLowMemory() {
        if DeviceMemory.budget?.isLow ?? false { loaded?.model.resetCache() }
        if let base = baselineFootprint, let now = DeviceMemory.footprintBytes,
           now - base > max(300 << 20, base / 3) {
            needsReload = true
        }
    }

    /// モデルを解放する。
    func unload() async {
        loadGeneration += 1
        await cancelCurrentGeneration()
        loaded = nil
        reloadSource = nil
        needsReload = false
        memoryContextCap = nil
    }

    /// 会話の途中経過（KV キャッシュ）を捨てる。生成中なら止めてから。
    func resetCache() async {
        await cancelCurrentGeneration()
        loaded?.model.resetCache()
    }

    /// 次の生成から使う（生成中のものには影響しない）。
    /// 1 回に使うコンテキストの上限（0 ならモデルの最大まで）
    func setContextLimit(_ limit: Int) {
        contextLimit = max(0, limit)
    }

    func setPrefillChunkSize(_ size: Int) {
        prefillChunkSize = max(1, size)
    }

    /// 文章を生成する。返ってくるストリームを途中で止める（for ループを抜ける／Task をキャンセルする）と生成も止まる。
    nonisolated func generate(_ request: GenerationRequest) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await self.startGeneration(request, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - 生成ループ

    private func startGeneration(_ request: GenerationRequest,
                                 continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation) async {
        // 前の生成が残っていれば止め、それが終わってから始める（同じ state を同時に触らないため）。
        // ここでは await しないで currentTask を差し替えるので、続けて呼ばれても必ず 1 つずつ順番に動く。
        let previous = currentTask
        previous?.cancel()
        let task = Task {
            await previous?.value
            await self.runGeneration(request, continuation: continuation)
        }
        currentTask = task
        await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        if currentTask == task { currentTask = nil }
    }

    /// 生成中のものをすべて止めて、終わるまで待つ。
    private func cancelCurrentGeneration() async {
        while let task = currentTask {
            task.cancel()
            await task.value
            if currentTask == task { currentTask = nil }
        }
    }

    private func runGeneration(_ request: GenerationRequest,
                               continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation) async {
        var stats = GenerationStats()
        if Task.isCancelled {
            stats.stopReason = .cancelled
            continuation.yield(.finished(stats))
            continuation.finish()
            return
        }
        if needsReload {
            continuation.yield(.status(L("メモリを空けるため、モデルを読み直しています…", "Reloading the model to free memory…")))
            do {
                try await reloadRunner()
            } catch {
                continuation.finish(throwing: error)
                return
            }
        }
        // トークナイザなどはモデルを読み直しても変わらない。モデル本体は読み直すことがあるので、ここでは持たない
        // （古いモデルを持ったまま読み直すと、2 つ分のメモリを使ってしまう）
        guard case let (tokenizer, renderer, stopTokenIDs, stopStrings, modelMax)? = loadedSetup() else {
            continuation.finish(throwing: EngineError.modelNotLoaded)
            return
        }
        let parameters = request.parameters
        let chunkSize = prefillChunkSize
        var maxContext = contextLimit > 0 ? min(modelMax, max(contextLimit, 256)) : modelMax
        // 前にメモリが足りなくなった長さより長くはしない
        if let cap = memoryContextCap { maxContext = min(maxContext, cap) }

        var sampler = TokenSampler(parameters: parameters)
        let detokenizer = StreamingDetokenizer(tokenizer: tokenizer, skipSpecialTokens: true)
        var stopMatcher = StopSequenceMatcher(stopSequences: stopStrings)
        /// ここまでに生成したトークン（読み直したあとも、この続きから生成する）
        var generated: [Int] = []
        var reason: StopReason = .maxTokens
        var stoppedByString = false
        var retries = 0
        var needsRecovery = false
        var decodeSeconds = 0.0

        do {
            attempt: while true {
                // 前の回のモデルへの参照が消えてから、メモリを空ける（モデルを読み直すこともある）
                if needsRecovery {
                    needsRecovery = false
                    try await freeMemoryForRetry()
                }
                guard let model = loaded?.model else { throw EngineError.modelNotLoaded }
                var limit = maxContext
                // 毎回系列全体を入力するモデルは、長いほど 1 回でメモリを使うので、今の空きで扱える長さまでにする
                if model.isStateless, let stateless = gate.statelessLimit() {
                    limit = max(64, min(limit, stateless))
                }
                // KV キャッシュをこれから作るなら、その分の空きがあるか
                if model.needsNewState, let bytes = reloadSource?.files.stateBytes, bytes > 0, !gate.canAllocate(bytes) {
                    throw EngineError.stateTooLarge(needed: bytes, available: DeviceMemory.availableBytes)
                }
                // 生成用に残しておく枠（コンテキストの半分まで。読み直すときは、もう生成した分も入るように）
                var reserve = max(16, min(parameters.maxNewTokens, limit / 2))
                if !generated.isEmpty { reserve = max(reserve, min(generated.count + 64, limit * 3 / 4)) }
                let prompt = try PromptAssembler.assemble(
                    systemPrompt: request.systemPrompt,
                    history: request.history,
                    renderer: renderer,
                    tokenizer: tokenizer,
                    contextLength: limit,
                    reserveTokens: reserve,
                    enableThinking: request.enableThinking,
                    tools: request.tools
                )
                stats.promptTokens = prompt.tokens.count
                continuation.yield(.promptReady(promptTokens: prompt.tokens.count, droppedMessages: prompt.droppedMessageCount))

                var history = prompt.tokens + generated
                let progress = PrefillProgress()
                do {
                    // プロンプト（と、読み直すときは生成済みの分）を読み込む
                    gate.begin()
                    let prefillStart = Date()
                    let (firstScores, reused) = try await model.evaluate(tokens: history, chunkSize: chunkSize) { done, total in
                        progress.done = done
                        continuation.yield(.prefill(done: done, total: total))
                    }
                    if retries == 0 { stats.reusedPromptTokens = reused }
                    stats.prefillSeconds += Date().timeIntervalSince(prefillStart)
                    progress.done = history.count

                    // 1 トークンずつ生成
                    var scores = firstScores
                    let decodeStart = Date()
                    defer { decodeSeconds += Date().timeIntervalSince(decodeStart) }
                    generation: while generated.count < parameters.maxNewTokens {
                        if Task.isCancelled { reason = .cancelled; break }
                        let next: Int
                        switch scores {
                        case .logits(var logits):
                            next = sampler.sample(logits: &logits, previousTokens: history)
                        case .token(let token):
                            // 出力層が選んだトークン（温度などは効かない形式）
                            next = token
                        }
                        if stopTokenIDs.contains(next) { reason = .endOfTurn; break }
                        history.append(next)
                        generated.append(next)
                        stats.generatedTokens = generated.count

                        let piece = detokenizer.append(next)
                        if !piece.isEmpty {
                            let (emit, stopped) = stopMatcher.feed(piece)
                            if !emit.isEmpty { continuation.yield(.text(emit)) }
                            if stopped {
                                reason = .endOfTurn
                                stoppedByString = true
                                break generation
                            }
                        }
                        if history.count >= limit { reason = .contextFull; break }
                        if generated.count >= parameters.maxNewTokens { reason = .maxTokens; break }
                        progress.done = history.count
                        do {
                            // 落ちる前に止める（毎トークン、残りのメモリと減り方を確かめる）
                            try gate.check()
                            scores = try await model.append(token: next)
                        } catch is CancellationError {
                            reason = .cancelled
                            break
                        }
                    }
                    break attempt
                } catch EngineError.lowMemory(let available) {
                    // 足りなくなった: メモリを空けて、短くして読み直す（ここまでに書いた分はそのまま続ける）
                    guard retries < MemoryRecovery.maxRetries, !Task.isCancelled else {
                        throw EngineError.lowMemory(available: available)
                    }
                    retries += 1
                    continuation.yield(.status(L("メモリが足りなくなりそうなので、空けてから古いやり取りを省いて読み直しています…",
                                                 "Running low on memory — freeing it and re-reading with less history…")))
                    if let shrunk = MemoryRecovery.shrunkContext(current: limit, reached: progress.done) {
                        maxContext = shrunk
                        memoryContextCap = min(memoryContextCap ?? modelMax, shrunk)
                    }
                    needsRecovery = true
                    continue attempt
                }
            }
        } catch EngineError.lowMemory {
            // 読み直しても足りなかった
            reason = .lowMemory
            recoverFromLowMemory()
        } catch is CancellationError {
            reason = .cancelled
        } catch {
            continuation.finish(throwing: error)
            return
        }
        stats.decodeSeconds = decodeSeconds

        // 残りを出し切る（停止文字列の一部かもしれないと保留していた分も含む）。
        // 停止文字列そのもので止まった場合は、それ以降は表示しない。
        if !stoppedByString {
            var (emit, stopped) = stopMatcher.feed(detokenizer.flush())
            if !stopped { emit += stopMatcher.flush() }
            if !emit.isEmpty { continuation.yield(.text(emit)) }
        }
        if Task.isCancelled { reason = .cancelled }
        stats.stopReason = reason
        // 最後まで書けて、メモリにも余裕があれば、縮めていた上限を少し戻す
        if reason != .lowMemory, reason != .cancelled, let cap = memoryContextCap,
           let budget = DeviceMemory.budget, budget.available > budget.low * 2 {
            memoryContextCap = MemoryRecovery.relaxedCap(cap, modelMax: modelMax)
        }
        continuation.yield(.finished(stats))
        continuation.finish()
    }

    /// 読み込んだモデルの、生成に使う部品（モデル本体は含めない）
    private func loadedSetup() -> (HFTokenizer, ChatTemplateRenderer, Set<Int>, [String], Int)? {
        guard let loaded else { return nil }
        return (loaded.tokenizer, loaded.renderer, loaded.stopTokenIDs, loaded.stopStrings, loaded.model.maxContext)
    }

    /// メモリ不足で止まったあと、続けるためにメモリを空ける: KV キャッシュを捨て、
    /// まだ足りないか、読み込んだときより大きく増えていれば、モデルも読み直す（Core ML の中に溜まった分が空く）
    private func freeMemoryForRetry() async throws {
        loaded?.model.resetCache()
        try await Task.sleep(nanoseconds: 300_000_000)
        let grown: Bool
        if let base = baselineFootprint, let now = DeviceMemory.footprintBytes {
            grown = now - base > max(200 << 20, base / 4)
        } else {
            grown = false
        }
        if grown || (DeviceMemory.budget?.isLow ?? false) {
            try await reloadRunner()
        }
    }

}

/// プロンプトをどこまで読めたか（足りなくなったとき、どこまで短くするかに使う）
private final class PrefillProgress: @unchecked Sendable {
    var done = 0
}

/// 読み込みに必要なファイルの場所。
struct ModelFiles: Sendable {
    enum Source: Sendable {
        /// 1 つの .mlmodelc
        case single(URL)
        /// 分割モデル（ANEMLL）
        case pipeline(PipelineFiles)
    }

    var modelID: String
    var displayName: String
    var source: Source
    var tokenizerJSON: URL
    var tokenizerConfigJSON: URL?
    var generationConfigJSON: URL?
    var configJSON: URL?
    /// chat_template.jinja / chat_template.json があればその中身（tokenizer_config より優先）
    var chatTemplateOverride: String?
    /// KV キャッシュ（state）の大きさの見積もり（バイト）
    var stateBytes: Int64? = nil
}
