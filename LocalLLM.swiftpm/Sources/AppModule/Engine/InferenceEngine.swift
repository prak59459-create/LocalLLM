import CoreML
import Foundation

/// 生成のリクエスト（UI から渡す内容）。
struct GenerationRequest: Sendable {
    var systemPrompt: String?
    var history: [ChatMessage]          // 最後が user のメッセージ
    var parameters: GenerationParameters
    var enableThinking: Bool?
}

/// 生成中に UI へ流すイベント。
enum GenerationEvent: Sendable {
    /// プロンプトを組み立てた（トークン数、捨てた古いメッセージ数）
    case promptReady(promptTokens: Int, droppedMessages: Int)
    /// プロンプトの読み込み（prefill）の進み具合
    case prefill(done: Int, total: Int)
    /// 新しく確定した文字列
    case text(String)
    /// 終了（統計付き）
    case finished(GenerationStats)
}

/// 読み込み済みモデルの情報（UI 表示用）。
struct LoadedModelInfo: Sendable, Hashable {
    var modelID: String
    var displayName: String
    var maxContext: Int
    var interfaceDescription: String
    var usesJinjaTemplate: Bool
    var vocabSize: Int
    var loadSeconds: Double
}

/// モデルの読み込みと文章生成をまとめて受け持つ actor。
/// Core ML のモデルと state は 1 つの生成ループからしか触らないよう、ここに閉じ込める。
actor InferenceEngine {
    private struct Loaded {
        var info: LoadedModelInfo
        var model: CoreMLLanguageModel
        var tokenizer: HFTokenizer
        var renderer: ChatTemplateRenderer
        var stopTokenIDs: Set<Int>
        var stopStrings: [String]
    }

    private var loaded: Loaded?
    private var currentTask: Task<Void, Never>?
    private var prefillChunkSize = 128
    /// 読み込みのたびに増やす。読み込み中に別の読み込み・解放が来たら、古いほうの結果は捨てる。
    private var loadGeneration = 0

    var loadedInfo: LoadedModelInfo? { loaded?.info }

    /// モデルを読み込む（既に別のモデルが読み込まれていれば解放してから）。
    func load(files: ModelFiles, computeUnits: MLComputeUnits, prefillChunkSize: Int) async throws -> LoadedModelInfo {
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

        let model = try await CoreMLLanguageModel.load(compiledModelURL: files.compiledModel, computeUnits: computeUnits)
        // 待っている間に別の読み込みや解放が始まっていたら、この結果は使わない
        guard generation == loadGeneration else { throw CancellationError() }

        let info = LoadedModelInfo(
            modelID: files.modelID,
            displayName: files.displayName,
            maxContext: model.maxContext,
            interfaceDescription: Self.describe(model.interface),
            usesJinjaTemplate: renderer.usesJinja,
            vocabSize: tokenizer.vocabSize,
            loadSeconds: Date().timeIntervalSince(start)
        )
        loaded = Loaded(info: info, model: model, tokenizer: tokenizer, renderer: renderer,
                        stopTokenIDs: stopIDs, stopStrings: stopStrings.filter { !$0.isEmpty })
        return info
    }

    /// モデルを解放する。
    func unload() async {
        loadGeneration += 1
        await cancelCurrentGeneration()
        loaded = nil
    }

    /// 会話の途中経過（KV キャッシュ）を捨てる。生成中なら止めてから。
    func resetCache() async {
        await cancelCurrentGeneration()
        loaded?.model.resetCache()
    }

    /// 次の生成から使う（生成中のものには影響しない）。
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
        guard let loaded else {
            continuation.finish(throwing: EngineError.modelNotLoaded)
            return
        }
        let chunkSize = prefillChunkSize
        do {
            let parameters = request.parameters
            let maxContext = loaded.model.maxContext
            // 生成用に残しておく枠（コンテキストの半分まで）。
            let reserve = max(16, min(parameters.maxNewTokens, maxContext / 2))
            let prompt = try PromptAssembler.assemble(
                systemPrompt: request.systemPrompt,
                history: request.history,
                renderer: loaded.renderer,
                tokenizer: loaded.tokenizer,
                contextLength: maxContext,
                reserveTokens: reserve,
                enableThinking: request.enableThinking
            )
            stats.promptTokens = prompt.tokens.count
            continuation.yield(.promptReady(promptTokens: prompt.tokens.count, droppedMessages: prompt.droppedMessageCount))

            // プロンプトの読み込み
            let prefillStart = Date()
            let (firstLogits, reused) = try await loaded.model.evaluate(tokens: prompt.tokens, chunkSize: chunkSize) { done, total in
                continuation.yield(.prefill(done: done, total: total))
            }
            stats.reusedPromptTokens = reused
            stats.prefillSeconds = Date().timeIntervalSince(prefillStart)

            // 1 トークンずつ生成
            var logits = firstLogits
            var sampler = TokenSampler(parameters: parameters)
            let detokenizer = StreamingDetokenizer(tokenizer: loaded.tokenizer, skipSpecialTokens: true)
            var stopMatcher = StopSequenceMatcher(stopSequences: loaded.stopStrings)
            var history = prompt.tokens
            let decodeStart = Date()
            var reason: StopReason = .maxTokens
            var stoppedByString = false

            generation: while stats.generatedTokens < parameters.maxNewTokens {
                if Task.isCancelled { reason = .cancelled; break }
                let next = sampler.sample(logits: &logits, previousTokens: history)
                if loaded.stopTokenIDs.contains(next) { reason = .endOfTurn; break }
                history.append(next)
                stats.generatedTokens += 1

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
                if history.count >= maxContext { reason = .contextFull; break }
                if stats.generatedTokens >= parameters.maxNewTokens { reason = .maxTokens; break }
                do {
                    logits = try await loaded.model.append(token: next)
                } catch is CancellationError {
                    reason = .cancelled
                    break
                }
            }
            stats.decodeSeconds = Date().timeIntervalSince(decodeStart)

            // 残りを出し切る（停止文字列の一部かもしれないと保留していた分も含む）。
            // 停止文字列そのもので止まった場合は、それ以降は表示しない。
            if !stoppedByString {
                var (emit, stopped) = stopMatcher.feed(detokenizer.flush())
                if !stopped { emit += stopMatcher.flush() }
                if !emit.isEmpty { continuation.yield(.text(emit)) }
            }
            if Task.isCancelled { reason = .cancelled }
            stats.stopReason = reason
            continuation.yield(.finished(stats))
            continuation.finish()
        } catch is CancellationError {
            stats.stopReason = .cancelled
            continuation.yield(.finished(stats))
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    private static func describe(_ interface: ModelInterface) -> String {
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
}

/// 読み込みに必要なファイルの場所。
struct ModelFiles: Sendable {
    var modelID: String
    var displayName: String
    var compiledModel: URL
    var tokenizerJSON: URL
    var tokenizerConfigJSON: URL?
    var generationConfigJSON: URL?
    var configJSON: URL?
    /// chat_template.jinja / chat_template.json があればその中身（tokenizer_config より優先）
    var chatTemplateOverride: String?
}
