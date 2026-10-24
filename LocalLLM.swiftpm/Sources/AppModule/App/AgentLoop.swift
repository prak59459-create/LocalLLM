import Foundation

/// 「生成 → ツールの呼び出しを読み取る → ツールを実行 → 結果を渡してまた生成」を繰り返す。
/// チャットとコーディングワークスペースの両方で使う。
@MainActor
struct AgentLoop {
    struct Configuration {
        var systemPrompt: String?
        var parameters: GenerationParameters
        var enableThinking: Bool?
        var tools: [ToolDefinition]
        /// ツールを使う往復の上限
        var maxRounds: Int = 6
        /// 毎回モデルに渡す前に、やり取りを短くする（コンテキストとメモリの節約）
        var compact: (([ChatMessage]) -> [ChatMessage])? = nil
    }

    enum Event {
        /// 新しい返答を書き始める
        case assistantStarted
        case promptReady(promptTokens: Int, droppedMessages: Int)
        case prefill(done: Int, total: Int)
        /// 画面に出す本文（ここまでの全体。ツールの呼び出し部分は隠してある）
        case text(String)
        /// 返答が終わった（ツールを呼ぶなら calls に入る）
        case assistantFinished(content: String, calls: [ToolCall], stats: GenerationStats)
        case toolStarted(ToolCall)
        case toolFinished(ToolCall, ToolOutcome)
        /// 途中経過のお知らせ（メモリを空けている、など）
        case status(String)
    }

    /// ツールを実行する前の確認（false なら実行しない）
    typealias Approval = (ToolCall) async -> Bool

    let engine: InferenceEngine

    func run(history: [ChatMessage], configuration: Configuration,
             execute: (ToolCall) async -> ToolOutcome,
             approve: Approval? = nil,
             onEvent: (Event) -> Void) async throws {
        var transcript = history
        let knownTools = Set(configuration.tools.map(\.name))
        for round in 0...max(configuration.maxRounds, 0) {
            try Task.checkCancellation()
            // 最後の往復ではツールを渡さない（必ず文章で答えさせる）
            let isLastRound = round == configuration.maxRounds
            let tools = isLastRound ? [] : configuration.tools
            let history = configuration.compact?(transcript) ?? transcript
            let request = GenerationRequest(systemPrompt: configuration.systemPrompt, history: history,
                                            parameters: configuration.parameters,
                                            enableThinking: configuration.enableThinking, tools: tools)
            onEvent(.assistantStarted)
            var raw = ""
            var stats = GenerationStats()
            // 画面の更新はまとめて行う（トークンごとに全体を読み直して描き直すと重い）
            var shownLength = 0
            var shownAt = Date.distantPast
            for try await event in engine.generate(request) {
                switch event {
                case .promptReady(let promptTokens, let dropped):
                    onEvent(.promptReady(promptTokens: promptTokens, droppedMessages: dropped))
                case .prefill(let done, let total):
                    onEvent(.prefill(done: done, total: total))
                case .status(let text):
                    onEvent(.status(text))
                case .text(let piece):
                    raw += piece
                    let now = Date()
                    if raw.utf8.count - shownLength >= 48 || now.timeIntervalSince(shownAt) >= 0.12 {
                        shownLength = raw.utf8.count
                        shownAt = now
                        onEvent(.text(tools.isEmpty ? raw : ToolCallParser.visiblePart(of: raw)))
                    }
                case .finished(let finalStats):
                    stats = finalStats
                }
            }
            if raw.utf8.count != shownLength {
                onEvent(.text(tools.isEmpty ? raw : ToolCallParser.visiblePart(of: raw)))
            }
            let parsed = tools.isEmpty ? ToolCallParser.Result(content: raw, calls: [])
                                       : ToolCallParser.parse(raw, knownTools: knownTools)
            onEvent(.assistantFinished(content: parsed.calls.isEmpty ? raw : parsed.content, calls: parsed.calls, stats: stats))
            // 止めたとき・メモリが足りなくなったときは、ツールを実行せずに終わる
            guard !parsed.calls.isEmpty, stats.stopReason != .cancelled, stats.stopReason != .lowMemory else { return }

            transcript.append(ChatMessage(role: .assistant, content: parsed.content, toolCalls: parsed.calls))
            for call in parsed.calls {
                try Task.checkCancellation()
                onEvent(.toolStarted(call))
                let outcome: ToolOutcome
                if let approve, await !approve(call) {
                    outcome = ToolOutcome(text: L("ユーザーがこの操作を許可しませんでした。", "The user declined this action."), isError: true)
                } else {
                    outcome = await execute(call)
                }
                onEvent(.toolFinished(call, outcome))
                transcript.append(ChatMessage(role: .tool, content: outcome.text, toolName: call.name, toolCallID: call.id))
            }
        }
    }
}
