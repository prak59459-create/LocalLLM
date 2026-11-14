import Foundation

/// コーディングエージェントに渡すやり取りを短くする（コンテキストとメモリを節約する）。
///
/// エージェントは読んだファイルや書いた中身をやり取りに積み上げていくので、そのままだとすぐにコンテキストが埋まり、
/// 1 回の計算も重くなる。そこで:
/// - 直近の依頼だけを残す
/// - 直近のツールの結果は上限まで残し、それより古い結果は 1 行の要約にする
/// - write_file などで書いた中身（ツールの引数）は、書き終わったら省く（結果に「書きました」が残る）
enum AgentContext {
    struct Budget: Sendable {
        /// 最新のツールの結果の文字数の上限
        var recentToolCharacters = 2400
        /// そのまま残す、最新のツールの結果の数
        var recentToolResults = 2
        /// 古いツールの結果の文字数
        var oldToolCharacters = 160
        /// 前の依頼でのアシスタントの発言の文字数
        var oldAssistantCharacters = 400
        /// ツールの引数（書いたファイルの中身など）の文字数
        var argumentCharacters = 240
        /// 残す依頼の数
        var keepTurns = 3

        /// メモリが足りなくなったときなどに使う、さらに短い設定
        static let tight = Budget(recentToolCharacters: 1200, recentToolResults: 1, oldToolCharacters: 100,
                                  oldAssistantCharacters: 200, argumentCharacters: 120, keepTurns: 2)
    }

    /// 直近 keepTurns 回の依頼（user の発言から始まるまとまり）だけを残す
    static func recentTurns(_ messages: [ChatMessage], keepTurns: Int) -> [ChatMessage] {
        var seen = 0
        var start = 0
        for index in messages.indices.reversed() where messages[index].role == .user {
            seen += 1
            start = index
            if seen >= max(1, keepTurns) { break }
        }
        return seen == 0 ? messages : Array(messages[start...])
    }

    /// 古いものほど短くしたやり取り
    static func compact(_ messages: [ChatMessage], budget: Budget = Budget()) -> [ChatMessage] {
        var result = recentTurns(messages, keepTurns: budget.keepTurns)
        let lastUser = result.lastIndex { $0.role == .user } ?? 0
        // 最新の依頼の中で、新しい順に recentToolResults 個までのツールの結果は長めに残す
        var recentTools = Set<Int>()
        for index in result.indices.reversed() where index > lastUser && result[index].role == .tool {
            if recentTools.count >= budget.recentToolResults { break }
            recentTools.insert(index)
        }
        for index in result.indices {
            switch result[index].role {
            case .tool:
                let limit = recentTools.contains(index) ? budget.recentToolCharacters : budget.oldToolCharacters
                result[index].content = shorten(result[index].content, to: limit, oldResult: !recentTools.contains(index))
            case .assistant:
                if index < lastUser {
                    result[index].content = shorten(result[index].content, to: budget.oldAssistantCharacters, oldResult: false)
                }
                if let calls = result[index].toolCalls {
                    result[index].toolCalls = calls.map { shortenArguments($0, limit: budget.argumentCharacters) }
                }
            case .user, .system:
                break
            }
        }
        return result
    }

    /// 文字数を上限までにする（行の途中では切らないようにする）
    static func shorten(_ text: String, to limit: Int, oldResult: Bool) -> String {
        guard text.count > limit else { return text }
        var head = String(text.prefix(max(0, limit)))
        if let newline = head.lastIndex(of: "\n"), head.distance(from: head.startIndex, to: newline) > limit / 2 {
            head = String(head[..<newline])
        }
        let omitted = text.count - head.count
        let note = oldResult
            ? L("…（\(omitted) 文字省略。必要なら読み直す）", "… (\(omitted) chars omitted; read again if needed)")
            : L("…（\(omitted) 文字省略）", "… (\(omitted) chars omitted)")
        return head + "\n" + note
    }

    /// ツールの引数の長い文字列（書いたファイルの中身など）を省く
    static func shortenArguments(_ call: ToolCall, limit: Int) -> ToolCall {
        guard let pairs = call.arguments.objectValue else { return call }
        var changed = false
        let shortened = pairs.map { key, value -> (String, JSONValue) in
            guard case .string(let text) = value, text.count > limit else { return (key, value) }
            changed = true
            return (key, .string(String(text.prefix(limit / 2)) + L("…（\(text.count) 文字）", "… (\(text.count) chars)")))
        }
        guard changed else { return call }
        var copy = call
        copy.arguments = .object(shortened)
        return copy
    }
}
