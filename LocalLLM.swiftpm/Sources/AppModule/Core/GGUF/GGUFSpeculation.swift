import Foundation

/// 推測デコード（MTP）の 1 回の往復の、受け入れ方。
///
/// 本体がまだ読んでいないトークン first（位置 p）と、先読みが予想した d1 … dk を、本体がまとめて読む（位置 p … p + k）。
/// 本体の logits から 1 つずつ（ふつうの生成と同じ選び方で）選び、予想と同じなら次へ進む。違ったら、そこで選んだトークンを
/// 使って止める。全部当たったら、最後の logits からもう 1 つ選ぶ。選ぶのは毎回本体の logits からなので、
/// 出てくる文章の確率は、先読みを使わないときと変わらない（速さだけが変わる）。
enum GGUFSpeculation {
    /// logits（予想の数 + 1 個）から選んで、予想と比べる。sample(logits, i) は i 番目の位置で選ぶ。
    /// 返すのは、受け入れた予想の数と、選んだトークン（受け入れた予想 + 最後に選んだもの）
    static func accept(drafts: [Int], logits: inout [[Float]], sample: (inout [Float], Int) -> Int) -> (accepted: Int, tokens: [Int]) {
        var tokens: [Int] = []
        var accepted = 0
        while accepted < logits.count {
            let token = sample(&logits[accepted], accepted)
            tokens.append(token)
            guard accepted < drafts.count, token == drafts[accepted] else { break }
            accepted += 1
        }
        return (accepted, tokens)
    }

    /// CPU の本体と先読みで 1 往復する（テストと、GPU が使えないとき用）。
    /// first は本体がまだ読んでいないトークン、hidden はその前の位置の本体の隠れ状態。
    /// 返すのは選んだトークン（1 個以上）と、次の往復に渡す隠れ状態。本体の KV キャッシュは、選んだ最後のトークンの手前まで進む
    static func round(target: GGUFModel, drafter: GGUFDrafter, first: Int, hidden: [Float], count: Int,
                      sample: (inout [Float], Int) -> Int) throws -> (tokens: [Int], hidden: [Float]) {
        let position = target.length
        let drafts = drafter.draft(after: first, hidden: hidden, position: position, count: count, target: target)
        var (logits, states) = try target.evaluateAll([first] + drafts)
        let (accepted, tokens) = accept(drafts: drafts, logits: &logits, sample: sample)
        target.truncate(to: position + accepted + 1)
        return (tokens, states[accepted])
    }
}
