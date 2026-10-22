import Foundation

/// 生成の設定。
struct GenerationParameters: Codable, Hashable, Sendable {
    /// 0 なら毎回いちばん確率の高いトークンを選ぶ（貪欲法）
    var temperature: Double = 0.7
    /// 0 なら無効
    var topK: Int = 40
    /// 1.0 なら無効
    var topP: Double = 0.9
    /// 0 なら無効
    var minP: Double = 0.0
    /// 1.0 なら無効
    var repetitionPenalty: Double = 1.1
    /// 繰り返しペナルティを掛ける直近のトークン数
    var repetitionWindow: Int = 64
    var maxNewTokens: Int = 1024
    /// nil なら毎回ランダム
    var seed: UInt64? = nil

    static let `default` = GenerationParameters()

    init() {}

    // 保存した設定に無い項目は既定値にする（あとから項目を増やしても読めるように）
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = GenerationParameters()
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? defaults.temperature
        topK = try container.decodeIfPresent(Int.self, forKey: .topK) ?? defaults.topK
        topP = try container.decodeIfPresent(Double.self, forKey: .topP) ?? defaults.topP
        minP = try container.decodeIfPresent(Double.self, forKey: .minP) ?? defaults.minP
        repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? defaults.repetitionPenalty
        repetitionWindow = try container.decodeIfPresent(Int.self, forKey: .repetitionWindow) ?? defaults.repetitionWindow
        maxNewTokens = try container.decodeIfPresent(Int.self, forKey: .maxNewTokens) ?? defaults.maxNewTokens
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
    }
}

/// 再現できる乱数（SplitMix64）
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// logits（語彙ごとのスコア）から次のトークンを選ぶ。
struct TokenSampler {
    let parameters: GenerationParameters
    private var seeded: SplitMix64?
    private var system = SystemRandomNumberGenerator()

    init(parameters: GenerationParameters) {
        self.parameters = parameters
        seeded = parameters.seed.map { SplitMix64(seed: $0) }
    }

    /// - Parameter previousTokens: 繰り返しペナルティ用（直近 repetitionWindow 個だけ使う）
    mutating func sample(logits: inout [Float], previousTokens: [Int]) -> Int {
        let count = logits.count
        guard count > 0 else { return 0 }

        // 1. 繰り返しペナルティ（Hugging Face と同じ: 正なら割る、負なら掛ける）
        let penalty = Float(parameters.repetitionPenalty)
        if penalty != 1, penalty > 0, parameters.repetitionWindow > 0 {
            var seen = Set<Int>()
            for token in previousTokens.suffix(parameters.repetitionWindow) where token >= 0 && token < count {
                guard seen.insert(token).inserted else { continue }
                let value = logits[token]
                logits[token] = value > 0 ? value / penalty : value * penalty
            }
        }

        // 2. 最大値を探す（NaN は無視）
        var best = -1
        var maxValue = -Float.infinity
        logits.withUnsafeBufferPointer { buffer in
            for index in 0..<count {
                let value = buffer[index]
                if value > maxValue {
                    maxValue = value
                    best = index
                }
            }
        }
        if best < 0 { return 0 }   // 全部 NaN / -inf
        if parameters.temperature <= 0 || parameters.topK == 1 { return best }

        // 3. 温度を掛けたあと、最大値から大きく離れたものは最初から候補にしない（全語彙を並べ替えないため）
        let temperature = Float(parameters.temperature)
        let cutoff: Float = 30   // exp(-30) ≈ 1e-13 は無視できる
        var candidates: [(index: Int, logit: Float)] = []
        candidates.reserveCapacity(256)
        logits.withUnsafeBufferPointer { buffer in
            for index in 0..<count {
                let value = buffer[index]
                guard value.isFinite else { continue }
                let scaled = (value - maxValue) / temperature
                if scaled >= -cutoff { candidates.append((index, scaled)) }
            }
        }
        if candidates.isEmpty { return best }
        candidates.sort { $0.logit != $1.logit ? $0.logit > $1.logit : $0.index < $1.index }

        // 4. top-k
        if parameters.topK > 0 && candidates.count > parameters.topK {
            candidates.removeSubrange(parameters.topK...)
        }

        // 5. 確率にする（Double で計算）
        var probabilities = candidates.map { exp(Double($0.logit)) }
        var total = probabilities.reduce(0, +)
        guard total > 0, total.isFinite else { return best }
        for index in probabilities.indices { probabilities[index] /= total }

        // 6. min-p（いちばん高い確率 × minP 未満を捨てる）
        var keep = probabilities.count
        if parameters.minP > 0 {
            let threshold = probabilities[0] * parameters.minP
            keep = max(1, probabilities.prefix { $0 >= threshold }.count)
        }

        // 7. top-p（確率の高い順に足して topP に達するまで）
        if parameters.topP > 0 && parameters.topP < 1 {
            var cumulative = 0.0
            var cut = 0
            for index in 0..<keep {
                cumulative += probabilities[index]
                cut = index + 1
                if cumulative >= parameters.topP { break }
            }
            keep = max(1, cut)
        }

        // 8. 残った候補から確率に比例して選ぶ
        total = probabilities.prefix(keep).reduce(0, +)
        let random = Double(nextRandom() >> 11) / Double(1 << 53) * total
        var running = 0.0
        for index in 0..<keep {
            running += probabilities[index]
            if random < running { return candidates[index].index }
        }
        return candidates[keep - 1].index
    }

    private mutating func nextRandom() -> UInt64 {
        if seeded != nil { return seeded!.next() }
        return system.next()
    }
}

/// トークンを 1 つずつ足していき、確定した文字列だけを返す。
/// UTF-8 の途中（日本語や絵文字が複数のトークンに分かれる）では文字化けした「�」を出さない。
final class StreamingDetokenizer {
    private let tokenizer: TextTokenizer
    private let skipSpecialTokens: Bool
    private(set) var tokens: [Int] = []
    private(set) var text: String = ""
    private var prefixOffset = 0
    private var readOffset = 0

    init(tokenizer: TextTokenizer, skipSpecialTokens: Bool = true) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
    }

    /// 新しく確定した部分（空文字のこともある）
    func append(_ token: Int) -> String {
        tokens.append(token)
        let previous = tokenizer.decode(Array(tokens[prefixOffset..<readOffset]), skipSpecialTokens: skipSpecialTokens)
        let current = tokenizer.decode(Array(tokens[prefixOffset...]), skipSpecialTokens: skipSpecialTokens)
        let previousScalars = Array(previous.unicodeScalars)
        let currentScalars = Array(current.unicodeScalars)
        guard currentScalars.count > previousScalars.count, currentScalars.last != "\u{FFFD}" else {
            return ""
        }
        // 共通部分の後ろだけを出す（通常は previous がそのまま先頭に含まれる）
        var common = 0
        while common < previousScalars.count && currentScalars[common] == previousScalars[common] { common += 1 }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: currentScalars[common...])
        let newText = String(view)
        prefixOffset = readOffset
        readOffset = tokens.count
        text += newText
        return newText
    }

    /// 残りを全部出す（最後に呼ぶ）
    func flush() -> String {
        guard readOffset < tokens.count else { return "" }
        let previous = tokenizer.decode(Array(tokens[prefixOffset..<readOffset]), skipSpecialTokens: skipSpecialTokens)
        let current = tokenizer.decode(Array(tokens[prefixOffset...]), skipSpecialTokens: skipSpecialTokens)
        let previousScalars = Array(previous.unicodeScalars)
        let currentScalars = Array(current.unicodeScalars)
        var common = 0
        while common < previousScalars.count && common < currentScalars.count && currentScalars[common] == previousScalars[common] {
            common += 1
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: currentScalars[common...])
        let newText = String(view)
        prefixOffset = readOffset
        readOffset = tokens.count
        text += newText
        return newText
    }
}

/// 停止文字列の検出。停止文字列の先頭と一致しかけている部分は、確定するまで出力を保留する。
struct StopSequenceMatcher {
    private let stops: [[Unicode.Scalar]]
    private var pending: [Unicode.Scalar] = []

    init(stopSequences: [String]) {
        stops = stopSequences.map { Array($0.unicodeScalars) }.filter { !$0.isEmpty }
    }

    mutating func feed(_ chunk: String) -> (emit: String, stopped: Bool) {
        pending.append(contentsOf: chunk.unicodeScalars)
        guard !stops.isEmpty else { return (flush(), false) }
        // 停止文字列が丸ごと含まれていれば、その手前までを出して止める
        var earliest: Int?
        for stop in stops {
            if let index = Self.find(stop, in: pending), index < (earliest ?? Int.max) { earliest = index }
        }
        if let earliest {
            let text = Self.string(pending[..<earliest])
            pending.removeAll()
            return (text, true)
        }
        // 末尾が停止文字列の先頭と一致していれば、その分は保留する
        var hold = 0
        for stop in stops {
            let limit = min(stop.count - 1, pending.count)
            if limit <= 0 { continue }
            for length in stride(from: limit, through: 1, by: -1) where Array(pending.suffix(length)) == Array(stop.prefix(length)) {
                hold = max(hold, length)
                break
            }
        }
        let emitCount = pending.count - hold
        let text = Self.string(pending[..<emitCount])
        pending.removeFirst(emitCount)
        return (text, false)
    }

    /// 保留していた分を出す
    mutating func flush() -> String {
        let text = Self.string(pending[...])
        pending.removeAll()
        return text
    }

    private static func find(_ needle: [Unicode.Scalar], in haystack: [Unicode.Scalar]) -> Int? {
        guard needle.count <= haystack.count else { return nil }
        for start in 0...(haystack.count - needle.count) where haystack[start] == needle[0] {
            if Array(haystack[start..<(start + needle.count)]) == needle { return start }
        }
        return nil
    }

    private static func string(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }
}

/// 停止トークン ID の集合を作る。
enum StopTokenResolver {
    static func resolve(tokenizer: TextTokenizer, eosToken: String?, generationConfigJSON: Data?,
                        configJSON: Data?, extraTokens: [String]) -> Set<Int> {
        var ids = Set<Int>()
        if let eosToken, let id = tokenizer.tokenID(for: eosToken) { ids.insert(id) }
        for data in [generationConfigJSON, configJSON].compactMap({ $0 }) {
            guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let value = object["eos_token_id"] as? Int {
                ids.insert(value)
            } else if let values = object["eos_token_id"] as? [Int] {
                ids.formUnion(values)
            } else if let values = object["eos_token_id"] as? [Any] {
                ids.formUnion(values.compactMap { $0 as? Int })
            }
        }
        let common = ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>",
                      "<|end_of_text|>", "[|endofturn|]", "<｜end▁of▁sentence｜>"]
        for token in extraTokens + common {
            if let id = tokenizer.tokenID(for: token) { ids.insert(id) }
        }
        return ids
    }
}

/// 生成を止めた理由
enum StopReason: String, Codable, Sendable {
    case endOfTurn, maxTokens, contextFull, cancelled, error
}

/// 生成の統計（画面に表示する）
struct GenerationStats: Codable, Hashable, Sendable {
    var promptTokens: Int = 0
    /// KV キャッシュを再利用できたトークン数
    var reusedPromptTokens: Int = 0
    var generatedTokens: Int = 0
    var prefillSeconds: Double = 0
    var decodeSeconds: Double = 0
    var stopReason: StopReason? = nil

    var prefillTokensPerSecond: Double {
        let computed = promptTokens - reusedPromptTokens
        return prefillSeconds > 0 ? Double(computed) / prefillSeconds : 0
    }

    var decodeTokensPerSecond: Double {
        decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0
    }
}
