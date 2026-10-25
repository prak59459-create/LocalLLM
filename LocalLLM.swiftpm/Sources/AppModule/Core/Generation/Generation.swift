import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

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
///
/// 語彙は 15 万語以上あるので、全部を並べ替えると最適化なしのビルドでは 1 トークンに数百 ms かかる。
/// そこで「上位 K 個がちょうど入るしきい値」を数えながら二分探索し、しきい値以上の候補だけを取り出して並べる。
/// 数える・取り出す処理は iPad では vDSP（Accelerate）で行う。
struct TokenSampler {
    let parameters: GenerationParameters
    /// top-k が無効（0）のときに top-p の前に残す候補の上限
    static let maxCandidates = 512
    /// 最大値からこれ以上（温度で割ったあと）離れた候補は確率が exp(-30) 未満なので捨てる
    static let cutoff: Float = 30

    private var seeded: SplitMix64?
    private var system = SystemRandomNumberGenerator()
    private var scratch = LogitsScratch()

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

        // 2. 最大値（NaN は無視）
        guard let top = LogitsMath.argmax(logits) else { return 0 }   // 全部 NaN / -inf
        if parameters.temperature <= 0 || parameters.topK == 1 { return top.index }

        // 3. 上位の候補だけを取り出す
        let temperature = Float(parameters.temperature)
        let limit = parameters.topK > 0 ? parameters.topK : Self.maxCandidates
        let lowest = top.value - Self.cutoff * temperature
        var candidates = scratch.candidates(logits, atLeast: lowest, maxValue: top.value, keep: limit)
        if candidates.isEmpty { return top.index }
        candidates.sort { $0.logit != $1.logit ? $0.logit > $1.logit : $0.index < $1.index }
        if candidates.count > limit { candidates.removeSubrange(limit...) }

        // 4. 確率にする（Double で計算）
        var probabilities = candidates.map { exp(Double(($0.logit - top.value) / temperature)) }
        var total = probabilities.reduce(0, +)
        guard total > 0, total.isFinite else { return top.index }
        for index in probabilities.indices { probabilities[index] /= total }

        // 5. min-p（いちばん高い確率 × minP 未満を捨てる）
        var keep = probabilities.count
        if parameters.minP > 0 {
            let threshold = probabilities[0] * parameters.minP
            keep = max(1, probabilities.prefix { $0 >= threshold }.count)
        }

        // 6. top-p（確率の高い順に足して topP に達するまで）
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

        // 7. 残った候補から確率に比例して選ぶ
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

/// logits の高速な集計（iPad では vDSP、それ以外は素朴なループ。どちらも同じ結果になる）。
enum LogitsMath {
    /// 最大値とその位置（同じ値なら小さい位置）。NaN は無視。すべて NaN / -inf なら nil。
    static func argmax(_ logits: [Float]) -> (index: Int, value: Float)? {
        logits.withUnsafeBufferPointer { buffer -> (index: Int, value: Float)? in
            guard let base = buffer.baseAddress, buffer.count > 0 else { return nil }
            #if canImport(Accelerate)
            var sum: Float = 0
            vDSP_sve(base, 1, &sum, vDSP_Length(buffer.count))
            if !sum.isNaN {
                var value: Float = 0
                var index: vDSP_Length = 0
                vDSP_maxvi(base, 1, &value, &index, vDSP_Length(buffer.count))
                return value > -Float.infinity ? (Int(index), value) : nil
            }
            #endif
            var best = -1
            var value = -Float.infinity
            var i = 0
            while i < buffer.count {
                if base[i] > value {
                    value = base[i]
                    best = i
                }
                i += 1
            }
            return best >= 0 ? (best, value) : nil
        }
    }

    /// threshold 以上の要素の数
    static func count(_ base: UnsafePointer<Float>, _ n: Int, atLeast threshold: Float,
                      scratch: UnsafeMutablePointer<Float>) -> Int {
        #if canImport(Accelerate)
        var t = threshold
        var one: Float = 1
        vDSP_vthrsc(base, 1, &t, &one, scratch, 1, vDSP_Length(n))
        var sum: Float = 0
        vDSP_sve(scratch, 1, &sum, vDSP_Length(n))
        return (Int(sum.rounded()) + n) / 2
        #else
        var result = 0
        var i = 0
        while i < n {
            if base[i] >= threshold { result += 1 }
            i += 1
        }
        return result
        #endif
    }
}

/// 候補の取り出しに使う作業領域（語彙数ぶん。サンプラーごとに使い回す）
struct LogitsScratch {
    private var buffer: [Float] = []
    private var ramp: [Float] = []

    /// threshold を二分探索して、上位 keep 個（と同点）が入る範囲の候補を返す（並べてはいない）。
    /// atLeast より小さい要素は候補にしない。
    mutating func candidates(_ logits: [Float], atLeast lowest: Float, maxValue: Float,
                             keep: Int) -> [(index: Int, logit: Float)] {
        let n = logits.count
        if buffer.count != n {
            buffer = [Float](repeating: 0, count: n)
            ramp = (0..<n).map { Float($0) }
        }
        return logits.withUnsafeBufferPointer { values -> [(index: Int, logit: Float)] in
            buffer.withUnsafeMutableBufferPointer { work -> [(index: Int, logit: Float)] in
                guard let base = values.baseAddress, let scratch = work.baseAddress else { return [] }
                let (threshold, found) = Self.threshold(base, n, lowest: lowest, maxValue: maxValue, keep: keep,
                                                        scratch: scratch)
                return Self.collect(base, n, atLeast: threshold, expected: found, scratch: scratch, ramp: ramp)
            }
        }
    }

    /// count(≥ t) が keep 以上で、なるべく keep に近いしきい値 t と、その count。
    /// 上位 keep 個はかならず含まれるので、どちらの方法でも並べたあとの結果は同じになる。
    private static func threshold(_ base: UnsafePointer<Float>, _ n: Int, lowest: Float, maxValue: Float, keep: Int,
                                  scratch: UnsafeMutablePointer<Float>) -> (Float, Int) {
        #if canImport(Accelerate)
        // vDSP で数えるのは速いので、二分探索する
        var threshold = lowest
        var found = LogitsMath.count(base, n, atLeast: lowest, scratch: scratch)
        guard found > keep else { return (threshold, found) }
        var lo = lowest, hi = maxValue
        for _ in 0..<24 {
            let mid = (lo + hi) / 2
            if mid <= lo || mid >= hi { break }
            let c = LogitsMath.count(base, n, atLeast: mid, scratch: scratch)
            if c >= keep {
                lo = mid
                threshold = mid
                found = c
                if c <= keep * 2 { break }
            } else {
                hi = mid
            }
        }
        return (threshold, found)
        #else
        // ループは 1 回にしたいので、ヒストグラムで決める（語彙が少なければ全部を候補に）
        if n <= keep { return (lowest, LogitsMath.count(base, n, atLeast: lowest, scratch: scratch)) }
        let bucketCount = 1024
        var buckets = [Int](repeating: 0, count: bucketCount)
        let span = maxValue - lowest
        guard span > 0 else { return (lowest, LogitsMath.count(base, n, atLeast: lowest, scratch: scratch)) }
        let scale = Float(bucketCount - 1) / span
        buckets.withUnsafeMutableBufferPointer { b in
            var i = 0
            while i < n {
                let x = base[i]
                if x >= lowest {
                    b[min(bucketCount - 1, Int((x - lowest) * scale))] += 1
                }
                i += 1
            }
        }
        var cumulative = 0
        var bucket = bucketCount - 1
        while bucket > 0 {
            cumulative += buckets[bucket]
            if cumulative >= keep { break }
            bucket -= 1
        }
        if bucket == 0 { return (lowest, LogitsMath.count(base, n, atLeast: lowest, scratch: scratch)) }
        // 境目の丸め誤差を避けるため、しきい値の位置で数え直す
        let threshold = lowest + Float(bucket) / scale
        return (threshold, LogitsMath.count(base, n, atLeast: threshold, scratch: scratch))
        #endif
    }

    private static func collect(_ base: UnsafePointer<Float>, _ n: Int, atLeast threshold: Float, expected: Int,
                                scratch: UnsafeMutablePointer<Float>, ramp: [Float]) -> [(index: Int, logit: Float)] {
        guard expected > 0 else { return [] }
        #if canImport(Accelerate)
        // scratch = (x ≥ t ? 2 : 0) を目印にして、位置と値をまとめて詰める
        var t = threshold
        var one: Float = 1
        vDSP_vthrsc(base, 1, &t, &one, scratch, 1, vDSP_Length(n))
        vDSP_vsadd(scratch, 1, &one, scratch, 1, vDSP_Length(n))
        var indices = [Float](repeating: 0, count: expected)
        var values = [Float](repeating: 0, count: expected)
        ramp.withUnsafeBufferPointer { r in
            indices.withUnsafeMutableBufferPointer { out in
                vDSP_vcmprs(r.baseAddress!, 1, scratch, 1, out.baseAddress!, 1, vDSP_Length(n))
            }
        }
        values.withUnsafeMutableBufferPointer { out in
            vDSP_vcmprs(base, 1, scratch, 1, out.baseAddress!, 1, vDSP_Length(n))
        }
        var result: [(index: Int, logit: Float)] = []
        result.reserveCapacity(expected)
        for k in 0..<expected { result.append((Int(indices[k]), values[k])) }
        return result
        #else
        var result: [(index: Int, logit: Float)] = []
        result.reserveCapacity(expected)
        var i = 0
        while i < n {
            if base[i] >= threshold { result.append((i, base[i])) }
            i += 1
        }
        return result
        #endif
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
    /// メモリが足りなくなりそうなので、落ちる前に止めた
    case lowMemory
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
