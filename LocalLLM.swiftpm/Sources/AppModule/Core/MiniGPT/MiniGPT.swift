import Foundation

/// モデルの大きさのひな形
enum MiniGPTPreset: String, Codable, CaseIterable, Sendable {
    case tiny     // 2 層, 埋め込み 64, 2 ヘッド, 文脈 64
    case small    // 4 層, 埋め込み 128, 4 ヘッド, 文脈 128
    case medium   // 6 層, 埋め込み 192, 6 ヘッド, 文脈 128
}

struct MiniGPTConfig: Codable, Hashable, Sendable {
    var vocabSize: Int
    var contextLength: Int
    var embeddingSize: Int
    var layerCount: Int
    var headCount: Int
    var mlpMultiplier: Int = 4

    init(vocabSize: Int, contextLength: Int, embeddingSize: Int, layerCount: Int, headCount: Int, mlpMultiplier: Int = 4) {
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.embeddingSize = embeddingSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.mlpMultiplier = mlpMultiplier
    }

    static func preset(_ preset: MiniGPTPreset, vocabSize: Int) -> MiniGPTConfig {
        switch preset {
        case .tiny: return MiniGPTConfig(vocabSize: vocabSize, contextLength: 64, embeddingSize: 64, layerCount: 2, headCount: 2)
        case .small: return MiniGPTConfig(vocabSize: vocabSize, contextLength: 128, embeddingSize: 128, layerCount: 4, headCount: 4)
        case .medium: return MiniGPTConfig(vocabSize: vocabSize, contextLength: 128, embeddingSize: 192, layerCount: 6, headCount: 6)
        }
    }

    var hiddenSize: Int { embeddingSize * mlpMultiplier }
    var headSize: Int { embeddingSize / max(headCount, 1) }

    var parameterCount: Int {
        let c = embeddingSize, m = hiddenSize
        let perLayer = 2 * c + (c * 3 * c + 3 * c) + (c * c + c) + 2 * c + (c * m + m) + (m * c + c)
        return vocabSize * c + contextLength * c + layerCount * perLayer + 2 * c
    }
}

struct TrainingHyperparameters: Codable, Hashable, Sendable {
    var learningRate: Float = 3e-3
    var minLearningRate: Float = 3e-4
    var batchSize: Int = 16
    var weightDecay: Float = 0.01
    var beta1: Float = 0.9
    var beta2: Float = 0.99
    var epsilon: Float = 1e-8
    var gradientClip: Float = 1.0
    var warmupSteps: Int = 50
    var totalSteps: Int = 2000
    var seed: UInt64 = 1
}

/// 文字（Unicode スカラー）単位の語彙。ID 0 は <unk>。
struct CharVocabulary: Codable, Hashable, Sendable {
    /// ID 1 以降の文字（スカラー値）
    private(set) var symbols: [UInt32]
    private var lookup: [UInt32: Int]

    init(text: String, maxSize: Int = 4096) {
        var counts: [UInt32: Int] = [:]
        for scalar in text.unicodeScalars { counts[scalar.value, default: 0] += 1 }
        let sorted = counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
        symbols = sorted.prefix(max(0, maxSize - 1)).map(\.key)
        lookup = [:]
        for (index, value) in symbols.enumerated() { lookup[value] = index + 1 }
    }

    enum CodingKeys: String, CodingKey { case symbols }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        symbols = try container.decode([UInt32].self, forKey: .symbols)
        lookup = [:]
        for (index, value) in symbols.enumerated() { lookup[value] = index + 1 }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(symbols, forKey: .symbols)
    }

    var size: Int { symbols.count + 1 }

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { lookup[$0.value] ?? 0 }
    }

    func decode(_ ids: [Int]) -> String {
        var view = String.UnicodeScalarView()
        for id in ids {
            if id > 0 && id <= symbols.count, let scalar = Unicode.Scalar(symbols[id - 1]) {
                view.append(scalar)
            } else {
                view.append("\u{FFFD}")
            }
        }
        return String(view)
    }
}

/// 名前付きのパラメータ（値と勾配）
final class MiniGPTParameter {
    let name: String
    let shape: [Int]
    var value: [Float]
    var gradient: [Float]

    init(name: String, shape: [Int], value: [Float]) {
        self.name = name
        self.shape = shape
        self.value = value
        self.gradient = [Float](repeating: 0, count: value.count)
    }

    var count: Int { value.count }
}

enum MiniGPTError: Error, LocalizedError {
    case invalidFile(String)

    var errorDescription: String? {
        switch self {
        case .invalidFile(let detail):
            return L("ミニモデルのファイルを読めません: \(detail)", "Cannot read the mini model file: \(detail)")
        }
    }
}

/// 小さな GPT（GPT-2 と同じ構造）。順伝播と逆伝播を手書きで実装している。
final class MiniGPT {
    let config: MiniGPTConfig
    let parameters: [MiniGPTParameter]
    private let byName: [String: MiniGPTParameter]

    // よく使うパラメータ
    private let wte: MiniGPTParameter
    private let wpe: MiniGPTParameter
    private let lnfWeight: MiniGPTParameter
    private let lnfBias: MiniGPTParameter
    private let layers: [Layer]

    private struct Layer {
        var ln1Weight, ln1Bias, qkvWeight, qkvBias, projWeight, projBias: MiniGPTParameter
        var ln2Weight, ln2Bias, fcWeight, fcBias, mlpProjWeight, mlpProjBias: MiniGPTParameter
    }

    /// 逆伝播のために順伝播の途中結果をとっておく場所（同じ大きさなら使い回す）
    private var cache = MiniGPTActivations()

    init(config: MiniGPTConfig, seed: UInt64) {
        var random = MiniRandom(seed: seed)
        let c = config.embeddingSize, m = config.hiddenSize
        let projectionScale = 0.02 / (2.0 * Double(config.layerCount)).squareRoot()
        func normal(_ count: Int, _ std: Double) -> [Float] {
            (0..<count).map { _ in Float(random.nextNormal() * std) }
        }
        var list: [MiniGPTParameter] = []
        func make(_ name: String, _ shape: [Int], _ values: [Float]) -> MiniGPTParameter {
            let parameter = MiniGPTParameter(name: name, shape: shape, value: values)
            list.append(parameter)
            return parameter
        }
        let wte = make("wte", [config.vocabSize, c], normal(config.vocabSize * c, 0.02))
        let wpe = make("wpe", [config.contextLength, c], normal(config.contextLength * c, 0.02))
        var layers: [Layer] = []
        for index in 0..<config.layerCount {
            let prefix = "h.\(index)."
            let ln1w = make(prefix + "ln1.weight", [c], [Float](repeating: 1, count: c))
            let ln1b = make(prefix + "ln1.bias", [c], [Float](repeating: 0, count: c))
            let qkvw = make(prefix + "attn.qkv.weight", [c, 3 * c], normal(c * 3 * c, 0.02))
            let qkvb = make(prefix + "attn.qkv.bias", [3 * c], [Float](repeating: 0, count: 3 * c))
            let projw = make(prefix + "attn.proj.weight", [c, c], normal(c * c, projectionScale))
            let projb = make(prefix + "attn.proj.bias", [c], [Float](repeating: 0, count: c))
            let ln2w = make(prefix + "ln2.weight", [c], [Float](repeating: 1, count: c))
            let ln2b = make(prefix + "ln2.bias", [c], [Float](repeating: 0, count: c))
            let fcw = make(prefix + "mlp.fc.weight", [c, m], normal(c * m, 0.02))
            let fcb = make(prefix + "mlp.fc.bias", [m], [Float](repeating: 0, count: m))
            let mpw = make(prefix + "mlp.proj.weight", [m, c], normal(m * c, projectionScale))
            let mpb = make(prefix + "mlp.proj.bias", [c], [Float](repeating: 0, count: c))
            layers.append(Layer(ln1Weight: ln1w, ln1Bias: ln1b, qkvWeight: qkvw, qkvBias: qkvb, projWeight: projw, projBias: projb,
                                ln2Weight: ln2w, ln2Bias: ln2b, fcWeight: fcw, fcBias: fcb, mlpProjWeight: mpw, mlpProjBias: mpb))
        }
        let lnfw = make("ln_f.weight", [c], [Float](repeating: 1, count: c))
        let lnfb = make("ln_f.bias", [c], [Float](repeating: 0, count: c))
        self.config = config
        self.parameters = list
        self.byName = Dictionary(uniqueKeysWithValues: list.map { ($0.name, $0) })
        self.wte = wte
        self.wpe = wpe
        self.lnfWeight = lnfw
        self.lnfBias = lnfb
        self.layers = layers
    }

    func parameter(named name: String) -> MiniGPTParameter? {
        byName[name]
    }

    func zeroGradients() {
        for parameter in parameters {
            parameter.gradient.withUnsafeMutableBufferPointer { buffer in
                buffer.initialize(repeating: 0)
            }
        }
    }

    // MARK: - 損失・勾配

    /// 平均交差エントロピー（勾配は計算しない）
    func loss(inputs: [[Int]], targets: [[Int]]) -> Float {
        guard let shape = batchShape(inputs, targets) else { return .nan }
        forward(inputs: inputs, batch: shape.batch, length: shape.length)
        return crossEntropy(targets: targets, batch: shape.batch, length: shape.length, computeGradient: false)
    }

    /// 順伝播 + 逆伝播。勾配を gradient に加算して損失を返す。
    func lossAndGradients(inputs: [[Int]], targets: [[Int]]) -> Float {
        guard let shape = batchShape(inputs, targets) else { return .nan }
        forward(inputs: inputs, batch: shape.batch, length: shape.length)
        let loss = crossEntropy(targets: targets, batch: shape.batch, length: shape.length, computeGradient: true)
        backward(inputs: inputs, batch: shape.batch, length: shape.length)
        return loss
    }

    /// 文脈の次のトークンの logits
    func logits(context: [Int]) -> [Float] {
        var tokens = Array(context.suffix(config.contextLength))
        if tokens.isEmpty { tokens = [0] }
        tokens = tokens.map { min(max($0, 0), config.vocabSize - 1) }
        forward(inputs: [tokens], batch: 1, length: tokens.count)
        let v = config.vocabSize
        let start = (tokens.count - 1) * v
        return Array(cache.logits[start..<(start + v)])
    }

    /// 文章の続きを作る。戻り値は新しく生成したトークンだけ。
    func generate(prompt: [Int], maxNewTokens: Int, temperature: Float, topK: Int, seed: UInt64) -> [Int] {
        var random = MiniRandom(seed: seed)
        var context = prompt.isEmpty ? [0] : prompt
        var generated: [Int] = []
        for _ in 0..<max(0, maxNewTokens) {
            let next = Self.sample(logits(context: context), temperature: temperature, topK: topK, random: &random)
            generated.append(next)
            context.append(next)
        }
        return generated
    }

    static func sample(_ logits: [Float], temperature: Float, topK: Int, random: inout MiniRandom) -> Int {
        var best = 0
        for index in logits.indices where logits[index] > logits[best] { best = index }
        if temperature <= 0 { return best }
        var order = Array(logits.indices)
        if topK > 0 && topK < logits.count {
            order.sort { logits[$0] > logits[$1] }
            order = Array(order.prefix(topK))
        }
        let maxValue = logits[best]
        var weights = order.map { Double(exp((logits[$0] - maxValue) / temperature)) }
        let total = weights.reduce(0, +)
        guard total > 0, total.isFinite else { return best }
        for index in weights.indices { weights[index] /= total }
        let r = random.nextUniform()
        var running = 0.0
        for (position, weight) in weights.enumerated() {
            running += weight
            if r < running { return order[position] }
        }
        return order.last ?? best
    }

    private func batchShape(_ inputs: [[Int]], _ targets: [[Int]]) -> (batch: Int, length: Int)? {
        guard let first = inputs.first, !first.isEmpty, inputs.count == targets.count else { return nil }
        let length = first.count
        guard length <= config.contextLength,
              inputs.allSatisfy({ $0.count == length }), targets.allSatisfy({ $0.count == length }) else { return nil }
        return (inputs.count, length)
    }

    // MARK: - 順伝播

    private func forward(inputs: [[Int]], batch: Int, length: Int) {
        let c = config.embeddingSize, m = config.hiddenSize, v = config.vocabSize
        let rows = batch * length
        cache.prepare(rows: rows, batch: batch, length: length, config: config)

        // 埋め込み
        cache.x[0].withUnsafeMutableBufferPointer { x in
            wte.value.withUnsafeBufferPointer { wte in
                wpe.value.withUnsafeBufferPointer { wpe in
                    for b in 0..<batch {
                        for t in 0..<length {
                            let token = min(max(inputs[b][t], 0), v - 1)
                            MiniVector.add(wte.baseAddress! + token * c, wpe.baseAddress! + t * c,
                                           x.baseAddress! + (b * length + t) * c, c)
                        }
                    }
                }
            }
        }

        for (index, layer) in layers.enumerated() {
            let layerCache = cache.layers[index]
            // a = LN1(x)
            MiniKernels.layerNormForward(input: cache.x[index], weight: layer.ln1Weight.value, bias: layer.ln1Bias.value,
                                         rows: rows, columns: c, output: &layerCache.ln1Out,
                                         mean: &layerCache.ln1Mean, rstd: &layerCache.ln1Rstd)
            // qkv = a · W + b
            MiniKernels.linear(input: layerCache.ln1Out, weight: layer.qkvWeight.value, bias: layer.qkvBias.value,
                               rows: rows, inputSize: c, outputSize: 3 * c, output: &layerCache.qkv)
            // 注意機構
            MiniKernels.attentionForward(qkv: layerCache.qkv, batch: batch, length: length, channels: c, heads: config.headCount,
                                         probabilities: &layerCache.probabilities, output: &layerCache.attention)
            // x1 = x + proj(att)
            MiniKernels.linear(input: layerCache.attention, weight: layer.projWeight.value, bias: layer.projBias.value,
                               rows: rows, inputSize: c, outputSize: c, output: &layerCache.residual)
            MiniKernels.addInPlace(cache.x[index], into: &layerCache.residual)
            // m = LN2(x1)
            MiniKernels.layerNormForward(input: layerCache.residual, weight: layer.ln2Weight.value, bias: layer.ln2Bias.value,
                                         rows: rows, columns: c, output: &layerCache.ln2Out,
                                         mean: &layerCache.ln2Mean, rstd: &layerCache.ln2Rstd)
            // h = GELU(m · Wfc + b)
            MiniKernels.linear(input: layerCache.ln2Out, weight: layer.fcWeight.value, bias: layer.fcBias.value,
                               rows: rows, inputSize: c, outputSize: m, output: &layerCache.fcOut)
            MiniKernels.geluForward(layerCache.fcOut, output: &layerCache.geluOut)
            // x_next = x1 + proj(h)
            MiniKernels.linear(input: layerCache.geluOut, weight: layer.mlpProjWeight.value, bias: layer.mlpProjBias.value,
                               rows: rows, inputSize: m, outputSize: c, output: &cache.x[index + 1])
            MiniKernels.addInPlace(layerCache.residual, into: &cache.x[index + 1])
        }

        // 最後の LayerNorm と、埋め込みを使い回した出力層
        MiniKernels.layerNormForward(input: cache.x[layers.count], weight: lnfWeight.value, bias: lnfBias.value,
                                     rows: rows, columns: c, output: &cache.lnfOut, mean: &cache.lnfMean, rstd: &cache.lnfRstd)
        cache.lnfOut.withUnsafeBufferPointer { f in
            wte.value.withUnsafeBufferPointer { w in
                cache.logits.withUnsafeMutableBufferPointer { out in
                    MiniMath.gemm(transA: false, transB: true, m: rows, n: v, k: c, alpha: 1,
                                  a: f.baseAddress!, lda: c, b: w.baseAddress!, ldb: c, beta: 0, c: out.baseAddress!, ldc: v)
                }
            }
        }
    }

    /// 交差エントロピー。computeGradient なら dlogits（softmax − one-hot）/ 行数 を cache.dLogits に入れる。
    /// cache.logits は書き換わる。
    private func crossEntropy(targets: [[Int]], batch: Int, length: Int, computeGradient: Bool) -> Float {
        let v = config.vocabSize
        let rows = batch * length
        var total: Double = 0
        cache.logits.withUnsafeMutableBufferPointer { logits in
            for r in 0..<rows {
                let row = logits.baseAddress! + r * v
                let target = min(max(targets[r / length][r % length], 0), v - 1)
                let targetLogit = row[target]
                let maxValue = MiniVector.maximum(row, v)
                MiniVector.shift(row, -maxValue, row, v)
                MiniVector.exp(row, row, v)
                let sum = MiniVector.sum(row, v)
                total -= Double(targetLogit - maxValue - log(sum))
                if computeGradient {
                    MiniVector.scale(row, 1 / (sum * Float(rows)), row, v)
                    row[target] -= 1 / Float(rows)
                }
            }
        }
        return Float(total / Double(rows))
    }

    // MARK: - 逆伝播

    private func backward(inputs: [[Int]], batch: Int, length: Int) {
        let c = config.embeddingSize, m = config.hiddenSize, v = config.vocabSize
        let rows = batch * length
        // 出力層: logits = f · wteᵀ（cache.logits には dlogits が入っている）
        var dx = [Float](repeating: 0, count: rows * c)
        var dF = [Float](repeating: 0, count: rows * c)
        cache.logits.withUnsafeBufferPointer { dl in
            wte.value.withUnsafeBufferPointer { w in
                dF.withUnsafeMutableBufferPointer { out in
                    MiniMath.gemm(transA: false, transB: false, m: rows, n: c, k: v, alpha: 1,
                                  a: dl.baseAddress!, lda: v, b: w.baseAddress!, ldb: c, beta: 0, c: out.baseAddress!, ldc: c)
                }
            }
            cache.lnfOut.withUnsafeBufferPointer { f in
                wte.gradient.withUnsafeMutableBufferPointer { g in
                    MiniMath.gemm(transA: true, transB: false, m: v, n: c, k: rows, alpha: 1,
                                  a: dl.baseAddress!, lda: v, b: f.baseAddress!, ldb: c, beta: 1, c: g.baseAddress!, ldc: c)
                }
            }
        }
        MiniKernels.layerNormBackward(dOutput: dF, input: cache.x[layers.count], weight: lnfWeight.value,
                                      mean: cache.lnfMean, rstd: cache.lnfRstd, rows: rows, columns: c,
                                      dInput: &dx, dWeight: &lnfWeight.gradient, dBias: &lnfBias.gradient)

        var dHidden = [Float](repeating: 0, count: rows * m)
        var dSmall = [Float](repeating: 0, count: rows * c)
        var dQKV = [Float](repeating: 0, count: rows * 3 * c)
        for index in stride(from: layers.count - 1, through: 0, by: -1) {
            let layer = layers[index]
            let layerCache = cache.layers[index]
            // x_next = x1 + mlpProj(gelu(fc(ln2(x1))))
            MiniKernels.linearBackward(dOutput: dx, input: layerCache.geluOut, weight: layer.mlpProjWeight.value,
                                       rows: rows, inputSize: m, outputSize: c, dInput: &dHidden,
                                       dWeight: &layer.mlpProjWeight.gradient, dBias: &layer.mlpProjBias.gradient)
            MiniKernels.geluBackward(input: layerCache.fcOut, dOutput: &dHidden)
            MiniKernels.linearBackward(dOutput: dHidden, input: layerCache.ln2Out, weight: layer.fcWeight.value,
                                       rows: rows, inputSize: c, outputSize: m, dInput: &dSmall,
                                       dWeight: &layer.fcWeight.gradient, dBias: &layer.fcBias.gradient)
            // dx1 = dx + LN2 の逆伝播（dx に足し込む）
            MiniKernels.layerNormBackward(dOutput: dSmall, input: layerCache.residual, weight: layer.ln2Weight.value,
                                          mean: layerCache.ln2Mean, rstd: layerCache.ln2Rstd, rows: rows, columns: c,
                                          dInput: &dx, dWeight: &layer.ln2Weight.gradient, dBias: &layer.ln2Bias.gradient)
            // x1 = x + proj(attention(qkv(ln1(x))))
            MiniKernels.linearBackward(dOutput: dx, input: layerCache.attention, weight: layer.projWeight.value,
                                       rows: rows, inputSize: c, outputSize: c, dInput: &dSmall,
                                       dWeight: &layer.projWeight.gradient, dBias: &layer.projBias.gradient)
            MiniKernels.attentionBackward(dOutput: dSmall, qkv: layerCache.qkv, probabilities: layerCache.probabilities,
                                          batch: batch, length: length, channels: c, heads: config.headCount, dQKV: &dQKV)
            MiniKernels.linearBackward(dOutput: dQKV, input: layerCache.ln1Out, weight: layer.qkvWeight.value,
                                       rows: rows, inputSize: c, outputSize: 3 * c, dInput: &dSmall,
                                       dWeight: &layer.qkvWeight.gradient, dBias: &layer.qkvBias.gradient)
            MiniKernels.layerNormBackward(dOutput: dSmall, input: cache.x[index], weight: layer.ln1Weight.value,
                                          mean: layerCache.ln1Mean, rstd: layerCache.ln1Rstd, rows: rows, columns: c,
                                          dInput: &dx, dWeight: &layer.ln1Weight.gradient, dBias: &layer.ln1Bias.gradient)
        }

        // 埋め込みの勾配
        dx.withUnsafeBufferPointer { d in
            wte.gradient.withUnsafeMutableBufferPointer { gte in
                wpe.gradient.withUnsafeMutableBufferPointer { gpe in
                    for b in 0..<batch {
                        for t in 0..<length {
                            let token = min(max(inputs[b][t], 0), v - 1)
                            let row = d.baseAddress! + (b * length + t) * c
                            MiniMath.add(row, into: gte.baseAddress! + token * c, count: c)
                            MiniMath.add(row, into: gpe.baseAddress! + t * c, count: c)
                        }
                    }
                }
            }
        }
    }

    // MARK: - 保存と読み込み

    private struct FileHeader: Codable {
        struct Entry: Codable {
            var name: String
            var shape: [Int]
        }
        var format: String
        var config: MiniGPTConfig
        var parameters: [Entry]
    }

    private static let magic = Array("MINIGPT1".utf8)

    /// 形式: "MINIGPT1" + ヘッダの長さ（UInt32 LE）+ JSON ヘッダ + Float32 LE の重み（パラメータの順）
    func save(to url: URL) throws {
        let header = FileHeader(format: "minigpt-1", config: config,
                                parameters: parameters.map { FileHeader.Entry(name: $0.name, shape: $0.shape) })
        let headerData = try JSONEncoder().encode(header)
        var data = Data(Self.magic)
        var headerLength = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
        data.append(headerData)
        for parameter in parameters {
            parameter.value.withUnsafeBufferPointer { buffer in
                var bits = buffer.map { $0.bitPattern.littleEndian }
                bits.withUnsafeMutableBytes { data.append(contentsOf: $0) }
            }
        }
        try data.write(to: url, options: .atomic)
    }

    static func load(from url: URL) throws -> MiniGPT {
        let data = try Data(contentsOf: url)
        let bytes = [UInt8](data)
        guard bytes.count > magic.count + 4, Array(bytes[0..<magic.count]) == magic else {
            throw MiniGPTError.invalidFile(L("形式が違います", "wrong format"))
        }
        var offset = magic.count
        let headerLength = Int(UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24)
        offset += 4
        guard offset + headerLength <= bytes.count else { throw MiniGPTError.invalidFile(L("ヘッダが壊れています", "broken header")) }
        let header = try JSONDecoder().decode(FileHeader.self, from: Data(bytes[offset..<(offset + headerLength)]))
        offset += headerLength
        let model = MiniGPT(config: header.config, seed: 0)
        guard header.parameters.count == model.parameters.count else {
            throw MiniGPTError.invalidFile(L("パラメータの数が合いません", "parameter count mismatch"))
        }
        for (entry, parameter) in zip(header.parameters, model.parameters) {
            guard entry.name == parameter.name, entry.shape == parameter.shape else {
                throw MiniGPTError.invalidFile(L("パラメータ \(entry.name) の形が合いません", "shape mismatch for \(entry.name)"))
            }
            let byteCount = parameter.count * 4
            guard offset + byteCount <= bytes.count else { throw MiniGPTError.invalidFile(L("データが足りません", "truncated data")) }
            for index in 0..<parameter.count {
                let base = offset + index * 4
                let bits = UInt32(bytes[base]) | UInt32(bytes[base + 1]) << 8 | UInt32(bytes[base + 2]) << 16 | UInt32(bytes[base + 3]) << 24
                parameter.value[index] = Float(bitPattern: bits)
            }
            offset += byteCount
        }
        return model
    }
}

/// 順伝播の途中結果
final class MiniGPTActivations {
    final class LayerCache {
        var ln1Out: [Float] = [], ln1Mean: [Float] = [], ln1Rstd: [Float] = []
        var qkv: [Float] = [], probabilities: [Float] = [], attention: [Float] = [], residual: [Float] = []
        var ln2Out: [Float] = [], ln2Mean: [Float] = [], ln2Rstd: [Float] = []
        var fcOut: [Float] = [], geluOut: [Float] = []
    }

    var x: [[Float]] = []
    var layers: [LayerCache] = []
    var lnfOut: [Float] = [], lnfMean: [Float] = [], lnfRstd: [Float] = []
    var logits: [Float] = []
    private var preparedKey: [Int] = []

    func prepare(rows: Int, batch: Int, length: Int, config: MiniGPTConfig) {
        let key = [rows, batch, length, config.embeddingSize, config.hiddenSize, config.vocabSize, config.layerCount, config.headCount]
        if key == preparedKey { return }
        preparedKey = key
        let c = config.embeddingSize, m = config.hiddenSize
        func zeros(_ count: Int) -> [Float] { [Float](repeating: 0, count: count) }
        x = (0...config.layerCount).map { _ in zeros(rows * c) }
        layers = (0..<config.layerCount).map { _ in
            let cache = LayerCache()
            cache.ln1Out = zeros(rows * c); cache.ln1Mean = zeros(rows); cache.ln1Rstd = zeros(rows)
            cache.qkv = zeros(rows * 3 * c)
            cache.probabilities = zeros(batch * config.headCount * length * length)
            cache.attention = zeros(rows * c); cache.residual = zeros(rows * c)
            cache.ln2Out = zeros(rows * c); cache.ln2Mean = zeros(rows); cache.ln2Rstd = zeros(rows)
            cache.fcOut = zeros(rows * m); cache.geluOut = zeros(rows * m)
            return cache
        }
        lnfOut = zeros(rows * c); lnfMean = zeros(rows); lnfRstd = zeros(rows)
        logits = zeros(rows * config.vocabSize)
    }
}
