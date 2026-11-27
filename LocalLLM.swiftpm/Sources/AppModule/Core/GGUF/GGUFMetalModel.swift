#if canImport(Metal)
import Foundation
import Metal

/// シェーダーに渡す値（GGUFMetalSource の Args と同じ並び。すべて 4 バイト）
struct GGUFMetalArgs {
    var rows: UInt32 = 0
    var columns: UInt32 = 0
    var count: UInt32 = 0
    var inOffset: UInt32 = 0
    var inStride: UInt32 = 0
    var outOffset: UInt32 = 0
    var outStride: UInt32 = 0
    var weightSkip: UInt32 = 0
    var flags: UInt32 = 0
    var heads: UInt32 = 0
    var kvHeads: UInt32 = 0
    var size: UInt32 = 0
    var total: UInt32 = 0
    var start: UInt32 = 0
    var window: UInt32 = 0
    var pairs: UInt32 = 0
    var epsilon: Float = 0
    var scale: Float = 0
    var softcap: Float = 0
    /// 足し算の倍率など（計算ごとの意味は GGUFMetalSource を見る）
    var extra: Float = 0
    var auxOffset: UInt32 = 0
    var auxStride: UInt32 = 0
}

enum GGUFMetalError: Error, LocalizedError, Equatable {
    /// GPU で計算する用意ができない（GPU が無い・この形のモデルには対応していない など）
    case unavailable(String)
    /// 計算の途中で GPU が失敗した（アプリが裏に回った など）
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let detail):
            return L("GPU で計算できません: \(detail)", "Cannot compute on the GPU: \(detail)")
        case .failed(let detail):
            return L("GPU の計算が失敗しました: \(detail)", "GPU computation failed: \(detail)")
        }
    }
}

/// GGUF から変換した重みで、トランスフォーマーの計算を GPU（Metal）で行う。計算の流れは GGUFModel（CPU）と同じ。
///
/// - 重みは保存ファイルをメモリに対応づけたものを、そのまま GPU から読む（コピーしない）。
/// - 1 回の計算（1 トークン、または読み込みの 1 まとまり）を 1 つの命令列にして GPU に渡し、終わるのを待つ。
/// - 埋め込み・層ごとの入力の表を引くのと、RoPE の表を作るのは CPU で行う。
/// - スレッドセーフではないので、1 つの場所から順番に使う。
final class GGUFMetalModel: GGUFEvaluator {
    typealias Kernel = GGUFMetalSource.Kernel
    private typealias Flag = GGUFMetalSource.Flag

    /// GPU から読む重み
    private struct Weights {
        var buffer: MTLBuffer
        /// バインドする位置（バイト）
        var offset: Int
        /// そこから重みの先頭までのずれ（バイト）
        var skip: Int
        var rows: Int
        var columns: Int
        var isInt8: Bool
        /// vectors の中の位置（Float の個数）。無ければ nil
        var scales: Int?
        var bias: Int?
        /// 4・5 ビット（GGUFGroupQuant）のときの、5 ビット目の面（5 ビットだけ）と倍率・下限の面
        var groupBits = 0
        var high: (buffer: MTLBuffer, offset: Int)? = nil
        var params: (buffer: MTLBuffer, offset: Int)? = nil
    }

    /// 正規化の重み（vectors の中の位置）
    private struct Norm {
        var weight: Int?
        var bias: Int?
        /// ヘッドごとに違う重みのときのヘッド数（1 なら全部の行で同じ）
        var heads = 1
    }

    private struct Layer {
        var attentionNorm: Norm?
        var query: Weights
        var key: Weights?
        var value: Weights?
        var output: Weights
        var queryNorm: Norm?
        var keyNorm: Norm?
        var postAttentionNorm: Norm?
        var feedForwardNorm: Norm?
        var gate: Weights?
        var up: Weights
        var down: Weights
        var postFeedForwardNorm: Norm?
        var inputGate: Weights?
        var projection: Weights?
        var perLayerPostNorm: Norm?
        var outputScale: Float
        /// 短い畳み込みの重み（LFM2。vectors の中の位置）
        var convolutionKernel: Int?
    }

    /// GPU に渡したメモリの範囲（同じ範囲を二重に渡さないため）
    private struct Window {
        var start: UInt
        var end: UInt
        var buffer: MTLBuffer
    }

    let config: GGUFModelConfig
    let maxContext: Int
    private(set) var length = 0
    /// 重みをコピーせずに GPU へ渡せたか（false のときはコピーしている）
    private(set) var usesMappedWeights = true
    /// 最後の計算で GPU が動いていた時間（秒。速さを調べる用）
    private(set) var lastGPUSeconds = 0.0
    /// 最後に logits を計算した位置（受け入れたあとは、最後に受け入れた位置）の、出力層の前の隠れ状態（先読みの入力）
    private(set) var lastHidden: [Float] = []
    /// 先読み用のモデル（つないだときだけ）と、まとめて確かめた結果
    private var drafter: Drafter?
    private var verifiedLogitsBuffer: MTLBuffer?
    private var verifiedHidden: [[Float]] = []
    private var verifyStart = 0
    let deviceName: String

    private let store: GGUFWeightStore
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var pipelines: [Kernel: MTLComputePipelineState] = [:]

    private var windows: [Window] = []
    private var copiedBytes = 0
    /// 正規化の重み・バイアス・行ごとの倍率をまとめたもの
    private var pool: [Float] = [0, 0, 0, 0]
    private var vectors: MTLBuffer

    private let shapes: [GGUFModelConfig.Layer]
    private var layers: [Layer] = []
    private let globals: GGUFGlobalWeights
    private var outputWeights: Weights?
    private var outputNorm = Norm()
    private var perLayerProjection: Weights?
    private var perLayerProjectionNorm = Norm()

    private let fullRope: GGUFRope.Table
    private let slidingRope: GGUFRope.Table

    /// 1 回の命令列で計算するトークン数の上限
    let batchCapacity: Int
    private let hidden: MTLBuffer, normed: MTLBuffer, normed2: MTLBuffer, queries: MTLBuffer, attended: MTLBuffer
    private let projected: MTLBuffer, feedForwardOut: MTLBuffer
    private let gateBuffer: MTLBuffer, upBuffer: MTLBuffer, logits: MTLBuffer
    private let rotation: MTLBuffer, slidingRotation: MTLBuffer
    /// 層ごとの入力（Gemma 4）: [トークン][層][大きさ] と、トークンごとの表、1 層ぶんの作業用
    private let perLayerInputs: MTLBuffer, perLayerTokens: MTLBuffer, perLayerWork: MTLBuffer
    private var scores: MTLBuffer
    private var scoresCapacity: Int

    /// 層ごとの KV キャッシュ: [位置][kvHead][headSize]。借りる層は nil。足りなくなったら大きくする
    private var keys: [MTLBuffer?] = []
    private var values: [MTLBuffer?] = []
    private var cacheCapacity = 0

    /// スコア用のメモリの上限（Float の個数）
    private static let scoresBudget = 16 * 1024 * 1024
    /// 最初に用意する KV キャッシュのトークン数
    private static let initialCache = 512

    static var isAvailable: Bool { MTLCreateSystemDefaultDevice() != nil }

    init(store: GGUFWeightStore, maxContext: Int? = nil, device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else { throw GGUFMetalError.unavailable("no Metal device") }
        guard let queue = device.makeCommandQueue() else { throw GGUFMetalError.unavailable("command queue") }
        let config = store.config
        let shapes = config.layers
        // Gated DeltaNet の層・出力のゲートは、まだ GPU では計算しない（CPU で動かす）
        if config.deltaNet != nil || config.attentionOutputGate { throw GGUFMetalError.unavailable("Gated DeltaNet") }
        // 専門家の混合（MoE）も、まだ GPU では計算しない
        if config.moe != nil { throw GGUFMetalError.unavailable("mixture of experts") }
        // 注意の足し合わせは、ヘッドの大きさ ÷ 4 個のスレッドを 1 つの threadgroup に入れる（256 まで）
        for shape in shapes where shape.headSize % 4 != 0 || shape.headSize > 1024 || shape.headCount % shape.kvHeadCount != 0 {
            throw GGUFMetalError.unavailable("head size \(shape.headSize)")
        }
        self.store = store
        self.device = device
        self.queue = queue
        self.config = config
        self.shapes = shapes
        deviceName = device.name
        self.maxContext = max(8, min(config.contextLength, maxContext ?? config.contextLength))
        let globals = try GGUFGlobalWeights(store: store, config: config)
        self.globals = globals
        fullRope = GGUFRope.table(config: config, sliding: false, factors: globals.ropeFactors)
        slidingRope = GGUFRope.table(config: config, sliding: true, factors: globals.ropeFactors)

        let capacity = Self.batchCapacity(config: config)
        batchCapacity = capacity
        let d = config.embeddingSize, ff = config.maxFeedForwardSize, q = config.maxQuerySize
        let perLayerTotal = config.perLayerInputSize * config.layerCount
        hidden = try Self.floats(device, capacity * d)
        normed = try Self.floats(device, capacity * d)
        normed2 = try Self.floats(device, capacity * d)
        queries = try Self.floats(device, capacity * q)
        attended = try Self.floats(device, capacity * max(q, d))
        projected = try Self.floats(device, capacity * d)
        feedForwardOut = try Self.floats(device, capacity * d)
        gateBuffer = try Self.floats(device, capacity * max(ff, 3 * d))
        upBuffer = try Self.floats(device, capacity * ff)
        logits = try Self.floats(device, max(config.vocabSize, globals.output.rows))
        rotation = try Self.floats(device, capacity * max(fullRope.pairs, 1) * 2)
        slidingRotation = try Self.floats(device, capacity * max(slidingRope.pairs, 1) * 2)
        perLayerInputs = try Self.floats(device, capacity * perLayerTotal)
        perLayerTokens = try Self.floats(device, capacity * perLayerTotal)
        perLayerWork = try Self.floats(device, capacity * config.perLayerInputSize)
        scoresCapacity = 64 * 1024
        scores = try Self.floats(device, scoresCapacity)
        vectors = try Self.floats(device, 4)

        try makePipelines()
        try bindWeights()
        try allocateCache(min(self.maxContext, Self.initialCache))
    }

    private static func floats(_ device: MTLDevice, _ count: Int) throws -> MTLBuffer {
        let bytes = max(count, 4) * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw GGUFMetalError.unavailable("buffer of \(bytes) bytes")
        }
        return buffer
    }

    /// 1 回の命令列が長くなりすぎないように、モデルが大きいほど少なくする
    private static func batchCapacity(config: GGUFModelConfig) -> Int {
        let d = Double(config.embeddingSize)
        let perToken = config.layers.reduce(0.0) { total, shape in
            let q = Double(shape.querySize), kv = Double(shape.kvSize), ff = Double(shape.feedForwardSize)
            return total + d * (q + 2 * kv) + q * d + 3 * d * ff
        }
        let fitting = Int(24e9 / max(perToken, 1))
        return max(4, min(GGUFModel.maxBatch, fitting / 4 * 4))
    }

    // MARK: - 用意

    private func makePipelines() throws {
        // simdgroup の行列の計算（まとめての線形層）は Apple7 以降と Mac2 の GPU だけ。コンパイルできなければ使わない
        var library: MTLLibrary?
        if Self.supportsMatrices(device) {
            do {
                library = try device.makeLibrary(source: GGUFMetalSource.code + GGUFMetalSource.matrixCode, options: nil)
                matrixNote = "compiled"
            } catch {
                matrixNote = "compile failed: \(error.localizedDescription)"
            }
        } else {
            matrixNote = "GPU family without simdgroup matrices"
        }
        let withMatrices = library != nil
        if library == nil {
            do {
                library = try device.makeLibrary(source: GGUFMetalSource.code, options: nil)
            } catch {
                throw GGUFMetalError.unavailable("shader: \(error.localizedDescription)")
            }
        }
        guard let library else { throw GGUFMetalError.unavailable("shader") }
        for kernel in Kernel.allCases {
            if !withMatrices && Kernel.matrixKernels.contains(kernel) { continue }
            guard let function = library.makeFunction(name: kernel.rawValue) else {
                throw GGUFMetalError.unavailable("kernel \(kernel.rawValue)")
            }
            do {
                pipelines[kernel] = try device.makeComputePipelineState(function: function)
            } catch {
                throw GGUFMetalError.unavailable("pipeline \(kernel.rawValue): \(error.localizedDescription)")
            }
        }
    }

    private func bindWeights() throws {
        for index in 0..<config.layerCount {
            layers.append(try bindLayer(index))
        }
        outputNorm = norm(globals.outputNorm, bias: globals.outputNormBias, count: config.embeddingSize)
        outputWeights = try weights(globals.output, bias: nil)
        if let projection = globals.perLayerProjection {
            perLayerProjection = try weights(projection, bias: nil)
            perLayerProjectionNorm = norm(globals.perLayerProjectionNorm, bias: nil, count: config.perLayerInputSize)
        }
        let bytes = pool.count * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(bytes: pool, length: bytes, options: .storageModeShared) else {
            throw GGUFMetalError.unavailable("vector buffer")
        }
        vectors = buffer
        pool = []
    }

    private func bindLayer(_ index: Int) throws -> Layer {
        let source = try GGUFLayerWeights.load(index, store: store, config: config)
        let shape = shapes[index]
        let d = config.embeddingSize, size = shape.headSize
        let qNormCount = config.qkNormIsFull || source.queryNormPerHead ? shape.querySize : size
        let kNormCount = config.qkNormIsFull || source.keyNormPerHead ? shape.kvSize : size
        var queryNorm = source.queryNorm.map { norm($0, bias: nil, count: qNormCount) }
        if source.queryNormPerHead { queryNorm?.heads = shape.headCount }
        var keyNorm = source.keyNorm.map { norm($0, bias: nil, count: kNormCount) }
        if source.keyNormPerHead { keyNorm?.heads = shape.kvHeadCount }
        return Layer(
            attentionNorm: config.hasPreNorms ? norm(source.attentionNorm, bias: source.attentionNormBias, count: d) : nil,
            query: try weights(source.query, bias: source.queryBias),
            key: try source.key.map { try weights($0, bias: source.keyBias) },
            value: try source.value.map { try weights($0, bias: source.valueBias) },
            output: try weights(source.output, bias: source.outputBias),
            queryNorm: queryNorm, keyNorm: keyNorm,
            postAttentionNorm: source.postAttentionNorm.map { norm($0, bias: nil, count: d) },
            feedForwardNorm: source.feedForwardNorm == nil && !config.hasPreNorms ? nil
                : norm(source.feedForwardNorm, bias: source.feedForwardNormBias, count: d),
            gate: try source.gate.map { try weights($0, bias: source.gateBias) },
            up: try weights(source.up, bias: source.upBias),
            down: try weights(source.down, bias: source.downBias),
            postFeedForwardNorm: source.postFeedForwardNorm.map { norm($0, bias: nil, count: d) },
            inputGate: try source.inputGate.map { try weights($0, bias: nil) },
            projection: try source.projection.map { try weights($0, bias: nil) },
            perLayerPostNorm: source.perLayerPostNorm.map { norm($0, bias: nil, count: d) },
            outputScale: source.outputScale,
            convolutionKernel: source.convolutionKernel.map { pooled($0, d * config.convolutionLength) })
    }

    private func norm(_ weight: UnsafePointer<Float>?, bias: UnsafePointer<Float>?, count: Int) -> Norm {
        Norm(weight: weight.map { pooled($0, count) }, bias: bias.map { pooled($0, count) })
    }

    /// 小さなベクトルをまとめ用のメモリに写し、その位置（Float の個数）を返す
    private func pooled(_ pointer: UnsafePointer<Float>, _ count: Int) -> Int {
        let position = pool.count
        pool.append(contentsOf: UnsafeBufferPointer(start: pointer, count: count))
        // 4 個の倍数にそろえる
        while pool.count % 4 != 0 { pool.append(0) }
        return position
    }

    private func weights(_ matrix: GGUFMatrix, bias: UnsafePointer<Float>?) throws -> Weights {
        guard matrix.columns % 4 == 0 else { throw GGUFMetalError.unavailable("matrix width \(matrix.columns)") }
        guard matrix.storage != .raw else { throw GGUFMetalError.unavailable("raw matrix") }
        if matrix.storage.isGrouped { return try groupedWeights(matrix, bias: bias) }
        let isInt8 = matrix.storage == .int8Rows
        let bytes = matrix.rows * matrix.columns * (isInt8 ? 1 : MemoryLayout<Float>.size)
        let (buffer, offset) = try mapped(matrix.data, bytes: bytes)
        let aligned = offset / 16 * 16
        return Weights(buffer: buffer, offset: aligned, skip: offset - aligned, rows: matrix.rows, columns: matrix.columns,
                       isInt8: isInt8,
                       scales: matrix.scales.map { pooled($0, matrix.rows) },
                       bias: bias.map { pooled($0, matrix.rows) })
    }

    /// 4・5 ビットの行列: 3 つの面をそれぞれ GPU から読めるようにする（どの面も保存ファイルで 64 バイトの境目にそろえてある）
    private func groupedWeights(_ matrix: GGUFMatrix, bias: UnsafePointer<Float>?) throws -> Weights {
        guard matrix.columns % GGUFGroupQuant.groupSize == 0, let groupParams = matrix.groupParams else {
            throw GGUFMetalError.unavailable("group matrix")
        }
        let rows = matrix.rows, columns = matrix.columns
        let nibbles = try mapped(matrix.data, bytes: GGUFGroupQuant.nibbleBytes(rows: rows, columns: columns))
        let params = try mapped(groupParams, bytes: GGUFGroupQuant.paramBytes(rows: rows, columns: columns))
        var high: (buffer: MTLBuffer, offset: Int)?
        if matrix.storage == .q5Groups {
            guard let bits = matrix.highBits else { throw GGUFMetalError.unavailable("group high bits") }
            high = try mapped(bits, bytes: GGUFGroupQuant.highBytes(rows: rows, columns: columns))
        }
        // 読む位置は 16 バイト（uint4）・4 バイトの境目にそろっている必要がある
        guard nibbles.1 % 16 == 0, params.1 % 4 == 0, (high?.1 ?? 0) % 4 == 0 else {
            throw GGUFMetalError.unavailable("group matrix alignment")
        }
        var weights = Weights(buffer: nibbles.0, offset: nibbles.1, skip: 0, rows: rows, columns: columns, isInt8: false,
                              scales: nil, bias: bias.map { pooled($0, rows) })
        weights.groupBits = GGUFGroupQuant.bits(matrix.storage)
        weights.high = high.map { ($0.buffer, $0.offset) }
        weights.params = (params.0, params.1)
        return weights
    }

    /// 保存ファイルを対応づけたメモリを、コピーせずに GPU から読めるようにする（ページの境目にそろえた範囲を渡す）
    private func mapped(_ pointer: UnsafeRawPointer, bytes: Int) throws -> (MTLBuffer, Int) {
        let page = UInt(getpagesize())
        let address = UInt(bitPattern: pointer)
        let start = address / page * page
        let end = (address + UInt(bytes) + page - 1) / page * page
        if let window = windows.first(where: { $0.start <= start && end <= $0.end }) {
            return (window.buffer, Int(address - window.start))
        }
        let length = Int(end - start)
        if length <= device.maxBufferLength, let base = UnsafeMutableRawPointer(bitPattern: start),
           let buffer = device.makeBuffer(bytesNoCopy: base, length: length, options: .storageModeShared, deallocator: nil) {
            windows.append(Window(start: start, end: end, buffer: buffer))
            return (buffer, Int(address - start))
        }
        // そのまま渡せないときは、小さなモデルに限ってコピーする
        let limit = Int(ProcessInfo.processInfo.physicalMemory / 8)
        guard copiedBytes + bytes <= limit, let buffer = device.makeBuffer(bytes: pointer, length: max(bytes, 16), options: .storageModeShared) else {
            throw GGUFMetalError.unavailable("weights of \(bytes) bytes")
        }
        copiedBytes += bytes
        usesMappedWeights = false
        return (buffer, 0)
    }

    // MARK: - KV キャッシュ

    private func allocateCache(_ capacity: Int) throws {
        var newKeys: [MTLBuffer?] = [], newValues: [MTLBuffer?] = []
        for (index, shape) in shapes.enumerated() {
            guard shape.kvLayer == index else {
                newKeys.append(nil)
                newValues.append(nil)
                continue
            }
            // 畳み込みの層（LFM2）は、位置ごとの b × x を keys に覚える
            let width = shape.convolution ? config.embeddingSize : shape.kvSize
            let key = try Self.floats(device, capacity * width)
            let value = try Self.floats(device, shape.convolution ? 4 : capacity * width)
            // 今入っている分は引き継ぐ
            let used = min(length, cacheCapacity, capacity) * width * MemoryLayout<Float>.size
            if used > 0, index < keys.count, let oldKey = keys[index], let oldValue = values[index] {
                key.contents().copyMemory(from: oldKey.contents(), byteCount: used)
                if !shape.convolution { value.contents().copyMemory(from: oldValue.contents(), byteCount: used) }
            }
            newKeys.append(key)
            newValues.append(value)
        }
        keys = newKeys
        values = newValues
        cacheCapacity = capacity
    }

    private func reserveCache(_ needed: Int) throws {
        guard needed > cacheCapacity else { return }
        try allocateCache(min(maxContext, max(needed, cacheCapacity * 2)))
    }

    func truncate(to newLength: Int) {
        length = max(0, min(length, newLength))
    }

    /// KV キャッシュを空にして、使っていたメモリを返す（最初の大きさで用意し直す）
    func releaseCache() {
        length = 0
        keys = []
        values = []
        cacheCapacity = 0
        try? allocateCache(min(maxContext, Self.initialCache))
    }

    var computeDescription: String { "GPU (Metal・\(deviceName))" }

    // MARK: - 計算

    @discardableResult
    func evaluate(_ tokens: [Int], batch: Int, progress: ((Int) throws -> Void)?) throws -> [Float] {
        try validate(tokens)
        var offset = 0
        var result: [Float] = []
        while offset < tokens.count {
            let count = batchCount(wanted: min(batch, tokens.count - offset))
            let isLast = offset + count == tokens.count
            let part = Array(tokens[offset..<(offset + count)])
            let output = try forward(part, wantsLogits: isLast)
            if isLast { result = output }
            offset += count
            try progress?(offset)
        }
        return result
    }

    /// 次に 1 回で計算するトークン数（スコア用のメモリに収まる範囲で）
    private func batchCount(wanted: Int) -> Int {
        var count = max(1, min(wanted, batchCapacity))
        while count > 1 && count * config.maxHeadCount * (length + count) > Self.scoresBudget { count /= 2 }
        return count
    }

    /// all: どの位置の logits と隠れ状態も計算する（先読みの確かめ。logits は verifiedLogitsBuffer に入る）
    private func forward(_ tokens: [Int], wantsLogits: Bool, all: Bool = false, profile: Commands.Profile? = nil) throws -> [Float] {
        let n = tokens.count
        let start = length
        if cacheCapacity == 0 { try allocateCache(min(maxContext, Self.initialCache)) }
        try reserveCache(start + n)
        try reserveScores(n * config.maxHeadCount * (start + n))
        embed(tokens, start: start)
        fillRotations(start: start, count: n)

        let commands = try Commands(queue: queue, profile: profile)
        try commands.begin("per-layer inputs")
        encodePerLayerInputs(commands.encoder, count: n)
        for index in layers.indices {
            try encodeLayer(commands, layer: index, start: start, count: n)
        }
        if wantsLogits {
            try commands.begin("output")
            encodeOutput(commands.encoder, count: n, all: all)
        }
        try commands.finish()
        lastGPUSeconds = commands.gpuSeconds
        length = start + n
        guard wantsLogits else { return [] }
        // 出力層の前の隠れ状態（正規化したもの）を覚えておく
        let d = config.embeddingSize
        let rows = normed.contents().assumingMemoryBound(to: Float.self)
        if all {
            verifiedHidden = (0..<n).map { Array(UnsafeBufferPointer(start: rows + $0 * d, count: d)) }
            return []
        }
        lastHidden = Array(UnsafeBufferPointer(start: rows, count: d))
        return readLogits()
    }

    /// 1 回の計算の、部分ごとの GPU の時間（秒）を測る（速さを調べる用。部分ごとに命令列を分けるので、全体は少し遅くなる）
    func profile(_ tokens: [Int]) throws -> [String: Double] {
        try validate(tokens)
        let profile = Commands.Profile()
        _ = try forward(tokens, wantsLogits: true, profile: profile)
        return profile.seconds
    }

    /// GPU に渡す命令列。ふだんは 1 つにまとめる。測るときは部分ごとに分けて、それぞれの GPU の時間を足していく
    private final class Commands {
        final class Profile {
            var seconds: [String: Double] = [:]
        }

        private let queue: MTLCommandQueue
        private let profile: Profile?
        private var buffer: MTLCommandBuffer
        private(set) var encoder: MTLComputeCommandEncoder
        private var section = ""
        private(set) var gpuSeconds = 0.0

        init(queue: MTLCommandQueue, profile: Profile?) throws {
            guard let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeComputeCommandEncoder() else {
                throw GGUFMetalError.failed("command buffer")
            }
            self.queue = queue
            self.profile = profile
            self.buffer = buffer
            self.encoder = encoder
        }

        /// ここから先を name の部分として測る（測らないときは何もしない）
        func begin(_ name: String) throws {
            guard profile != nil, name != section else { return }
            if section.isEmpty {
                section = name
                return
            }
            try finish()
            guard let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeComputeCommandEncoder() else {
                throw GGUFMetalError.failed("command buffer")
            }
            self.buffer = buffer
            self.encoder = encoder
            section = name
        }

        /// 命令列を GPU に渡して、終わるまで待つ
        func finish() throws {
            encoder.endEncoding()
            buffer.commit()
            buffer.waitUntilCompleted()
            if let error = buffer.error { throw GGUFMetalError.failed(error.localizedDescription) }
            let seconds = max(0, buffer.gpuEndTime - buffer.gpuStartTime)
            gpuSeconds += seconds
            profile?.seconds[section, default: 0] += seconds
        }
    }

    private func reserveScores(_ needed: Int) throws {
        guard needed > scoresCapacity else { return }
        let capacity = max(needed, min(Self.scoresBudget, scoresCapacity * 2))
        scores = try Self.floats(device, capacity)
        scoresCapacity = capacity
    }

    /// 埋め込み（と学習した位置、層ごとの入力の表）を CPU で引く
    private func embed(_ tokens: [Int], start: Int) {
        let d = config.embeddingSize
        let x = hidden.contents().assumingMemoryBound(to: Float.self)
        for (index, token) in tokens.enumerated() {
            GGUFKernels.embeddingRow(globals.embedding, row: token, scale: config.embeddingScale, into: x + index * d)
        }
        if let positions = globals.positions {
            var row = [Float](repeating: 0, count: d)
            row.withUnsafeMutableBufferPointer { buffer in
                guard let base = buffer.baseAddress else { return }
                for index in tokens.indices {
                    GGUFKernels.embeddingRow(positions, row: start + index, scale: 1, into: base)
                    MiniVector.add(x + index * d, base, x + index * d, d)
                }
            }
        }
        let size = config.perLayerInputSize
        if size > 0, let table = globals.perLayerEmbedding {
            let total = size * config.layerCount
            let rows = perLayerTokens.contents().assumingMemoryBound(to: Float.self)
            for (index, token) in tokens.enumerated() {
                GGUFKernels.embeddingRow(table, row: token, scale: Float(size).squareRoot(), into: rows + index * total)
            }
        }
    }

    private func readLogits(from buffer: MTLBuffer? = nil, row: Int = 0) -> [Float] {
        let rows = globals.output.rows
        let source = (buffer ?? logits).contents().assumingMemoryBound(to: Float.self) + row * rows
        var result = [Float](repeating: 0, count: max(config.vocabSize, rows))
        result.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            base.update(from: source, count: rows)
            GGUFLogits.finish(base, computedRows: rows, config: config, bias: globals.outputBias)
        }
        if result.count > config.vocabSize { result.removeLast(result.count - config.vocabSize) }
        return result
    }

    /// 位置 start ..< start + count の cos / sin を表に書く: [トークン][組][cos, sin]
    private func fillRotations(start: Int, count: Int) {
        fill(rotation, fullRope, start: start, count: count)
        fill(slidingRotation, slidingRope, start: start, count: count)
    }

    private func fill(_ buffer: MTLBuffer, _ table: GGUFRope.Table, start: Int, count: Int) {
        let pairs = table.pairs
        guard pairs > 0 else { return }
        let output = buffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<count {
            let position = Float(start + index)
            let row = output + index * pairs * 2
            for pair in 0..<pairs {
                let angle: Float = position * table.frequencies[pair]
                row[pair * 2] = Foundation.cos(angle) * table.magnitude
                row[pair * 2 + 1] = Foundation.sin(angle) * table.magnitude
            }
        }
    }

    // MARK: - 命令を並べる

    private typealias Binding = (buffer: MTLBuffer, offset: Int)

    /// 計算を 1 つ並べる。width × height 個のスレッドが動く（範囲の外は計算の側で何もしない）
    private func run(_ kernel: Kernel, _ encoder: MTLComputeCommandEncoder, _ bindings: [Binding], _ arguments: GGUFMetalArgs,
                     width: Int, height: Int = 1) {
        guard let pipeline = pipelines[kernel], width > 0, height > 0 else { return }
        bind(pipeline, encoder, bindings, arguments)
        let group = max(1, pipeline.threadExecutionWidth)
        let groups = MTLSize(width: (width + group - 1) / group, height: height, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
    }

    /// threadgroup の数を決めて並べる（1 つの threadgroup に threads 個のスレッド。GPU の上限までに抑える）
    private func run(_ kernel: Kernel, _ encoder: MTLComputeCommandEncoder, _ bindings: [Binding], _ arguments: GGUFMetalArgs,
                     groups: Int, height: Int = 1, threads: Int) {
        guard let pipeline = pipelines[kernel], groups > 0, height > 0 else { return }
        bind(pipeline, encoder, bindings, arguments)
        let limit = max(32, pipeline.maxTotalThreadsPerThreadgroup / 32 * 32)
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(threads, limit), height: 1, depth: 1))
    }

    private func bind(_ pipeline: MTLComputePipelineState, _ encoder: MTLComputeCommandEncoder, _ bindings: [Binding],
                      _ arguments: GGUFMetalArgs) {
        encoder.setComputePipelineState(pipeline)
        for (index, binding) in bindings.enumerated() {
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: index)
        }
        var values = arguments
        encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: bindings.count)
    }

    /// SIMD グループが 32 スレッドの GPU（Apple の GPU）なら、線形層を SIMD グループで何行かずつ計算する
    private var usesSimdRows: Bool {
        let kernels: [Kernel] = [.linearQ8Vector, .linearF32Vector, .linearQ4Rows, .linearQ5Rows, .linearQ4Batch, .linearQ5Batch,
                                 .gateUpQ4, .gateUpQ5, .linearQ8Rows, .linearQ8Tokens]
        return kernels.allSatisfy { pipelines[$0]?.threadExecutionWidth == 32 }
    }

    /// simdgroup の行列の計算が使える GPU か（iPad・iPhone は A14 以降、Mac は Apple silicon の Mac2）
    static func supportsMatrices(_ device: MTLDevice) -> Bool {
        device.supportsFamily(.apple7) || device.supportsFamily(.mac2)
    }

    /// 行列の計算を用意したときのようす（調べる用）
    private(set) var matrixNote = ""

    /// まとめての線形層を、simdgroup の行列の掛け算で計算できるか（Apple7 以降の GPU）
    var usesMatrixKernels: Bool {
        // 計算の中は 32 スレッドの SIMD グループ 4 つ（128 スレッド）を前提にしている
        Kernel.matrixKernels.allSatisfy { kernel in
            guard let pipeline = pipelines[kernel] else { return false }
            return pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= GGUFMetalSource.matrixThreads
        }
    }

    /// まとめて（2 トークン以上）を行列の掛け算で: threadgroup 1 つで 32 行 × 8 トークン
    private func runMatrix(_ kernel: Kernel, _ encoder: MTLComputeCommandEncoder, _ bindings: [Binding], _ arguments: GGUFMetalArgs,
                           rows: Int, count: Int) {
        let rowsPerGroup = GGUFMetalSource.matrixRows, tokensPerGroup = GGUFMetalSource.matrixTokens
        run(kernel, encoder, bindings, arguments, groups: (rows + rowsPerGroup - 1) / rowsPerGroup,
            height: (count + tokensPerGroup - 1) / tokensPerGroup, threads: GGUFMetalSource.matrixThreads)
    }

    /// 1 行をスレッド全部で計算する計算（正規化など）の、threadgroup のスレッド数
    private static func rowThreads(_ columns: Int) -> Int {
        min(256, max(32, (columns / 4 + 31) / 32 * 32))
    }

    private func vector(_ position: Int?) -> Binding {
        (vectors, (position ?? 0) * MemoryLayout<Float>.size)
    }

    /// 線形層の出力の仕上げ
    private enum Epilogue {
        /// そのまま書く
        case store
        /// 出力に足す（出力 += 値 × scale）
        case add(scale: Float)
        /// gelu(値) × aux（aux の offset から。トークンごとに stride ずつ進む）
        case geluTimes(aux: MTLBuffer, offset: Int, stride: Int)
    }

    private func linear(_ encoder: MTLComputeCommandEncoder, _ weights: Weights, input: MTLBuffer, inOffset: Int = 0, inStride: Int,
                        count: Int, output: MTLBuffer, outOffset: Int = 0, outStride: Int, epilogue: Epilogue = .store) {
        var arguments = GGUFMetalArgs()
        arguments.rows = UInt32(weights.rows)
        arguments.columns = UInt32(weights.columns)
        arguments.count = UInt32(count)
        arguments.inOffset = UInt32(inOffset)
        arguments.inStride = UInt32(inStride)
        arguments.outOffset = UInt32(outOffset)
        arguments.outStride = UInt32(outStride)
        arguments.weightSkip = UInt32(weights.skip)
        if weights.scales != nil { arguments.flags |= Flag.scales }
        if weights.bias != nil { arguments.flags |= Flag.bias }
        var aux = vector(nil)
        switch epilogue {
        case .store:
            break
        case .add(let scale):
            arguments.flags |= Flag.add
            arguments.extra = scale
        case .geluTimes(let buffer, let offset, let stride):
            arguments.flags |= Flag.geluTimesAux
            arguments.auxOffset = UInt32(offset)
            arguments.auxStride = UInt32(stride)
            aux = (buffer, 0)
        }
        let batched = count > 1
        if weights.groupBits > 0, let params = weights.params {
            linearGrouped(encoder, weights, params: params, input: input, output: output, aux: aux, arguments: arguments)
            return
        }
        let bindings: [Binding] = [(weights.buffer, weights.offset), vector(weights.scales), vector(weights.bias), (input, 0),
                                   (output, 0), vector(nil), aux]
        if weights.isInt8 && usesSimdRows && weights.columns % 16 == 0 && weights.skip % 16 == 0 {
            let threads = 32 * GGUFMetalSource.simdGroups
            if batched && usesMatrixKernels {
                runMatrix(.linearQ8Matrix, encoder, bindings, arguments, rows: weights.rows, count: count)
            } else if batched {
                let rows = GGUFMetalSource.batchRows * GGUFMetalSource.simdGroups
                let tokens = GGUFMetalSource.batchTokens
                run(.linearQ8Tokens, encoder, bindings, arguments, groups: (weights.rows + rows - 1) / rows,
                    height: (count + tokens - 1) / tokens, threads: threads)
            } else {
                let rows = GGUFMetalSource.q8Rows * GGUFMetalSource.simdGroups
                run(.linearQ8Rows, encoder, bindings, arguments, groups: (weights.rows + rows - 1) / rows, threads: threads)
            }
            return
        }
        if !batched && usesSimdRows {
            // 1 トークン: 32 スレッドで 1 行
            run(weights.isInt8 ? .linearQ8Vector : .linearF32Vector, encoder, bindings, arguments, groups: weights.rows, threads: 32)
            return
        }
        let kernel: Kernel
        if weights.isInt8 {
            kernel = batched ? .linearQ8Batch : .linearQ8
        } else {
            kernel = batched ? .linearF32Batch : .linearF32
        }
        run(kernel, encoder, bindings, arguments, width: weights.rows, height: batched ? (count + 3) / 4 : 1)
    }

    /// 4・5 ビットの線形層。1 トークンは SIMD グループで 4 行ずつ、まとめてのときはトークン 8 個 × 2 行ずつ
    private func linearGrouped(_ encoder: MTLComputeCommandEncoder, _ weights: Weights, params: Binding, input: MTLBuffer,
                               output: MTLBuffer, aux: Binding, arguments: GGUFMetalArgs) {
        let five = weights.groupBits == 5
        // 5 ビット目の面が無い 4 ビットでも、場所は埋めておく（読まない）
        let high: Binding = weights.high.map { ($0.buffer, $0.offset) } ?? params
        let bindings: [Binding] = [(weights.buffer, weights.offset), params, vector(weights.bias), (input, 0), (output, 0), high, aux]
        let count = Int(arguments.count)
        guard usesSimdRows else {
            run(five ? .linearQ5Vector : .linearQ4Vector, encoder, bindings, arguments, width: weights.rows, height: count)
            return
        }
        let threads = 32 * GGUFMetalSource.simdGroups
        if count > 1 && usesMatrixKernels {
            runMatrix(five ? .linearQ5Matrix : .linearQ4Matrix, encoder, bindings, arguments, rows: weights.rows, count: count)
            return
        }
        if count == 1 {
            let rows = GGUFMetalSource.vectorRows * GGUFMetalSource.simdGroups
            run(five ? .linearQ5Rows : .linearQ4Rows, encoder, bindings, arguments, groups: (weights.rows + rows - 1) / rows,
                threads: threads)
            return
        }
        let rows = GGUFMetalSource.batchRows * GGUFMetalSource.simdGroups
        let tokens = GGUFMetalSource.batchTokens
        run(five ? .linearQ5Batch : .linearQ4Batch, encoder, bindings, arguments, groups: (weights.rows + rows - 1) / rows,
            height: (count + tokens - 1) / tokens, threads: threads)
    }

    /// FFN の gate と up を一度に計算できるか（1 トークン・どちらも同じ 4・5 ビット・バイアスなし）
    private func canFuseGateUp(_ gate: Weights, _ up: Weights) -> Bool {
        usesSimdRows && gate.groupBits > 0 && gate.groupBits == up.groupBits && gate.params != nil && up.params != nil
            && gate.bias == nil && up.bias == nil && gate.rows == up.rows && gate.columns == up.columns
    }

    /// output = act(gate · input) × (up · input)（1 トークン）
    private func gateUp(_ encoder: MTLComputeCommandEncoder, gate: Weights, up: Weights, input: MTLBuffer, output: MTLBuffer) {
        guard let gateParams = gate.params, let upParams = up.params else { return }
        var arguments = GGUFMetalArgs()
        arguments.rows = UInt32(gate.rows)
        arguments.columns = UInt32(gate.columns)
        arguments.count = 1
        switch config.activation {
        case .gelu: arguments.flags |= Flag.gelu
        case .reluSquared: arguments.flags |= Flag.reluSquared
        case .silu: break
        }
        let five = gate.groupBits == 5
        let gateHigh: Binding = gate.high.map { ($0.buffer, $0.offset) } ?? gateParams
        let upHigh: Binding = up.high.map { ($0.buffer, $0.offset) } ?? upParams
        let bindings: [Binding] = [(gate.buffer, gate.offset), gateParams, (up.buffer, up.offset), upParams, (input, 0), (output, 0),
                                   gateHigh, upHigh]
        let rows = GGUFMetalSource.gateUpRows * GGUFMetalSource.simdGroups
        run(five ? .gateUpQ5 : .gateUpQ4, encoder, bindings, arguments, groups: (gate.rows + rows - 1) / rows,
            threads: 32 * GGUFMetalSource.simdGroups)
    }

    /// 正規化（weight が無ければ、重みを掛けない正規化）。add があれば出力に足す: 出力 = (出力 + 値 × branch) × result
    private func normalize(_ encoder: MTLComputeCommandEncoder, _ input: MTLBuffer, inOffset: Int = 0, _ norm: Norm?,
                           into output: MTLBuffer, outOffset: Int = 0, rows: Int, columns: Int, rms: Bool = false,
                           add: (branch: Float, result: Float)? = nil) {
        var arguments = GGUFMetalArgs()
        arguments.rows = UInt32(rows)
        arguments.columns = UInt32(columns)
        arguments.inOffset = UInt32(inOffset)
        arguments.outOffset = UInt32(outOffset)
        arguments.epsilon = config.normEpsilon
        arguments.heads = UInt32(norm?.heads ?? 1)
        if norm?.weight != nil { arguments.flags |= Flag.normWeight }
        if norm?.bias != nil { arguments.flags |= Flag.normBias }
        if config.norm == .layer && !rms { arguments.flags |= Flag.layerNorm }
        if let add {
            arguments.flags |= Flag.add
            arguments.extra = add.branch
            arguments.scale = add.result
        }
        let bindings: [Binding] = [(input, 0), vector(norm?.weight), vector(norm?.bias), (output, 0)]
        run(.rmsNorm, encoder, bindings, arguments, groups: rows, threads: Self.rowThreads(columns))
    }

    /// out = scale × x + second × y（y が nil なら scale × x だけ）
    private func scaleAdd(_ encoder: MTLComputeCommandEncoder, _ x: MTLBuffer, inOffset: Int = 0, scale: Float = 1,
                          plus y: MTLBuffer? = nil, yOffset: Int = 0, second: Float = 1,
                          into out: MTLBuffer, outOffset: Int = 0, count: Int) {
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.inOffset = UInt32(inOffset)
        arguments.outOffset = UInt32(outOffset)
        arguments.total = UInt32(yOffset)
        arguments.scale = scale
        arguments.softcap = second
        if y != nil { arguments.flags |= Flag.second }
        let bindings: [Binding] = [(x, 0), (y ?? x, 0), (out, 0)]
        run(.scaleAdd, encoder, bindings, arguments, width: count)
    }

    /// gate = act(gate) × up（up が nil なら act(gate) だけ）
    private func activate(_ encoder: MTLComputeCommandEncoder, _ gate: MTLBuffer, up: MTLBuffer?, upOffset: Int = 0,
                          upColumns: Int = 0, upStride: Int = 0, activation: GGUFModelConfig.Activation, count: Int) {
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.inOffset = UInt32(upOffset)
        arguments.inStride = UInt32(upStride)
        arguments.columns = UInt32(max(upColumns, 1))
        switch activation {
        case .gelu: arguments.flags |= Flag.gelu
        case .reluSquared: arguments.flags |= Flag.reluSquared
        case .silu: break
        }
        if up == nil { arguments.flags |= Flag.noUp }
        let bindings: [Binding] = [(gate, 0), (up ?? gate, 0)]
        run(.gatedActivation, encoder, bindings, arguments, width: count)
    }

    private func rotate(_ encoder: MTLComputeCommandEncoder, _ buffer: MTLBuffer, offset: Int, heads: Int, count: Int, layer: Int) {
        let shape = shapes[layer]
        guard shape.usesRope else { return }
        let table = shape.sliding ? slidingRope : fullRope
        guard table.pairs > 0 else { return }
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.heads = UInt32(heads)
        arguments.size = UInt32(shape.headSize)
        arguments.pairs = UInt32(table.pairs)
        arguments.outOffset = UInt32(offset)
        if config.ropePairsAreSplit { arguments.flags |= Flag.splitPairs }
        let bindings: [Binding] = [(buffer, 0), (shape.sliding ? slidingRotation : rotation, 0)]
        run(.rope, encoder, bindings, arguments, width: table.pairs, height: count * heads)
    }

    /// q / k の正規化と RoPE（1 つの計算にできるときはまとめる）
    private func normalizeAndRotate(_ encoder: MTLComputeCommandEncoder, _ buffer: MTLBuffer, offset: Int = 0, _ norm: Norm?,
                                    heads: Int, count n: Int, layer: Int) {
        let shape = shapes[layer]
        let table = shape.sliding ? slidingRope : fullRope
        let afterRope = config.qkNormAfterRope
        guard let norm else {
            rotate(encoder, buffer, offset: offset, heads: heads, count: n, layer: layer)
            return
        }
        // ヘッドごとの RMSNorm なら、RoPE と 1 つの計算にする（RoPE が先のときは、回しても大きさが変わらない表に限る）
        if !config.qkNormIsFull && config.norm == .rms && (!afterRope || table.magnitude == 1 || !shape.usesRope) {
            var arguments = GGUFMetalArgs()
            arguments.heads = UInt32(heads)
            arguments.kvHeads = UInt32(norm.heads)
            arguments.size = UInt32(shape.headSize)
            arguments.pairs = UInt32(shape.usesRope ? table.pairs : 0)
            arguments.outOffset = UInt32(offset)
            arguments.epsilon = config.normEpsilon
            if norm.weight != nil { arguments.flags |= Flag.normWeight }
            if config.ropePairsAreSplit { arguments.flags |= Flag.splitPairs }
            if afterRope { arguments.flags |= Flag.ropeFirst }
            let bindings: [Binding] = [(buffer, 0), vector(norm.weight), (shape.sliding ? slidingRotation : rotation, 0)]
            run(.headNormRope, encoder, bindings, arguments, groups: n * heads, threads: Self.rowThreads(shape.headSize))
            return
        }
        if afterRope { rotate(encoder, buffer, offset: offset, heads: heads, count: n, layer: layer) }
        if config.qkNormIsFull {
            normalize(encoder, buffer, inOffset: offset, norm, into: buffer, outOffset: offset, rows: n, columns: heads * shape.headSize)
        } else {
            normalize(encoder, buffer, inOffset: offset, norm, into: buffer, outOffset: offset, rows: n * heads, columns: shape.headSize)
        }
        if !afterRope { rotate(encoder, buffer, offset: offset, heads: heads, count: n, layer: layer) }
    }

    // MARK: - 層

    private func encodeLayer(_ commands: Commands, layer index: Int, start: Int, count n: Int) throws {
        let layer = layers[index]
        let d = config.embeddingSize
        try commands.begin("attention")
        var encoder = commands.encoder
        // 注意の入力
        var attentionInput = hidden
        if let norm = layer.attentionNorm {
            normalize(encoder, hidden, norm, into: normed, rows: n, columns: d)
            attentionInput = normed
        }
        encodeAttention(encoder, layer: index, input: attentionInput, start: start, count: n)
        // 層ごとの入力が無ければ、層の倍率は FFN の足し算のところで掛ける
        let hasPerLayerInput = layer.inputGate != nil && layer.projection != nil
        if config.parallelResidual {
            finishBranch(encoder, projected, postNorm: layer.postAttentionNorm, count: n)
            try commands.begin("feed-forward")
            encoder = commands.encoder
            var input = attentionInput
            if layer.feedForwardNorm?.weight != nil {
                normalize(encoder, hidden, layer.feedForwardNorm, into: normed2, rows: n, columns: d)
                input = normed2
            }
            encodeFeedForward(encoder, layer: index, input: input, count: n)
            finishBranch(encoder, feedForwardOut, postNorm: layer.postFeedForwardNorm, count: n)
            scaleAdd(encoder, hidden, plus: projected, into: hidden, count: n * d)
            scaleAdd(encoder, hidden, plus: feedForwardOut, into: hidden, count: n * d)
            if !hasPerLayerInput && layer.outputScale != 1 {
                scaleAdd(encoder, hidden, scale: layer.outputScale, into: hidden, count: n * d)
            }
        } else {
            addBranch(encoder, projected, postNorm: layer.postAttentionNorm, count: n)
            try commands.begin("feed-forward")
            encoder = commands.encoder
            var input = hidden
            if config.hasPreNorms {
                normalize(encoder, hidden, layer.feedForwardNorm, into: normed, rows: n, columns: d)
                input = normed
            }
            encodeFeedForward(encoder, layer: index, input: input, count: n)
            addBranch(encoder, feedForwardOut, postNorm: layer.postFeedForwardNorm, count: n,
                      resultScale: hasPerLayerInput ? 1 : layer.outputScale)
        }
        if hasPerLayerInput {
            try commands.begin("per-layer input")
            encodePerLayerInput(commands.encoder, layer: index, count: n)
        }
    }

    /// 注意・FFN の出力の仕上げ（後ろの正規化と倍率）
    private func finishBranch(_ encoder: MTLComputeCommandEncoder, _ buffer: MTLBuffer, postNorm: Norm?, count n: Int) {
        let d = config.embeddingSize
        if let postNorm { normalize(encoder, buffer, postNorm, into: buffer, rows: n, columns: d) }
        if config.residualScale != 1 { scaleAdd(encoder, buffer, scale: config.residualScale, into: buffer, count: n * d) }
    }

    /// hidden = (hidden + 仕上げた branch) × resultScale。後ろの正規化があれば、足し算も正規化の中でする
    private func addBranch(_ encoder: MTLComputeCommandEncoder, _ branch: MTLBuffer, postNorm: Norm?, count n: Int,
                           resultScale: Float = 1) {
        let d = config.embeddingSize
        if let postNorm {
            normalize(encoder, branch, postNorm, into: hidden, rows: n, columns: d, add: (config.residualScale, resultScale))
            return
        }
        scaleAdd(encoder, hidden, plus: branch, second: config.residualScale, into: hidden, count: n * d)
        if resultScale != 1 { scaleAdd(encoder, hidden, scale: resultScale, into: hidden, count: n * d) }
    }

    private func attentionArguments(layer: Int, start: Int, count: Int) -> GGUFMetalArgs {
        let shape = shapes[layer]
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.heads = UInt32(shape.headCount)
        arguments.kvHeads = UInt32(shape.kvHeadCount)
        arguments.size = UInt32(shape.headSize)
        arguments.total = UInt32(start + count)
        arguments.start = UInt32(start)
        arguments.window = UInt32(shape.sliding ? config.slidingWindow : 0)
        arguments.scale = config.attentionScale
        arguments.softcap = config.attentionSoftcap
        return arguments
    }

    /// 短い畳み込み（LFM2）: 射影 → bx を記録 → 畳み込み × c → 出力の射影
    private func encodeConvolution(_ encoder: MTLComputeCommandEncoder, layer index: Int, input: MTLBuffer, start: Int, count n: Int) {
        let layer = layers[index]
        guard let history = keys[index], let kernel = layer.convolutionKernel else { return }
        let d = config.embeddingSize
        linear(encoder, layer.query, input: input, inStride: d, count: n, output: gateBuffer, outStride: 3 * d)
        var arguments = GGUFMetalArgs()
        arguments.columns = UInt32(d)
        arguments.count = UInt32(n)
        arguments.start = UInt32(start)
        arguments.size = UInt32(config.convolutionLength)
        let prepare: [Binding] = [(gateBuffer, 0), (history, 0)]
        run(.convolutionPrepare, encoder, prepare, arguments, width: d, height: n)
        let apply: [Binding] = [(gateBuffer, 0), (history, 0), vector(kernel), (attended, 0)]
        run(.convolutionApply, encoder, apply, arguments, width: d, height: n)
        linear(encoder, layer.output, input: attended, inStride: d, count: n, output: projected, outStride: d)
    }

    private func encodeAttention(_ encoder: MTLComputeCommandEncoder, layer index: Int, input: MTLBuffer, start: Int, count n: Int) {
        let layer = layers[index], shape = shapes[index]
        if shape.convolution {
            encodeConvolution(encoder, layer: index, input: input, start: start, count: n)
            return
        }
        let d = config.embeddingSize, q = shape.querySize, kv = shape.kvSize, size = shape.headSize
        let heads = shape.headCount, kvHeads = shape.kvHeadCount
        guard let keyBuffer = keys[shape.kvLayer], let valueBuffer = values[shape.kvLayer] else { return }
        let cacheOffset = start * kv

        linear(encoder, layer.query, input: input, inStride: d, count: n, output: queries, outStride: q)
        normalizeAndRotate(encoder, queries, layer.queryNorm, heads: heads, count: n, layer: index)

        if shape.kvLayer == index, let key = layer.key {
            // k と v は KV キャッシュの続きに直接書く
            linear(encoder, key, input: input, inStride: d, count: n, output: keyBuffer, outOffset: cacheOffset, outStride: kv)
            if let value = layer.value {
                linear(encoder, value, input: input, inStride: d, count: n, output: valueBuffer, outOffset: cacheOffset, outStride: kv)
            } else {
                // Gemma 4: v が無ければ、正規化する前の k を使う
                scaleAdd(encoder, keyBuffer, inOffset: cacheOffset, into: valueBuffer, outOffset: cacheOffset, count: n * kv)
            }
            normalizeAndRotate(encoder, keyBuffer, offset: cacheOffset, layer.keyNorm, heads: kvHeads, count: n, layer: index)
            if config.hasValueNorm {
                normalize(encoder, valueBuffer, inOffset: cacheOffset, nil, into: valueBuffer, outOffset: cacheOffset,
                          rows: n * kvHeads, columns: size, rms: true)
            }
        }

        let arguments = attentionArguments(layer: index, start: start, count: n)
        let rows = n * heads
        let scoreBindings: [Binding] = [(queries, 0), (keyBuffer, 0), (scores, 0)]
        run(.attentionScores, encoder, scoreBindings, arguments, width: start + n, height: rows)
        let softmaxBindings: [Binding] = [(scores, 0)]
        run(.attentionSoftmax, encoder, softmaxBindings, arguments, groups: rows, threads: Self.rowThreads(4 * (start + n)))
        let combineBindings: [Binding] = [(scores, 0), (valueBuffer, 0), (attended, 0)]
        let quads = size / 4
        run(.attentionCombine, encoder, combineBindings, arguments, groups: rows, threads: quads * max(1, 256 / quads))

        linear(encoder, layer.output, input: attended, inStride: q, count: n, output: projected, outStride: d)
    }

    private func encodeFeedForward(_ encoder: MTLComputeCommandEncoder, layer index: Int, input: MTLBuffer, count n: Int) {
        let layer = layers[index]
        let d = config.embeddingSize, ff = shapes[index].feedForwardSize
        if let gate = layer.gate, n == 1, canFuseGateUp(gate, layer.up) {
            gateUp(encoder, gate: gate, up: layer.up, input: input, output: gateBuffer)
            linear(encoder, layer.down, input: gateBuffer, inStride: ff, count: n, output: feedForwardOut, outStride: d)
            return
        }
        linear(encoder, layer.up, input: input, inStride: d, count: n, output: upBuffer, outStride: ff)
        if let gate = layer.gate {
            linear(encoder, gate, input: input, inStride: d, count: n, output: gateBuffer, outStride: ff)
            activate(encoder, gateBuffer, up: upBuffer, activation: config.activation, count: n * ff)
            linear(encoder, layer.down, input: gateBuffer, inStride: ff, count: n, output: feedForwardOut, outStride: d)
        } else {
            activate(encoder, upBuffer, up: nil, activation: config.activation, count: n * ff)
            linear(encoder, layer.down, input: upBuffer, inStride: ff, count: n, output: feedForwardOut, outStride: d)
        }
    }

    /// 層ごとの入力の用意（Gemma 4）: 埋め込みからの射影 × 1/√d → 層ごとに正規化 → 表を足して × 1/√2
    private func encodePerLayerInputs(_ encoder: MTLComputeCommandEncoder, count n: Int) {
        let size = config.perLayerInputSize
        guard size > 0, let projection = perLayerProjection else { return }
        let d = config.embeddingSize, total = size * config.layerCount
        linear(encoder, projection, input: hidden, inStride: d, count: n, output: perLayerInputs, outStride: total)
        scaleAdd(encoder, perLayerInputs, scale: 1 / Float(d).squareRoot(), into: perLayerInputs, count: n * total)
        normalize(encoder, perLayerInputs, perLayerProjectionNorm, into: perLayerInputs, rows: n * config.layerCount,
                  columns: size, rms: true)
        let half = 1 / Float(2).squareRoot()
        scaleAdd(encoder, perLayerInputs, scale: half, plus: perLayerTokens, second: half, into: perLayerInputs, count: n * total)
    }

    /// 層ごとの入力を足して、層の倍率を掛ける（Gemma 4）:
    /// hidden = (hidden + norm(射影(gelu(入力のゲート · hidden) × その層の入力))) × 層の倍率
    private func encodePerLayerInput(_ encoder: MTLComputeCommandEncoder, layer index: Int, count n: Int) {
        let layer = layers[index]
        guard let inputGate = layer.inputGate, let projection = layer.projection else { return }
        let size = config.perLayerInputSize, total = size * config.layerCount, d = config.embeddingSize
        linear(encoder, inputGate, input: hidden, inStride: d, count: n, output: perLayerWork, outStride: size,
               epilogue: .geluTimes(aux: perLayerInputs, offset: index * size, stride: total))
        linear(encoder, projection, input: perLayerWork, inStride: size, count: n, output: feedForwardOut, outStride: d)
        normalize(encoder, feedForwardOut, layer.perLayerPostNorm, into: hidden, rows: n, columns: d, rms: true,
                  add: (1, layer.outputScale))
    }

    /// 最後のトークンだけ、出力層を計算する（all ならどのトークンも。verifiedLogitsBuffer に書く）
    private func encodeOutput(_ encoder: MTLComputeCommandEncoder, count n: Int, all: Bool = false) {
        guard let outputWeights else { return }
        let d = config.embeddingSize
        if all, let buffer = verifiedLogitsBuffer {
            normalize(encoder, hidden, outputNorm, into: normed, rows: n, columns: d)
            linear(encoder, outputWeights, input: normed, inStride: d, count: n, output: buffer, outStride: outputWeights.rows)
            return
        }
        normalize(encoder, hidden, inOffset: (n - 1) * d, outputNorm, into: normed, rows: 1, columns: d)
        linear(encoder, outputWeights, input: normed, inStride: d, count: 1, output: logits, outStride: outputWeights.rows)
    }
}

// MARK: - 先読み（MTP）

extension GGUFMetalModel: GGUFSpeculativeEvaluator {
    /// 先読み用のモデル（gemma4-assistant）の重みと作業用のメモリ
    private final class Drafter {
        struct Layer {
            var attentionNorm: Norm
            var query: Weights
            var queryNorm: Norm
            var output: Weights
            var postAttentionNorm: Norm
            var feedForwardNorm: Norm
            var gate: Weights
            var up: Weights
            var down: Weights
            var postFeedForwardNorm: Norm
            var outputScale: Float
        }

        let config: GGUFModelConfig
        let store: GGUFWeightStore
        let layers: [Layer]
        let preProjection: Weights, postProjection: Weights, output: Weights
        let outputNorm: Norm
        /// 使う本体の層（KV キャッシュ）: 直近だけを見る層と、全体を見る層
        let sources: (sliding: Int, full: Int)
        let fullRope: GGUFRope.Table, slidingRope: GGUFRope.Table
        let input: MTLBuffer, current: MTLBuffer, normed: MTLBuffer, projected: MTLBuffer, queries: MTLBuffer, attended: MTLBuffer
        let gate: MTLBuffer, up: MTLBuffer, logits: MTLBuffer, rotation: MTLBuffer, slidingRotation: MTLBuffer
        /// 予想したトークン（0 番目は本体が最後に選んだもの。GPU が 1 つずつ書き足す）
        let tokens: MTLBuffer
        /// 1 回に予想するトークンの数の上限
        static let maxDrafts = 8

        init(config: GGUFModelConfig, store: GGUFWeightStore, layers: [Layer], preProjection: Weights, postProjection: Weights,
             output: Weights, outputNorm: Norm, sources: (sliding: Int, full: Int), ropeFactors: UnsafePointer<Float>?,
             device: MTLDevice) throws {
            self.config = config
            self.store = store
            self.layers = layers
            self.preProjection = preProjection
            self.postProjection = postProjection
            self.output = output
            self.outputNorm = outputNorm
            self.sources = sources
            fullRope = GGUFRope.table(config: config, sliding: false, factors: ropeFactors)
            slidingRope = GGUFRope.table(config: config, sliding: true, factors: ropeFactors)
            let d = config.embeddingSize, outer = config.drafterOutputSize
            input = try GGUFMetalModel.floats(device, 2 * outer)
            current = try GGUFMetalModel.floats(device, d)
            normed = try GGUFMetalModel.floats(device, d)
            projected = try GGUFMetalModel.floats(device, d)
            queries = try GGUFMetalModel.floats(device, config.maxQuerySize)
            attended = try GGUFMetalModel.floats(device, config.maxQuerySize)
            gate = try GGUFMetalModel.floats(device, config.maxFeedForwardSize)
            up = try GGUFMetalModel.floats(device, config.maxFeedForwardSize)
            logits = try GGUFMetalModel.floats(device, output.rows)
            guard let tokens = device.makeBuffer(length: (Self.maxDrafts + 1) * MemoryLayout<UInt32>.size,
                                                 options: .storageModeShared) else {
                throw GGUFMetalError.unavailable("draft token buffer")
            }
            self.tokens = tokens
            rotation = try GGUFMetalModel.floats(device, max(fullRope.pairs, 1) * 2)
            slidingRotation = try GGUFMetalModel.floats(device, max(slidingRope.pairs, 1) * 2)
        }
    }

    /// 先読み用のモデルをつなぐ（組にならなければエラー）
    func attachDrafter(store drafterStore: GGUFWeightStore) throws {
        let drafterConfig = drafterStore.config
        let sources = try GGUFDrafterWeights.cacheLayers(drafter: drafterConfig, target: config)
        guard drafterConfig.normEpsilon == config.normEpsilon, drafterConfig.norm == config.norm else {
            throw GGUFMetalError.unavailable("draft model normalization")
        }
        let source = try GGUFDrafterWeights(store: drafterStore)
        // 小さなベクトルのまとめに、先読みの分を足す（今までの位置はそのまま使える）
        let existing = vectors.contents().assumingMemoryBound(to: Float.self)
        pool = Array(UnsafeBufferPointer(start: existing, count: vectors.length / MemoryLayout<Float>.size))
        let d = drafterConfig.embeddingSize
        var layers: [Drafter.Layer] = []
        for (index, layer) in source.layers.enumerated() {
            let shape = drafterConfig.layers[index]
            layers.append(Drafter.Layer(
                attentionNorm: norm(layer.attentionNorm, bias: nil, count: d), query: try weights(layer.query, bias: nil),
                queryNorm: norm(layer.queryNorm, bias: nil, count: shape.headSize), output: try weights(layer.output, bias: nil),
                postAttentionNorm: norm(layer.postAttentionNorm, bias: nil, count: d),
                feedForwardNorm: norm(layer.feedForwardNorm, bias: nil, count: d),
                gate: try weights(layer.gate, bias: nil), up: try weights(layer.up, bias: nil), down: try weights(layer.down, bias: nil),
                postFeedForwardNorm: norm(layer.postFeedForwardNorm, bias: nil, count: d), outputScale: layer.outputScale))
        }
        let preProjection = try weights(source.preProjection, bias: nil)
        let postProjection = try weights(source.postProjection, bias: nil)
        let output = try weights(source.output, bias: nil)
        let outputNorm = norm(source.outputNorm, bias: nil, count: d)
        let bytes = pool.count * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(bytes: pool, length: bytes, options: .storageModeShared) else {
            throw GGUFMetalError.unavailable("vector buffer")
        }
        vectors = buffer
        pool = []
        drafter = try Drafter(config: drafterConfig, store: drafterStore, layers: layers, preProjection: preProjection,
                              postProjection: postProjection, output: output, outputNorm: outputNorm, sources: sources,
                              ropeFactors: source.ropeFactors, device: device)
    }

    var canDraft: Bool { drafter != nil }

    func draft(after token: Int, hidden state: [Float], count: Int) throws -> [Int] {
        guard let drafter, count > 0, length > 0 else { return [] }
        let position = length, outer = drafter.config.drafterOutputSize
        let steps = min(count, Drafter.maxDrafts)
        try reserveScores(drafter.config.maxHeadCount * position)
        fill(drafter.rotation, drafter.fullRope, start: position, count: 1)
        fill(drafter.slidingRotation, drafter.slidingRope, start: position, count: 1)
        // 入力: [本体の埋め込み × √(本体の大きさ), 隠れ状態（最初は本体の、あとは前の予想の。予想の計算がそこに書く）]
        let input = drafter.input.contents().assumingMemoryBound(to: Float.self)
        state.withUnsafeBufferPointer { (input + outer).update(from: $0.baseAddress!, count: min(outer, $0.count)) }
        let tokens = drafter.tokens.contents().assumingMemoryBound(to: UInt32.self)
        tokens[0] = UInt32(token)
        let scale = Float(outer).squareRoot()
        // 本体の埋め込みの表が GPU にあれば（出力層と同じ表）、全部の予想を 1 つの命令列で計算する（CPU と往復しない）
        if config.tiedOutput, let embedding = outputWeights {
            let commands = try Commands(queue: queue, profile: nil)
            for step in 0..<steps {
                encodeEmbedding(commands.encoder, embedding, tokens: drafter.tokens, slot: step, scale: scale, into: drafter.input)
                encodeDraftStep(commands.encoder, drafter, position: position)
                encodeArgmax(commands.encoder, drafter.logits, count: drafter.output.rows, into: drafter.tokens, slot: step + 1)
            }
            try commands.finish()
            lastGPUSeconds = commands.gpuSeconds
            return (1...steps).map { Int(tokens[$0]) }
        }
        var drafts: [Int] = []
        var current = token
        for _ in 0..<steps {
            GGUFKernels.embeddingRow(globals.embedding, row: current, scale: scale, into: input)
            let commands = try Commands(queue: queue, profile: nil)
            encodeDraftStep(commands.encoder, drafter, position: position)
            encodeArgmax(commands.encoder, drafter.logits, count: drafter.output.rows, into: drafter.tokens, slot: 1)
            try commands.finish()
            current = Int(tokens[1])
            drafts.append(current)
        }
        return drafts
    }

    /// 埋め込みの表の 1 行（行の番号は GPU の tokens[slot]）× scale を output の先頭に書く
    private func encodeEmbedding(_ encoder: MTLComputeCommandEncoder, _ weights: Weights, tokens: MTLBuffer, slot: Int, scale: Float,
                                 into output: MTLBuffer) {
        var arguments = GGUFMetalArgs()
        arguments.columns = UInt32(weights.columns)
        arguments.start = UInt32(slot)
        arguments.weightSkip = UInt32(weights.skip)
        arguments.scale = scale
        var rowScales: Binding = vector(nil)
        var high: Binding = vector(nil)
        if weights.groupBits > 0, let params = weights.params {
            arguments.size = UInt32(weights.groupBits)
            rowScales = params
            if let bits = weights.high { high = (bits.buffer, bits.offset) }
        } else if weights.isInt8 {
            arguments.size = 8
            rowScales = vector(weights.scales)
            if weights.scales != nil { arguments.flags |= Flag.scales }
        } else {
            arguments.size = 32
        }
        run(.embeddingRow, encoder, [(weights.buffer, weights.offset), rowScales, (tokens, 0), (output, 0), high], arguments,
            width: weights.columns)
    }

    /// values の count 個のうち、いちばん大きいものの位置を result[slot] に書く
    private func encodeArgmax(_ encoder: MTLComputeCommandEncoder, _ values: MTLBuffer, count: Int, into result: MTLBuffer, slot: Int) {
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.outOffset = UInt32(slot)
        run(.argmax, encoder, [(values, 0), (result, 0)], arguments, groups: 1, threads: 1024)
    }

    func verify(_ tokens: [Int]) throws {
        try validate(tokens)
        let rows = globals.output.rows
        if (verifiedLogitsBuffer?.length ?? 0) < tokens.count * rows * MemoryLayout<Float>.size {
            verifiedLogitsBuffer = try Self.floats(device, max(tokens.count, 4) * rows)
        }
        verifyStart = length
        _ = try forward(tokens, wantsLogits: true, all: true)
    }

    func verifiedLogits(_ index: Int) -> [Float] {
        guard let buffer = verifiedLogitsBuffer, index < verifiedHidden.count else { return [] }
        return readLogits(from: buffer, row: index)
    }

    func accept(_ count: Int) {
        guard count < verifiedHidden.count else { return }
        truncate(to: verifyStart + count + 1)
        lastHidden = verifiedHidden[count]
    }

    /// 先読み 1 回（入力は drafter.input、logits は drafter.logits、次の隠れ状態は drafter.input の後ろ半分）
    private func encodeDraftStep(_ encoder: MTLComputeCommandEncoder, _ drafter: Drafter, position: Int) {
        let config = drafter.config, d = config.embeddingSize, outer = config.drafterOutputSize
        linear(encoder, drafter.preProjection, input: drafter.input, inStride: 2 * outer, count: 1, output: drafter.current, outStride: d)
        for (index, layer) in drafter.layers.enumerated() {
            let shape = config.layers[index], ff = shape.feedForwardSize
            normalize(encoder, drafter.current, layer.attentionNorm, into: drafter.normed, rows: 1, columns: d)
            linear(encoder, layer.query, input: drafter.normed, inStride: d, count: 1, output: drafter.queries, outStride: shape.querySize)
            encodeDraftQueryNorm(encoder, drafter, layer: index)
            encodeDraftAttention(encoder, drafter, layer: index, position: position)
            linear(encoder, layer.output, input: drafter.attended, inStride: shape.querySize, count: 1, output: drafter.projected,
                   outStride: d)
            normalize(encoder, drafter.projected, layer.postAttentionNorm, into: drafter.current, rows: 1, columns: d, add: (1, 1))
            normalize(encoder, drafter.current, layer.feedForwardNorm, into: drafter.normed, rows: 1, columns: d)
            linear(encoder, layer.up, input: drafter.normed, inStride: d, count: 1, output: drafter.up, outStride: ff)
            linear(encoder, layer.gate, input: drafter.normed, inStride: d, count: 1, output: drafter.gate, outStride: ff)
            activate(encoder, drafter.gate, up: drafter.up, activation: .gelu, count: ff)
            linear(encoder, layer.down, input: drafter.gate, inStride: ff, count: 1, output: drafter.projected, outStride: d)
            normalize(encoder, drafter.projected, layer.postFeedForwardNorm, into: drafter.current, rows: 1, columns: d,
                      add: (1, layer.outputScale))
        }
        normalize(encoder, drafter.current, drafter.outputNorm, into: drafter.normed, rows: 1, columns: d)
        linear(encoder, drafter.output, input: drafter.normed, inStride: d, count: 1, output: drafter.logits, outStride: drafter.output.rows)
        // 次の隠れ状態は、そのまま次の予想の入力の後ろ半分に書く
        linear(encoder, drafter.postProjection, input: drafter.normed, inStride: d, count: 1, output: drafter.input, outOffset: outer,
               outStride: outer)
    }

    /// 先読みの q のヘッドごとの正規化と RoPE（位置は本体がまだ読んでいない位置。表の 0 行目）
    private func encodeDraftQueryNorm(_ encoder: MTLComputeCommandEncoder, _ drafter: Drafter, layer index: Int) {
        let config = drafter.config, shape = config.layers[index], norm = drafter.layers[index].queryNorm
        let table = shape.sliding ? drafter.slidingRope : drafter.fullRope
        var arguments = GGUFMetalArgs()
        arguments.heads = UInt32(shape.headCount)
        arguments.kvHeads = UInt32(norm.heads)
        arguments.size = UInt32(shape.headSize)
        arguments.pairs = UInt32(shape.usesRope ? table.pairs : 0)
        arguments.epsilon = config.normEpsilon
        if norm.weight != nil { arguments.flags |= Flag.normWeight }
        if config.ropePairsAreSplit { arguments.flags |= Flag.splitPairs }
        let bindings: [Binding] = [(drafter.queries, 0), vector(norm.weight), (shape.sliding ? drafter.slidingRotation : drafter.rotation, 0)]
        run(.headNormRope, encoder, bindings, arguments, groups: shape.headCount, threads: Self.rowThreads(shape.headSize))
    }

    /// 先読みの注意: 本体の KV キャッシュの位置 0 … position − 1 を見る（直近だけを見る層は、position から窓の大きさ未満）
    private func encodeDraftAttention(_ encoder: MTLComputeCommandEncoder, _ drafter: Drafter, layer index: Int, position: Int) {
        let config = drafter.config, shape = config.layers[index]
        let source = shape.sliding ? drafter.sources.sliding : drafter.sources.full
        guard let keyBuffer = keys[source], let valueBuffer = values[source] else { return }
        var arguments = GGUFMetalArgs()
        arguments.count = 1
        arguments.heads = UInt32(shape.headCount)
        arguments.kvHeads = 1
        arguments.size = UInt32(shape.headSize)
        arguments.total = UInt32(position)
        // q は位置 position にあるが、キャッシュは position − 1 まで: 位置 position − 1 から見ることにして、窓を 1 つ小さくする
        arguments.start = UInt32(position - 1)
        arguments.window = UInt32(shape.sliding && config.slidingWindow > 1 ? config.slidingWindow - 1 : 0)
        arguments.scale = config.attentionScale
        arguments.softcap = config.attentionSoftcap
        let heads = shape.headCount
        run(.attentionScores, encoder, [(drafter.queries, 0), (keyBuffer, 0), (scores, 0)], arguments, width: position, height: heads)
        run(.attentionSoftmax, encoder, [(scores, 0)], arguments, groups: heads, threads: Self.rowThreads(4 * position))
        let quads = shape.headSize / 4
        run(.attentionCombine, encoder, [(scores, 0), (valueBuffer, 0), (drafter.attended, 0)], arguments, groups: heads,
            threads: quads * max(1, 256 / quads))
    }
}
#endif
