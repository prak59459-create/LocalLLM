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
    var pad: Float = 0
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
        for shape in shapes where shape.headSize % 4 != 0 || shape.headCount % shape.kvHeadCount != 0 {
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
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: GGUFMetalSource.code, options: nil)
        } catch {
            throw GGUFMetalError.unavailable("shader: \(error.localizedDescription)")
        }
        for kernel in Kernel.allCases {
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

    private func forward(_ tokens: [Int], wantsLogits: Bool) throws -> [Float] {
        let n = tokens.count
        let start = length
        if cacheCapacity == 0 { try allocateCache(min(maxContext, Self.initialCache)) }
        try reserveCache(start + n)
        try reserveScores(n * config.maxHeadCount * (start + n))
        embed(tokens, start: start)
        fillRotations(start: start, count: n)

        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw GGUFMetalError.failed("command buffer")
        }
        encodePerLayerInputs(encoder, count: n)
        for index in layers.indices {
            encodeLayer(encoder, layer: index, start: start, count: n)
        }
        if wantsLogits { encodeOutput(encoder, count: n) }
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error {
            throw GGUFMetalError.failed(error.localizedDescription)
        }
        length = start + n
        return wantsLogits ? readLogits() : []
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

    private func readLogits() -> [Float] {
        let rows = globals.output.rows
        let source = logits.contents().assumingMemoryBound(to: Float.self)
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
        encoder.setComputePipelineState(pipeline)
        for (index, binding) in bindings.enumerated() {
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: index)
        }
        var values = arguments
        encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: bindings.count)
        let group = max(1, pipeline.threadExecutionWidth)
        let groups = MTLSize(width: (width + group - 1) / group, height: height, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
    }

    /// threadgroup の数を決めて並べる（1 つの threadgroup に threads 個のスレッド）
    private func run(_ kernel: Kernel, _ encoder: MTLComputeCommandEncoder, _ bindings: [Binding], _ arguments: GGUFMetalArgs,
                     groups: Int, threads: Int) {
        guard let pipeline = pipelines[kernel], groups > 0 else { return }
        encoder.setComputePipelineState(pipeline)
        for (index, binding) in bindings.enumerated() {
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: index)
        }
        var values = arguments
        encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: bindings.count)
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// SIMD グループが 32 スレッドの GPU（Apple の GPU）なら、1 トークンの線形層を 32 スレッドで 1 行ずつ計算する
    private var usesSimdRows: Bool {
        pipelines[.linearQ8Vector]?.threadExecutionWidth == 32 && pipelines[.linearF32Vector]?.threadExecutionWidth == 32
            && pipelines[.linearQ4Vector]?.threadExecutionWidth == 32 && pipelines[.linearQ5Vector]?.threadExecutionWidth == 32
    }

    private func vector(_ position: Int?) -> Binding {
        (vectors, (position ?? 0) * MemoryLayout<Float>.size)
    }

    private func linear(_ encoder: MTLComputeCommandEncoder, _ weights: Weights, input: MTLBuffer, inOffset: Int = 0, inStride: Int,
                        count: Int, output: MTLBuffer, outOffset: Int = 0, outStride: Int) {
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
        let batched = count > 1
        if weights.groupBits > 0, let params = weights.params {
            linearGrouped(encoder, weights, params: params, input: input, output: output, arguments: arguments, batched: batched)
            return
        }
        let bindings: [Binding] = [(weights.buffer, weights.offset), vector(weights.scales), vector(weights.bias), (input, 0), (output, 0)]
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

    /// 4・5 ビットの線形層（1 トークンは 32 スレッドで 1 行、それ以外はトークン 4 個ずつ）
    private func linearGrouped(_ encoder: MTLComputeCommandEncoder, _ weights: Weights, params: Binding, input: MTLBuffer,
                               output: MTLBuffer, arguments: GGUFMetalArgs, batched: Bool) {
        let five = weights.groupBits == 5
        // 5 ビット目の面が無い 4 ビットでも、場所は埋めておく（読まない）
        let high: Binding = weights.high.map { ($0.buffer, $0.offset) } ?? params
        let bindings: [Binding] = [(weights.buffer, weights.offset), params, vector(weights.bias), (input, 0), (output, 0), high]
        if !batched && usesSimdRows {
            run(five ? .linearQ5Vector : .linearQ4Vector, encoder, bindings, arguments, groups: weights.rows, threads: 32)
            return
        }
        run(five ? .linearQ5Batch : .linearQ4Batch, encoder, bindings, arguments, width: weights.rows,
            height: (Int(arguments.count) + 3) / 4)
    }

    /// 正規化（weight が無ければ、重みを掛けない正規化）
    private func normalize(_ encoder: MTLComputeCommandEncoder, _ input: MTLBuffer, inOffset: Int = 0, _ norm: Norm?,
                           into output: MTLBuffer, outOffset: Int = 0, rows: Int, columns: Int, rms: Bool = false) {
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
        let bindings: [Binding] = [(input, 0), vector(norm?.weight), vector(norm?.bias), (output, 0)]
        run(.rmsNorm, encoder, bindings, arguments, width: rows)
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

    /// q / k の正規化（全体・ヘッドごと）
    private func normalizeHeads(_ encoder: MTLComputeCommandEncoder, _ buffer: MTLBuffer, offset: Int = 0, _ norm: Norm,
                                heads: Int, size: Int, count n: Int) {
        if config.qkNormIsFull {
            normalize(encoder, buffer, inOffset: offset, norm, into: buffer, outOffset: offset, rows: n, columns: heads * size)
        } else {
            normalize(encoder, buffer, inOffset: offset, norm, into: buffer, outOffset: offset, rows: n * heads, columns: size)
        }
    }

    // MARK: - 層

    private func encodeLayer(_ encoder: MTLComputeCommandEncoder, layer index: Int, start: Int, count n: Int) {
        let layer = layers[index]
        let d = config.embeddingSize
        // 注意の入力
        var attentionInput = hidden
        if let norm = layer.attentionNorm {
            normalize(encoder, hidden, norm, into: normed, rows: n, columns: d)
            attentionInput = normed
        }
        encodeAttention(encoder, layer: index, input: attentionInput, start: start, count: n)
        finishBranch(encoder, projected, postNorm: layer.postAttentionNorm, count: n)

        if config.parallelResidual {
            var input = attentionInput
            if layer.feedForwardNorm?.weight != nil {
                normalize(encoder, hidden, layer.feedForwardNorm, into: normed2, rows: n, columns: d)
                input = normed2
            }
            encodeFeedForward(encoder, layer: index, input: input, count: n)
            finishBranch(encoder, feedForwardOut, postNorm: layer.postFeedForwardNorm, count: n)
            scaleAdd(encoder, hidden, plus: projected, into: hidden, count: n * d)
            scaleAdd(encoder, hidden, plus: feedForwardOut, into: hidden, count: n * d)
        } else {
            scaleAdd(encoder, hidden, plus: projected, into: hidden, count: n * d)
            var input = hidden
            if config.hasPreNorms {
                normalize(encoder, hidden, layer.feedForwardNorm, into: normed, rows: n, columns: d)
                input = normed
            }
            encodeFeedForward(encoder, layer: index, input: input, count: n)
            finishBranch(encoder, feedForwardOut, postNorm: layer.postFeedForwardNorm, count: n)
            scaleAdd(encoder, hidden, plus: feedForwardOut, into: hidden, count: n * d)
        }
        encodePerLayerInput(encoder, layer: index, count: n)
        if layer.outputScale != 1 { scaleAdd(encoder, hidden, scale: layer.outputScale, into: hidden, count: n * d) }
    }

    /// 注意・FFN の出力の仕上げ（後ろの正規化と倍率）
    private func finishBranch(_ encoder: MTLComputeCommandEncoder, _ buffer: MTLBuffer, postNorm: Norm?, count n: Int) {
        let d = config.embeddingSize
        if let postNorm { normalize(encoder, buffer, postNorm, into: buffer, rows: n, columns: d) }
        if config.residualScale != 1 { scaleAdd(encoder, buffer, scale: config.residualScale, into: buffer, count: n * d) }
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
        let afterRope = config.qkNormAfterRope

        linear(encoder, layer.query, input: input, inStride: d, count: n, output: queries, outStride: q)
        if afterRope { rotate(encoder, queries, offset: 0, heads: heads, count: n, layer: index) }
        if let norm = layer.queryNorm { normalizeHeads(encoder, queries, norm, heads: heads, size: size, count: n) }
        if !afterRope { rotate(encoder, queries, offset: 0, heads: heads, count: n, layer: index) }

        if shape.kvLayer == index, let key = layer.key {
            // k と v は KV キャッシュの続きに直接書く
            linear(encoder, key, input: input, inStride: d, count: n, output: keyBuffer, outOffset: cacheOffset, outStride: kv)
            if let value = layer.value {
                linear(encoder, value, input: input, inStride: d, count: n, output: valueBuffer, outOffset: cacheOffset, outStride: kv)
            } else {
                // Gemma 4: v が無ければ、正規化する前の k を使う
                scaleAdd(encoder, keyBuffer, inOffset: cacheOffset, into: valueBuffer, outOffset: cacheOffset, count: n * kv)
            }
            if afterRope { rotate(encoder, keyBuffer, offset: cacheOffset, heads: kvHeads, count: n, layer: index) }
            if let norm = layer.keyNorm {
                normalizeHeads(encoder, keyBuffer, offset: cacheOffset, norm, heads: kvHeads, size: size, count: n)
            }
            if config.hasValueNorm {
                normalize(encoder, valueBuffer, inOffset: cacheOffset, nil, into: valueBuffer, outOffset: cacheOffset,
                          rows: n * kvHeads, columns: size, rms: true)
            }
            if !afterRope { rotate(encoder, keyBuffer, offset: cacheOffset, heads: kvHeads, count: n, layer: index) }
        }

        let arguments = attentionArguments(layer: index, start: start, count: n)
        let rows = n * heads
        let scoreBindings: [Binding] = [(queries, 0), (keyBuffer, 0), (scores, 0)]
        run(.attentionScores, encoder, scoreBindings, arguments, width: start + n, height: rows)
        let softmaxBindings: [Binding] = [(scores, 0)]
        run(.attentionSoftmax, encoder, softmaxBindings, arguments, width: rows)
        let combineBindings: [Binding] = [(scores, 0), (valueBuffer, 0), (attended, 0)]
        run(.attentionCombine, encoder, combineBindings, arguments, width: size / 4, height: rows)

        linear(encoder, layer.output, input: attended, inStride: q, count: n, output: projected, outStride: d)
    }

    private func encodeFeedForward(_ encoder: MTLComputeCommandEncoder, layer index: Int, input: MTLBuffer, count n: Int) {
        let layer = layers[index]
        let d = config.embeddingSize, ff = shapes[index].feedForwardSize
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

    /// 層ごとの入力を足す（Gemma 4）
    private func encodePerLayerInput(_ encoder: MTLComputeCommandEncoder, layer index: Int, count n: Int) {
        let layer = layers[index]
        guard let inputGate = layer.inputGate, let projection = layer.projection else { return }
        let size = config.perLayerInputSize, total = size * config.layerCount, d = config.embeddingSize
        linear(encoder, inputGate, input: hidden, inStride: d, count: n, output: perLayerWork, outStride: size)
        activate(encoder, perLayerWork, up: perLayerInputs, upOffset: index * size, upColumns: size, upStride: total,
                 activation: .gelu, count: n * size)
        linear(encoder, projection, input: perLayerWork, inStride: size, count: n, output: feedForwardOut, outStride: d)
        normalize(encoder, feedForwardOut, layer.perLayerPostNorm, into: feedForwardOut, rows: n, columns: d, rms: true)
        scaleAdd(encoder, hidden, plus: feedForwardOut, into: hidden, count: n * d)
    }

    /// 最後のトークンだけ、出力層を計算する
    private func encodeOutput(_ encoder: MTLComputeCommandEncoder, count n: Int) {
        guard let outputWeights else { return }
        let d = config.embeddingSize
        normalize(encoder, hidden, inOffset: (n - 1) * d, outputNorm, into: normed, rows: 1, columns: d)
        linear(encoder, outputWeights, input: normed, inStride: d, count: 1, output: logits, outStride: outputWeights.rows)
    }
}
#endif
