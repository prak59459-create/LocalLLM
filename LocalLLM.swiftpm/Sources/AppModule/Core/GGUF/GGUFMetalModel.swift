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
/// - スレッドセーフではないので、1 つの場所から順番に使う。
final class GGUFMetalModel: GGUFEvaluator {
    typealias Kernel = GGUFMetalSource.Kernel

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
    }

    private struct Layer {
        var attentionNorm: Int
        var query: Weights
        var key: Weights
        var value: Weights
        var output: Weights
        var queryNorm: Int?
        var keyNorm: Int?
        var postAttentionNorm: Int?
        var feedForwardNorm: Int
        var gate: Weights
        var up: Weights
        var down: Weights
        var postFeedForwardNorm: Int?
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

    private var layers: [Layer] = []
    private let embedding: GGUFMatrix
    private var outputWeights: Weights?
    private var outputNorm = 0
    private let outputBias: UnsafePointer<Float>?

    private let frequencies: [Float]
    private let slidingFrequencies: [Float]

    /// 1 回の命令列で計算するトークン数の上限
    let batchCapacity: Int
    private let hidden: MTLBuffer, normed: MTLBuffer, queries: MTLBuffer, attended: MTLBuffer, projected: MTLBuffer
    private let gateBuffer: MTLBuffer, upBuffer: MTLBuffer, logits: MTLBuffer
    private let rotation: MTLBuffer, slidingRotation: MTLBuffer
    private var scores: MTLBuffer
    private var scoresCapacity: Int

    /// 層ごとの KV キャッシュ: [位置][kvHead][headSize]。足りなくなったら大きくする
    private var keys: [MTLBuffer] = []
    private var values: [MTLBuffer] = []
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
        guard config.headSize % 4 == 0, config.headCount % config.kvHeadCount == 0 else {
            throw GGUFMetalError.unavailable("head size \(config.headSize)")
        }
        self.store = store
        self.device = device
        self.queue = queue
        self.config = config
        deviceName = device.name
        self.maxContext = max(8, min(config.contextLength, maxContext ?? config.contextLength))
        embedding = try store.matrix("token_embd.weight")
        outputBias = store.vector("output.bias")
        let factors = store.vector("rope_freqs.weight")
        frequencies = GGUFModel.ropeFrequencies(config: config, base: config.ropeBase, factors: factors)
        slidingFrequencies = GGUFModel.ropeFrequencies(config: config, base: config.slidingRopeBase, factors: factors)

        let capacity = Self.batchCapacity(config: config)
        batchCapacity = capacity
        let d = config.embeddingSize, ff = config.feedForwardSize, q = config.querySize
        hidden = try Self.floats(device, capacity * d)
        normed = try Self.floats(device, capacity * d)
        queries = try Self.floats(device, capacity * q)
        attended = try Self.floats(device, capacity * q)
        projected = try Self.floats(device, capacity * d)
        gateBuffer = try Self.floats(device, capacity * ff)
        upBuffer = try Self.floats(device, capacity * ff)
        logits = try Self.floats(device, config.vocabSize)
        let tableCount = capacity * max(frequencies.count, 1) * 2
        rotation = try Self.floats(device, tableCount)
        slidingRotation = try Self.floats(device, tableCount)
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
        let d = Double(config.embeddingSize), ff = Double(config.feedForwardSize)
        let q = Double(config.querySize), kv = Double(config.kvSize)
        let perLayer = d * (q + 2 * kv) + q * d + 3 * d * ff
        let perToken = perLayer * Double(config.layerCount)
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
        let d = config.embeddingSize, size = config.headSize
        for index in 0..<config.layerCount {
            let source = try GGUFLayerWeights.load(index, store: store, config: config)
            let layer = Layer(
                attentionNorm: pooled(source.attentionNorm, d),
                query: try weights(source.query, bias: source.queryBias),
                key: try weights(source.key, bias: source.keyBias),
                value: try weights(source.value, bias: source.valueBias),
                output: try weights(source.output, bias: source.outputBias),
                queryNorm: source.queryNorm.map { pooled($0, size) },
                keyNorm: source.keyNorm.map { pooled($0, size) },
                postAttentionNorm: source.postAttentionNorm.map { pooled($0, d) },
                feedForwardNorm: pooled(source.feedForwardNorm, d),
                gate: try weights(source.gate, bias: source.gateBias),
                up: try weights(source.up, bias: source.upBias),
                down: try weights(source.down, bias: source.downBias),
                postFeedForwardNorm: source.postFeedForwardNorm.map { pooled($0, d) }
            )
            layers.append(layer)
        }
        outputNorm = pooled(try store.requiredVector("output_norm.weight"), d)
        let head = try store.matrix(config.tiedOutput ? "token_embd.weight" : "output.weight")
        outputWeights = try weights(head, bias: nil)

        let bytes = pool.count * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(bytes: pool, length: bytes, options: .storageModeShared) else {
            throw GGUFMetalError.unavailable("vector buffer")
        }
        vectors = buffer
        pool = []
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
        let isInt8 = matrix.storage == .int8Rows
        let bytes = matrix.rows * matrix.columns * (isInt8 ? 1 : MemoryLayout<Float>.size)
        let (buffer, offset) = try mapped(matrix.data, bytes: bytes)
        let aligned = offset / 16 * 16
        return Weights(buffer: buffer, offset: aligned, skip: offset - aligned, rows: matrix.rows, columns: matrix.columns,
                       isInt8: isInt8,
                       scales: matrix.scales.map { pooled($0, matrix.rows) },
                       bias: bias.map { pooled($0, matrix.rows) })
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
        let count = capacity * config.kvSize
        var newKeys: [MTLBuffer] = [], newValues: [MTLBuffer] = []
        for _ in 0..<config.layerCount {
            newKeys.append(try Self.floats(device, count))
            newValues.append(try Self.floats(device, count))
        }
        // 今入っている分は引き継ぐ
        let used = min(length, cacheCapacity, capacity) * config.kvSize * MemoryLayout<Float>.size
        if used > 0 {
            for index in keys.indices {
                newKeys[index].contents().copyMemory(from: keys[index].contents(), byteCount: used)
                newValues[index].contents().copyMemory(from: values[index].contents(), byteCount: used)
            }
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
        while count > 1 && count * config.headCount * (length + count) > Self.scoresBudget { count /= 2 }
        return count
    }

    private func forward(_ tokens: [Int], wantsLogits: Bool) throws -> [Float] {
        let n = tokens.count
        let d = config.embeddingSize
        let start = length
        if cacheCapacity == 0 { try allocateCache(min(maxContext, Self.initialCache)) }
        try reserveCache(start + n)
        try reserveScores(n * config.headCount * (start + n))

        let x = hidden.contents().assumingMemoryBound(to: Float.self)
        for (index, token) in tokens.enumerated() {
            GGUFKernels.embeddingRow(embedding, row: token, scale: config.embeddingScale, into: x + index * d)
        }
        fillRotations(start: start, count: n)

        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw GGUFMetalError.failed("command buffer")
        }
        for index in layers.indices {
            encodeAttention(encoder, layer: index, start: start, count: n)
            encodeFeedForward(encoder, layer: index, count: n)
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

    private func readLogits() -> [Float] {
        let count = config.vocabSize
        let source = logits.contents().assumingMemoryBound(to: Float.self)
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            base.update(from: source, count: count)
            if let outputBias { MiniMath.add(outputBias, into: base, count: count) }
            if config.finalSoftcap > 0 { GGUFKernels.softcap(base, count: count, cap: config.finalSoftcap) }
        }
        return result
    }

    /// 位置 start ..< start + count の cos / sin を表に書く: [トークン][組][cos, sin]
    private func fillRotations(start: Int, count: Int) {
        let pairs = frequencies.count
        guard pairs > 0 else { return }
        fill(rotation, frequencies, start: start, count: count)
        if config.slidingWindow > 0 { fill(slidingRotation, slidingFrequencies, start: start, count: count) }
    }

    private func fill(_ buffer: MTLBuffer, _ source: [Float], start: Int, count: Int) {
        let pairs = source.count
        let table = buffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<count {
            let position = Float(start + index)
            let row = table + index * pairs * 2
            for pair in 0..<pairs {
                let angle: Float = position * source[pair]
                row[pair * 2] = Foundation.cos(angle)
                row[pair * 2 + 1] = Foundation.sin(angle)
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
        if weights.scales != nil { arguments.flags |= GGUFMetalSource.Flag.scales }
        if weights.bias != nil { arguments.flags |= GGUFMetalSource.Flag.bias }
        let batched = count > 1
        let kernel: Kernel
        if weights.isInt8 {
            kernel = batched ? .linearQ8Batch : .linearQ8
        } else {
            kernel = batched ? .linearF32Batch : .linearF32
        }
        let bindings: [Binding] = [(weights.buffer, weights.offset), vector(weights.scales), vector(weights.bias), (input, 0), (output, 0)]
        run(kernel, encoder, bindings, arguments, width: weights.rows, height: batched ? (count + 3) / 4 : 1)
    }

    private func norm(_ encoder: MTLComputeCommandEncoder, _ input: MTLBuffer, inOffset: Int = 0, weight: Int,
                      into output: MTLBuffer, outOffset: Int = 0, rows: Int, columns: Int) {
        var arguments = GGUFMetalArgs()
        arguments.rows = UInt32(rows)
        arguments.columns = UInt32(columns)
        arguments.inOffset = UInt32(inOffset)
        arguments.outOffset = UInt32(outOffset)
        arguments.epsilon = config.normEpsilon
        let bindings: [Binding] = [(input, 0), vector(weight), (output, 0)]
        run(.rmsNorm, encoder, bindings, arguments, width: rows)
    }

    private func rotate(_ encoder: MTLComputeCommandEncoder, _ buffer: MTLBuffer, offset: Int, heads: Int, count: Int, sliding: Bool) {
        let pairs = frequencies.count
        guard pairs > 0 else { return }
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.heads = UInt32(heads)
        arguments.size = UInt32(config.headSize)
        arguments.pairs = UInt32(pairs)
        arguments.outOffset = UInt32(offset)
        if config.ropePairsAreSplit { arguments.flags |= GGUFMetalSource.Flag.splitPairs }
        let bindings: [Binding] = [(buffer, 0), (sliding ? slidingRotation : rotation, 0)]
        run(.rope, encoder, bindings, arguments, width: pairs, height: count * heads)
    }

    private func add(_ encoder: MTLComputeCommandEncoder, _ target: MTLBuffer, _ source: MTLBuffer, count: Int) {
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        let bindings: [Binding] = [(target, 0), (source, 0)]
        run(.addRows, encoder, bindings, arguments, width: count)
    }

    private func attentionArguments(layer: Int, start: Int, count: Int) -> GGUFMetalArgs {
        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(count)
        arguments.heads = UInt32(config.headCount)
        arguments.kvHeads = UInt32(config.kvHeadCount)
        arguments.size = UInt32(config.headSize)
        arguments.total = UInt32(start + count)
        arguments.start = UInt32(start)
        arguments.window = UInt32(config.isSliding(layer) ? config.slidingWindow : 0)
        arguments.scale = config.attentionScale
        arguments.softcap = config.attentionSoftcap
        return arguments
    }

    private func encodeAttention(_ encoder: MTLComputeCommandEncoder, layer index: Int, start: Int, count n: Int) {
        let layer = layers[index]
        let d = config.embeddingSize, q = config.querySize, kv = config.kvSize, size = config.headSize
        let heads = config.headCount, kvHeads = config.kvHeadCount
        let keyBuffer = keys[index], valueBuffer = values[index]
        let cacheOffset = start * kv
        let sliding = config.isSliding(index)

        norm(encoder, hidden, weight: layer.attentionNorm, into: normed, rows: n, columns: d)
        linear(encoder, layer.query, input: normed, inStride: d, count: n, output: queries, outStride: q)
        // k と v は KV キャッシュの続きに直接書く
        linear(encoder, layer.key, input: normed, inStride: d, count: n, output: keyBuffer, outOffset: cacheOffset, outStride: kv)
        linear(encoder, layer.value, input: normed, inStride: d, count: n, output: valueBuffer, outOffset: cacheOffset, outStride: kv)
        if let queryNorm = layer.queryNorm, let keyNorm = layer.keyNorm {
            norm(encoder, queries, weight: queryNorm, into: queries, rows: n * heads, columns: size)
            norm(encoder, keyBuffer, inOffset: cacheOffset, weight: keyNorm, into: keyBuffer, outOffset: cacheOffset,
                 rows: n * kvHeads, columns: size)
        }
        rotate(encoder, queries, offset: 0, heads: heads, count: n, sliding: sliding)
        rotate(encoder, keyBuffer, offset: cacheOffset, heads: kvHeads, count: n, sliding: sliding)

        let arguments = attentionArguments(layer: index, start: start, count: n)
        let rows = n * heads
        let scoreBindings: [Binding] = [(queries, 0), (keyBuffer, 0), (scores, 0)]
        run(.attentionScores, encoder, scoreBindings, arguments, width: start + n, height: rows)
        let softmaxBindings: [Binding] = [(scores, 0)]
        run(.attentionSoftmax, encoder, softmaxBindings, arguments, width: rows)
        let combineBindings: [Binding] = [(scores, 0), (valueBuffer, 0), (attended, 0)]
        run(.attentionCombine, encoder, combineBindings, arguments, width: size / 4, height: rows)

        linear(encoder, layer.output, input: attended, inStride: q, count: n, output: projected, outStride: d)
        if let postNorm = layer.postAttentionNorm {
            norm(encoder, projected, weight: postNorm, into: projected, rows: n, columns: d)
        }
        add(encoder, hidden, projected, count: n * d)
    }

    private func encodeFeedForward(_ encoder: MTLComputeCommandEncoder, layer index: Int, count n: Int) {
        let layer = layers[index]
        let d = config.embeddingSize, ff = config.feedForwardSize
        norm(encoder, hidden, weight: layer.feedForwardNorm, into: normed, rows: n, columns: d)
        linear(encoder, layer.gate, input: normed, inStride: d, count: n, output: gateBuffer, outStride: ff)
        linear(encoder, layer.up, input: normed, inStride: d, count: n, output: upBuffer, outStride: ff)

        var arguments = GGUFMetalArgs()
        arguments.count = UInt32(n * ff)
        if config.activation == .gelu { arguments.flags |= GGUFMetalSource.Flag.gelu }
        let bindings: [Binding] = [(gateBuffer, 0), (upBuffer, 0)]
        run(.gatedActivation, encoder, bindings, arguments, width: n * ff)

        linear(encoder, layer.down, input: gateBuffer, inStride: ff, count: n, output: projected, outStride: d)
        if let postNorm = layer.postFeedForwardNorm {
            norm(encoder, projected, weight: postNorm, into: projected, rows: n, columns: d)
        }
        add(encoder, hidden, projected, count: n * d)
    }

    /// 最後のトークンだけ、出力層を計算する
    private func encodeOutput(_ encoder: MTLComputeCommandEncoder, count n: Int) {
        guard let outputWeights else { return }
        let d = config.embeddingSize
        norm(encoder, hidden, inOffset: (n - 1) * d, weight: outputNorm, into: normed, rows: 1, columns: d)
        linear(encoder, outputWeights, input: normed, inStride: d, count: 1, output: logits, outStride: config.vocabSize)
    }
}
#endif
