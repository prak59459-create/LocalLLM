#if canImport(Metal)
import Metal
import XCTest
@testable import LocalLLMCore

/// 線形層の計算のしかたごとの GPU の時間を、Gemma 4 E2B と同じ大きさの行列で測る（速さを測る流れでだけ動く）。
/// アプリの計算（rows・few・matrix・legacy・gate と up をまとめたもの）と、このファイルの試しの書き方を、まとめるトークンの数ごとに比べる。
/// 試しの書き方の結果が、アプリの計算と同じになることも確かめる。
final class GGUFMetalKernelBenchTests: XCTestCase {
    private typealias Binding = (buffer: MTLBuffer, offset: Int)

    /// 測る計算: 表示名、関数名、threadgroup 1 つが受け持つ行・トークン（0 なら 1 トークンだけ）、スレッド数、測るトークン数
    private struct Candidate {
        let label: String
        let function: String
        let rowsPerGroup: Int
        let tokensPerGroup: Int
        let threads: Int
        let counts: [Int]
    }

    /// 重み（copies 個。合わせてキャッシュより大きくして、毎回メモリから読む）
    private struct Matrix {
        let name: String
        let rows: Int
        let columns: Int
        let int8: Bool
        /// 4 ビット: 下位 4 ビット、8 ビット: 重み
        let weights: [MTLBuffer]
        /// 4 ビット: 倍率と下限（half2）、8 ビット: 行ごとの倍率
        let params: [MTLBuffer]
        let bytes: Int
    }

    func testLinearKernelSpeeds() throws {
        guard GGUFMetalModel.isAvailable else { throw XCTSkip("no Metal device") }
        guard GGUFFixtures.realModels != nil else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let matrices = GGUFMetalModel.supportsMatrices(device)
        let source = GGUFMetalSource.code + "\n" + (matrices ? GGUFMetalSource.matrixCode + "\n" : "") + Self.source
        let library = try device.makeLibrary(source: source, options: nil)
        var pipelines: [String: MTLComputePipelineState] = [:]
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            if let pipeline = pipelines[name] { return pipeline }
            let function = try XCTUnwrap(library.makeFunction(name: name), name)
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelines[name] = pipeline
            return pipeline
        }

        let q4: [Candidate] = [
            Candidate(label: "app-rows", function: "linear_q4_rows", rowsPerGroup: 2 * GGUFMetalSource.vectorRows, tokensPerGroup: 0,
                      threads: 64, counts: [1]),
            Candidate(label: "app-few2", function: "linear_q4_few2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2]),
            Candidate(label: "app-few3", function: "linear_q4_few3", rowsPerGroup: 8, tokensPerGroup: 3, threads: 64, counts: [3]),
            Candidate(label: "app-few4", function: "linear_q4_few4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [4, 8]),
            Candidate(label: "app-legacy", function: "linear_q4_batch", rowsPerGroup: 4, tokensPerGroup: 8, threads: 64, counts: [2, 8]),
            Candidate(label: "x-shift2", function: "xq4_one2", rowsPerGroup: 4, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-mask4", function: "xq4_mask4", rowsPerGroup: 8, tokensPerGroup: 0, threads: 64, counts: [1]),
        ] + (matrices ? [Candidate(label: "app-matrix", function: "linear_q4_mma", rowsPerGroup: 32, tokensPerGroup: 8, threads: 128,
                                   counts: [2, 4, 8])] : [])
        let q8: [Candidate] = [
            Candidate(label: "app-rows", function: "linear_q8_rows", rowsPerGroup: 2 * GGUFMetalSource.q8Rows, tokensPerGroup: 0,
                      threads: 64, counts: [1]),
            Candidate(label: "app-few2", function: "linear_q8_few2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2]),
            Candidate(label: "app-few3", function: "linear_q8_few3", rowsPerGroup: 8, tokensPerGroup: 3, threads: 64, counts: [3]),
            Candidate(label: "app-few4", function: "linear_q8_few4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [4, 8]),
            Candidate(label: "app-legacy", function: "linear_q8_tokens", rowsPerGroup: 4, tokensPerGroup: 8, threads: 64, counts: [2]),
            Candidate(label: "x-one1", function: "xq8_one1", rowsPerGroup: 2, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-one4", function: "xq8_one4", rowsPerGroup: 8, tokensPerGroup: 0, threads: 64, counts: [1]),
        ] + (matrices ? [Candidate(label: "app-matrix", function: "linear_q8_mma", rowsPerGroup: 32, tokensPerGroup: 8, threads: 128,
                                   counts: [2, 4, 8])] : [])

        var random = SplitMix64(seed: 7)
        // FFN の gate・up（後半の層）、down、注意の q、出力層（埋め込みの表と同じ 8 ビット）
        let shapes: [(String, Int, Int, Bool, Int)] = [("ffn-up", 12288, 1536, false, 6), ("ffn-down", 1536, 12288, false, 6),
                                                       ("attn-q", 2048, 1536, false, 24), ("output", 262144, 1536, true, 1)]
        for (name, rows, columns, int8, copies) in shapes {
            let matrix = try makeMatrix(device, name: name, rows: rows, columns: columns, int8: int8, copies: copies, random: &random)
            try measure(matrix, candidates: int8 ? q8 : q4, reference: int8 ? "linear_q8_tokens" : "linear_q4_batch",
                        device: device, queue: queue, pipeline: pipeline, random: &random)
        }
    }

    /// 行（出力）の少ない行列（FFN の down・注意の o など）は、SIMD グループの数が足りずにメモリを読みきれていない（23〜31 GB/s。
    /// 行の多い gate+up は 50 GB/s）。1 つの行の組を threadgroup の S 個の SIMD グループで列を分けて計算し、最後に足すと速くなるか
    func testSplitColumnsForMatricesWithFewRows() throws {
        guard GGUFMetalModel.isAvailable else { throw XCTSkip("no Metal device") }
        guard GGUFFixtures.realModels != nil else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let library = try device.makeLibrary(source: GGUFMetalSource.code + "\n" + Self.source, options: nil)
        var pipelines: [String: MTLComputePipelineState] = [:]
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            if let pipeline = pipelines[name] { return pipeline }
            let pipeline = try device.makeComputePipelineState(function: try XCTUnwrap(library.makeFunction(name: name), name))
            pipelines[name] = pipeline
            return pipeline
        }
        let candidates: [Candidate] = [
            Candidate(label: "app-rows", function: "linear_q4_rows", rowsPerGroup: 2 * GGUFMetalSource.vectorRows, tokensPerGroup: 0,
                      threads: 64, counts: [1]),
        ] + [2, 4, 8].map { split in
            Candidate(label: "x-split\(split)", function: "xq4_split\(split)", rowsPerGroup: 2, tokensPerGroup: 0, threads: 32 * split,
                      counts: [1])
        }
        var random = SplitMix64(seed: 13)
        let shapes: [(String, Int, Int)] = [("ffn-down", 1536, 12288), ("ffn-down", 1536, 6144), ("attn-o", 1536, 4096), ("attn-o", 1536, 2048),
                                            ("attn-q", 2048, 1536), ("attn-k", 256, 1536), ("ffn-up", 12288, 1536)]
        for (name, rows, columns) in shapes {
            let bytes = GGUFGroupQuant.nibbleBytes(rows: rows, columns: columns) + GGUFGroupQuant.paramBytes(rows: rows, columns: columns)
            let matrix = try makeMatrix(device, name: name, rows: rows, columns: columns, int8: false,
                                        copies: max(2, (32 << 20) / bytes + 1), random: &random)
            try measure(matrix, candidates: candidates, reference: "linear_q4_batch", device: device, queue: queue, pipeline: pipeline,
                        random: &random)
        }
    }

    /// Gemma 4 E2B の 1 トークンの計算のうち、線形層だけにかかる GPU の時間を、行列の形と数から見積もる。
    /// ほかの計算（注意・正規化など）と、計算を 1 つ並べるごとの手間（小さな計算 1 回の時間）も測る。
    /// 全体（TIME-GPU）との差で、線形層以外にどれだけかかっているかが分かる
    func testGemma4DecodeLinearTime() throws {
        guard GGUFMetalModel.isAvailable else { throw XCTSkip("no Metal device") }
        guard GGUFFixtures.realModels != nil else { throw XCTSkip("set LOCALLLM_GGUF_MODELS to run") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let library = try device.makeLibrary(source: GGUFMetalSource.code, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try device.makeComputePipelineState(function: try XCTUnwrap(library.makeFunction(name: name), name))
        }
        let rows4 = try pipeline("linear_q4_rows"), rows8 = try pipeline("linear_q8_rows"), gateUp = try pipeline("gate_up_q4")
        let columnsMax = 12288
        let input = try XCTUnwrap(device.makeBuffer(length: columnsMax * 4, options: .storageModeShared))
        let x = input.contents().bindMemory(to: Float.self, capacity: columnsMax)
        var random = SplitMix64(seed: 11)
        for i in 0..<columnsMax { x[i] = Float(Int(random.next() % 2001) - 1000) / 1000 }
        let output = try XCTUnwrap(device.makeBuffer(length: 262144 * 4, options: .storageModeShared))
        let dummy = try XCTUnwrap(device.makeBuffer(length: 64 * 1024, options: .storageModeShared))

        /// 1 回の命令列で repeats 回並べて、1 回あたりの GPU の時間（3 回のうちいちばん短いもの、秒）。最初の 1 回は温めるだけ
        func time(repeats: Int, _ encode: (MTLComputeCommandEncoder, Int) -> Void) throws -> Double {
            var best = Double.infinity
            for trial in 0..<4 {
                let buffer = try XCTUnwrap(queue.makeCommandBuffer())
                let encoder = try XCTUnwrap(buffer.makeComputeCommandEncoder())
                for index in 0..<repeats { encode(encoder, index) }
                encoder.endEncoding()
                buffer.commit()
                buffer.waitUntilCompleted()
                XCTAssertNil(buffer.error)
                if trial > 0 { best = min(best, (buffer.gpuEndTime - buffer.gpuStartTime) / Double(repeats)) }
            }
            return best
        }
        func arguments(rows: Int, columns: Int, int8: Bool) -> GGUFMetalArgs {
            var arguments = GGUFMetalArgs()
            arguments.rows = UInt32(rows)
            arguments.columns = UInt32(columns)
            arguments.count = 1
            arguments.inStride = UInt32(columns)
            arguments.outStride = UInt32(rows)
            if int8 { arguments.flags = GGUFMetalSource.Flag.scales }
            return arguments
        }

        // 1 トークンで使う線形層: 名前、行（出力）、列（入力）、8 ビットか、gate と up をまとめた計算か、1 トークンで使う回数
        // （35 層。直近だけを見る層は q 2048・kv 256、全体を見る層（7 層）は q 4096・kv 512。k・v を持つのは前の 15 層だけ）
        let layers: [(name: String, rows: Int, columns: Int, int8: Bool, fused: Bool, count: Int)] = [
            ("ffn gate+up", 6144, 1536, false, true, 15), ("ffn gate+up", 12288, 1536, false, true, 20),
            ("ffn down", 1536, 6144, false, false, 15), ("ffn down", 1536, 12288, false, false, 20),
            ("attn q", 2048, 1536, false, false, 28), ("attn q", 4096, 1536, false, false, 7),
            ("attn o", 1536, 2048, false, false, 28), ("attn o", 1536, 4096, false, false, 7),
            ("attn k+v", 256, 1536, false, false, 24), ("attn k+v", 512, 1536, false, false, 6),
            ("per-layer gate", 256, 1536, false, false, 35), ("per-layer proj", 1536, 256, false, false, 35),
            ("output", 262144, 1536, true, false, 1), ("per-layer model proj", 8960, 1536, true, false, 1),
        ]
        var totalSeconds = 0.0, totalBytes = 0, dispatches = 0
        for layer in layers {
            let bytesPerCopy = layer.int8 ? layer.rows * (layer.columns + 4)
                : GGUFGroupQuant.nibbleBytes(rows: layer.rows, columns: layer.columns) + GGUFGroupQuant.paramBytes(rows: layer.rows, columns: layer.columns)
            // 写しを合わせて 32 MB 以上にして、毎回メモリから読む（gate と up は 2 つずつ使う。64 MB より大きい行列は 1 つで足りる）
            let copies = bytesPerCopy > (64 << 20) ? 1 : max(layer.fused ? 3 : 2, (32 << 20) / bytesPerCopy + 1)
            let matrix = try makeMatrix(device, name: layer.name, rows: layer.rows, columns: layer.columns, int8: layer.int8,
                                        copies: copies, random: &random)
            let repeats = max(16, copies * 2)
            let seconds = try time(repeats: repeats) { encoder, index in
                let a = index % copies, b = (index + 1) % copies
                var values = arguments(rows: layer.rows, columns: layer.columns, int8: layer.int8)
                if layer.fused {
                    encoder.setComputePipelineState(gateUp)
                    let bindings: [MTLBuffer] = [matrix.weights[a], matrix.params[a], matrix.weights[b], matrix.params[b], input, output,
                                                 matrix.params[a], matrix.params[b]]
                    for (slot, buffer) in bindings.enumerated() { encoder.setBuffer(buffer, offset: 0, index: slot) }
                    encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: 8)
                    let perGroup = GGUFMetalSource.gateUpRows * GGUFMetalSource.simdGroups
                    encoder.dispatchThreadgroups(MTLSize(width: (layer.rows + perGroup - 1) / perGroup, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    return
                }
                encoder.setComputePipelineState(layer.int8 ? rows8 : rows4)
                let bindings: [MTLBuffer] = [matrix.weights[a], matrix.params[a], dummy, input, output, layer.int8 ? dummy : matrix.params[a],
                                             dummy]
                for (slot, buffer) in bindings.enumerated() { encoder.setBuffer(buffer, offset: 0, index: slot) }
                encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: 7)
                let perGroup = (layer.int8 ? GGUFMetalSource.q8Rows : GGUFMetalSource.vectorRows) * GGUFMetalSource.simdGroups
                encoder.dispatchThreadgroups(MTLSize(width: (layer.rows + perGroup - 1) / perGroup, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            let bytes = bytesPerCopy * (layer.fused ? 2 : 1)
            totalSeconds += seconds * Double(layer.count)
            totalBytes += bytes * layer.count
            dispatches += layer.count
            print(String(format: "KERNEL-DECODE %@ %dx%d x%d: %.1f us each = %.2f ms per token (%.1f GB/s)", layer.name, layer.rows,
                         layer.columns, layer.count, seconds * 1e6, seconds * Double(layer.count) * 1000, Double(bytes) / seconds / 1e9))
        }
        print(String(format: "KERNEL-DECODE linear layers: %.2f ms per token, %d dispatches, %.0f MB (%.1f GB/s)", totalSeconds * 1000,
                     dispatches, Double(totalBytes) / 1e6, Double(totalBytes) / totalSeconds / 1e9))

        // 小さな計算を 1 つ並べるごとの時間（足し算 64 個・1 行 1536 個の正規化）
        let scaleAdd = try pipeline("scale_add"), norm = try pipeline("rms_norm")
        var small = GGUFMetalArgs()
        small.count = 64
        small.scale = 1
        let add = try time(repeats: 512) { encoder, _ in
            encoder.setComputePipelineState(scaleAdd)
            for slot in 0..<3 { encoder.setBuffer(dummy, offset: 0, index: slot) }
            encoder.setBytes(&small, length: MemoryLayout<GGUFMetalArgs>.stride, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        var normArgs = GGUFMetalArgs()
        normArgs.rows = 1
        normArgs.columns = 1536
        normArgs.epsilon = 1e-6
        let normalize = try time(repeats: 512) { encoder, _ in
            encoder.setComputePipelineState(norm)
            for slot in 0..<4 { encoder.setBuffer(slot == 0 ? input : dummy, offset: 0, index: slot) }
            encoder.setBytes(&normArgs, length: MemoryLayout<GGUFMetalArgs>.stride, index: 4)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        print(String(format: "KERNEL-DECODE small dispatches: scale_add (64) %.1f us, rms_norm (1x1536) %.1f us", add * 1e6, normalize * 1e6))
    }

    private func makeMatrix(_ device: MTLDevice, name: String, rows: Int, columns: Int, int8: Bool, copies: Int,
                            random: inout SplitMix64) throws -> Matrix {
        let weightBytes = int8 ? rows * columns : GGUFGroupQuant.nibbleBytes(rows: rows, columns: columns)
        let paramBytes = int8 ? rows * 4 : GGUFGroupQuant.paramBytes(rows: rows, columns: columns)
        // 倍率と下限: 本物に近い大きさ（q は 0 … 15、値 = q × 倍率 + 下限）
        var table: [UInt32] = []
        for _ in 0..<1024 {
            let u = Float(random.next() >> 40) / Float(1 << 24)
            if int8 {
                table.append((0.002 + 0.002 * u).bitPattern)
            } else {
                let scale: Float = 0.004 + 0.004 * u
                let low = UInt32(Half.fromFloat(scale)), high = UInt32(Half.fromFloat(-8 * scale * (0.9 + 0.2 * u)))
                table.append(low | high << 16)
            }
        }
        var weights: [MTLBuffer] = [], params: [MTLBuffer] = []
        for _ in 0..<copies {
            let w = try XCTUnwrap(device.makeBuffer(length: weightBytes, options: .storageModeShared))
            let words = w.contents().bindMemory(to: UInt64.self, capacity: weightBytes / 8)
            for i in 0..<(weightBytes / 8) { words[i] = random.next() }
            let p = try XCTUnwrap(device.makeBuffer(length: paramBytes, options: .storageModeShared))
            let values = p.contents().bindMemory(to: UInt32.self, capacity: paramBytes / 4)
            let offset = Int(random.next() % 1024)
            for i in 0..<(paramBytes / 4) { values[i] = table[(i &+ offset) & 1023] }
            weights.append(w)
            params.append(p)
        }
        return Matrix(name: name, rows: rows, columns: columns, int8: int8, weights: weights, params: params,
                      bytes: weightBytes + paramBytes)
    }

    private func measure(_ matrix: Matrix, candidates: [Candidate], reference: String, device: MTLDevice, queue: MTLCommandQueue,
                         pipeline: (String) throws -> MTLComputePipelineState, random: inout SplitMix64) throws {
        let maxCount = 8
        let input = try XCTUnwrap(device.makeBuffer(length: maxCount * matrix.columns * 4, options: .storageModeShared))
        let x = input.contents().bindMemory(to: Float.self, capacity: maxCount * matrix.columns)
        for i in 0..<(maxCount * matrix.columns) { x[i] = Float(Int(random.next() % 2001) - 1000) / 1000 }
        let output = try XCTUnwrap(device.makeBuffer(length: maxCount * matrix.rows * 4, options: .storageModeShared))
        let expected = try XCTUnwrap(device.makeBuffer(length: maxCount * matrix.rows * 4, options: .storageModeShared))
        let dummy = try XCTUnwrap(device.makeBuffer(length: 64, options: .storageModeShared))
        let kind = matrix.int8 ? "q8" : "q4"
        let shape = "\(matrix.name) \(matrix.rows)x\(matrix.columns)"

        func bindings(_ copy: Int, into target: MTLBuffer) -> [Binding] {
            if matrix.int8 {
                return [(matrix.weights[copy], 0), (matrix.params[copy], 0), (dummy, 0), (input, 0), (target, 0), (dummy, 0), (dummy, 0)]
            }
            return [(matrix.weights[copy], 0), (matrix.params[copy], 0), (dummy, 0), (input, 0), (target, 0), (matrix.params[copy], 0),
                    (dummy, 0)]
        }
        func arguments(_ count: Int) -> GGUFMetalArgs {
            var arguments = GGUFMetalArgs()
            arguments.rows = UInt32(matrix.rows)
            arguments.columns = UInt32(matrix.columns)
            arguments.count = UInt32(count)
            arguments.inStride = UInt32(matrix.columns)
            arguments.outStride = UInt32(matrix.rows)
            if matrix.int8 { arguments.flags = GGUFMetalSource.Flag.scales }
            return arguments
        }
        func encode(_ encoder: MTLComputeCommandEncoder, _ candidate: Candidate, count: Int, copy: Int, into target: MTLBuffer) throws {
            encoder.setComputePipelineState(try pipeline(candidate.function))
            for (index, binding) in bindings(copy, into: target).enumerated() {
                encoder.setBuffer(binding.buffer, offset: binding.offset, index: index)
            }
            var values = arguments(count)
            encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: 7)
            let groups = (matrix.rows + candidate.rowsPerGroup - 1) / candidate.rowsPerGroup
            let height = candidate.tokensPerGroup == 0 ? 1 : (count + candidate.tokensPerGroup - 1) / candidate.tokensPerGroup
            encoder.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: candidate.threads, height: 1, depth: 1))
        }
        /// 1 回の命令列で何回か計算して、1 回あたりの GPU の時間（2 回のうち短いほう、秒）
        func time(_ candidate: Candidate, count: Int) throws -> Double {
            let repeats = max(2, matrix.weights.count * 2)
            var best = Double.infinity
            for _ in 0..<2 {
                let buffer = try XCTUnwrap(queue.makeCommandBuffer())
                let encoder = try XCTUnwrap(buffer.makeComputeCommandEncoder())
                for index in 0..<repeats { try encode(encoder, candidate, count: count, copy: index % matrix.weights.count, into: output) }
                encoder.endEncoding()
                buffer.commit()
                buffer.waitUntilCompleted()
                XCTAssertNil(buffer.error, candidate.label)
                best = min(best, (buffer.gpuEndTime - buffer.gpuStartTime) / Double(repeats))
            }
            return best
        }
        func run(_ function: String, rowsPerGroup: Int, tokensPerGroup: Int, threads: Int, count: Int, into target: MTLBuffer) throws {
            let candidate = Candidate(label: function, function: function, rowsPerGroup: rowsPerGroup, tokensPerGroup: tokensPerGroup,
                                      threads: threads, counts: [count])
            let buffer = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(buffer.makeComputeCommandEncoder())
            try encode(encoder, candidate, count: count, copy: 0, into: target)
            encoder.endEncoding()
            buffer.commit()
            buffer.waitUntilCompleted()
        }
        /// FFN の gate と up を一度に（1 トークン。2 つの行列を読む）: 隣どうしの写しを gate と up にする
        let gateUp = try pipeline("gate_up_q4")
        func timeGateUp() throws -> Double {
            var best = Double.infinity
            let repeats = matrix.weights.count
            for _ in 0..<2 {
                let buffer = try XCTUnwrap(queue.makeCommandBuffer())
                let encoder = try XCTUnwrap(buffer.makeComputeCommandEncoder())
                for index in 0..<repeats {
                    let gate = index % matrix.weights.count, up = (index + 1) % matrix.weights.count
                    encoder.setComputePipelineState(gateUp)
                    let bindings: [Binding] = [(matrix.weights[gate], 0), (matrix.params[gate], 0), (matrix.weights[up], 0),
                                               (matrix.params[up], 0), (input, 0), (output, 0), (matrix.params[gate], 0),
                                               (matrix.params[up], 0)]
                    for (slot, binding) in bindings.enumerated() { encoder.setBuffer(binding.buffer, offset: binding.offset, index: slot) }
                    var values = arguments(1)
                    encoder.setBytes(&values, length: MemoryLayout<GGUFMetalArgs>.stride, index: 8)
                    let rowsPerGroup = GGUFMetalSource.gateUpRows * GGUFMetalSource.simdGroups
                    encoder.dispatchThreadgroups(MTLSize(width: (matrix.rows + rowsPerGroup - 1) / rowsPerGroup, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                }
                encoder.endEncoding()
                buffer.commit()
                buffer.waitUntilCompleted()
                best = min(best, (buffer.gpuEndTime - buffer.gpuStartTime) / Double(repeats))
            }
            return best
        }

        // まず全部の結果を、前からある形（行の組ごとに入力を読み直す）と比べる（GPU も温まる）
        var configurations: [(candidate: Candidate, count: Int, difference: Float)] = []
        for candidate in candidates {
            for count in candidate.counts {
                try run(reference, rowsPerGroup: 4, tokensPerGroup: 8, threads: 64, count: count, into: expected)
                memset(output.contents(), 0, count * matrix.rows * 4)
                try run(candidate.function, rowsPerGroup: candidate.rowsPerGroup, tokensPerGroup: candidate.tokensPerGroup,
                        threads: candidate.threads, count: count, into: output)
                let a = output.contents().bindMemory(to: Float.self, capacity: count * matrix.rows)
                let b = expected.contents().bindMemory(to: Float.self, capacity: count * matrix.rows)
                var difference: Float = 0, largest: Float = 0
                for i in 0..<(count * matrix.rows) {
                    difference = max(difference, abs(a[i] - b[i]))
                    largest = max(largest, abs(b[i]))
                }
                let relative = difference / max(largest, 1e-12)
                XCTAssertLessThan(relative, 1e-4, "\(kind) \(shape) \(candidate.label) n=\(count)")
                configurations.append((candidate, count, relative))
            }
        }
        // 先に測るものほど遅く出る（GPU の速さが上がりきっていない）ので、全部を順に 3 周して、それぞれいちばん速かった回を取る
        let withGateUp = !matrix.int8 && matrix.weights.count >= 2
        var best = [Double](repeating: .infinity, count: configurations.count)
        var gateUpBest = Double.infinity
        for _ in 0..<3 {
            for (index, configuration) in configurations.enumerated() {
                best[index] = min(best[index], try time(configuration.candidate, count: configuration.count))
            }
            if withGateUp { gateUpBest = min(gateUpBest, try timeGateUp()) }
        }
        let baseline = zip(configurations, best).first { $0.0.candidate.label == "app-rows" }?.1 ?? best[0]
        if withGateUp {
            print(String(format: "KERNEL %@ %@ n=1 app-gateup (2 matrices): %.3f ms = %.1f GB/s", kind, shape, gateUpBest * 1000,
                         Double(2 * matrix.bytes) / gateUpBest / 1e9))
        }
        for (configuration, seconds) in zip(configurations, best) {
            print(String(format: "KERNEL %@ %@ n=%d %@: %.3f ms = %.1f GB/s (x%.2f of app-rows n=1, per token x%.2f) diff %.1e",
                         kind, shape, configuration.count, configuration.candidate.label, seconds * 1000,
                         Double(matrix.bytes) / seconds / 1e9, seconds / max(baseline, 1e-12),
                         seconds / max(baseline, 1e-12) / Double(configuration.count), configuration.difference))
        }
    }

    /// 試しの書き方（GGUFMetalSource.code のあとにつなげる。Args・nibbles4・linear_store などはそちらのものを使う）
    static let source = """
    // ===== 速さを比べるための試しの書き方（アプリでは使わない）=====

    // 4 ビット・1 トークン: SIMD グループ 1 つで R 行。q はシフトして取り出す（前のアプリの書き方）
    template <uint R>
    kernel void xq4_one(device const uchar *nibbles [[buffer(0)]],
                        device const half2 *params [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device const float *input [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        device const uint *high [[buffer(5)]],
                        device const float *aux [[buffer(6)]],
                        constant Args &a [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group * SIMD_GROUPS + simd) * R;
        if (first >= a.rows) { return; }
        uint rows = min(R, a.rows - first);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        device const uint2 *words = (device const uint2 *)nibbles;
        float acc[R];
        for (uint r = 0u; r < R; r++) { acc[r] = 0.0f; }
        for (uint h = lane; h < halves; h += 32u) {
            float4 x0 = float4(x[h * 4u]);
            float4 x1 = float4(x[h * 4u + 1u]);
            float4 x2 = float4(x[h * 4u + 2u]);
            float4 x3 = float4(x[h * 4u + 3u]);
            float xs = sum4(x0 + x1 + x2 + x3);
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            for (uint r = 0u; r < R; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                float2 p = float2(params[b]);
                acc[r] += p.x * half_dot(words[b * 2u + part], 0u, false, x0, x1, x2, x3) + p.y * xs;
            }
        }
        for (uint r = 0u; r < R; r++) {
            float sum = simd_sum(acc[r]);
            if (lane == 0u && r < rows) { linear_store(output, bias, aux, a, first + r, 0u, sum); }
        }
    }

    typedef decltype(xq4_one<2>) xq4_one_t;
    template [[host_name("xq4_one2")]] kernel xq4_one_t xq4_one<2>;

    // 4 ビット・1 トークン: SIMD グループ 1 つで R 行。q はマスクだけで取り出す（アプリと同じ masked_dot）
    template <uint R>
    kernel void xq4_mask(device const uchar *nibbles [[buffer(0)]],
                         device const half2 *params [[buffer(1)]],
                         device const float *bias [[buffer(2)]],
                         device const float *input [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         device const uint *high [[buffer(5)]],
                         device const float *aux [[buffer(6)]],
                         constant Args &a [[buffer(7)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group * SIMD_GROUPS + simd) * R;
        if (first >= a.rows) { return; }
        uint rows = min(R, a.rows - first);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        device const uint2 *words = (device const uint2 *)nibbles;
        float acc[R];
        for (uint r = 0u; r < R; r++) { acc[r] = 0.0f; }
        for (uint h = lane; h < halves; h += 32u) {
            float4 x0 = float4(x[h * 4u]);
            float4 x1 = float4(x[h * 4u + 1u]);
            float4 x2 = float4(x[h * 4u + 2u]);
            float4 x3 = float4(x[h * 4u + 3u]);
            float xs = sum4(x0 + x1 + x2 + x3);
            float4 y0 = x0 * NIBBLE_SCALES;
            float4 y1 = x1 * NIBBLE_SCALES;
            float4 y2 = x2 * NIBBLE_SCALES;
            float4 y3 = x3 * NIBBLE_SCALES;
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            for (uint r = 0u; r < R; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                float2 p = float2(params[b]);
                acc[r] += p.x * masked_dot(words[b * 2u + part], y0, y1, y2, y3) + p.y * xs;
            }
        }
        for (uint r = 0u; r < R; r++) {
            float sum = simd_sum(acc[r]);
            if (lane == 0u && r < rows) { linear_store(output, bias, aux, a, first + r, 0u, sum); }
        }
    }

    typedef decltype(xq4_mask<2>) xq4_mask_t;
    template [[host_name("xq4_mask4")]] kernel xq4_mask_t xq4_mask<4>;

    // 4 ビット・1 トークン: threadgroup 1 つで 2 行。S 個の SIMD グループで列（組の半分）を分けて計算し、最後に threadgroup で足す。
    // 行の少ない行列でも、SIMD グループの数が S 倍になる
    template <uint S>
    kernel void xq4_split(device const uchar *nibbles [[buffer(0)]],
                          device const half2 *params [[buffer(1)]],
                          device const float *bias [[buffer(2)]],
                          device const float *input [[buffer(3)]],
                          device float *output [[buffer(4)]],
                          device const uint *high [[buffer(5)]],
                          device const float *aux [[buffer(6)]],
                          constant Args &a [[buffer(7)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float partial[S][2];
        uint first = group * 2u;
        if (first >= a.rows) { return; }
        uint rows = min(2u, a.rows - first);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        device const uint2 *words = (device const uint2 *)nibbles;
        float acc[2] = {0.0f, 0.0f};
        for (uint h = simd * 32u + lane; h < halves; h += 32u * S) {
            float4 x0 = float4(x[h * 4u]);
            float4 x1 = float4(x[h * 4u + 1u]);
            float4 x2 = float4(x[h * 4u + 2u]);
            float4 x3 = float4(x[h * 4u + 3u]);
            float xs = sum4(x0 + x1 + x2 + x3);
            float4 y0 = x0 * NIBBLE_SCALES;
            float4 y1 = x1 * NIBBLE_SCALES;
            float4 y2 = x2 * NIBBLE_SCALES;
            float4 y3 = x3 * NIBBLE_SCALES;
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            for (uint r = 0u; r < 2u; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                float2 p = float2(params[b]);
                acc[r] += p.x * masked_dot(words[b * 2u + part], y0, y1, y2, y3) + p.y * xs;
            }
        }
        for (uint r = 0u; r < 2u; r++) {
            float sum = simd_sum(acc[r]);
            if (lane == 0u) { partial[simd][r] = sum; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u && lane < rows) {
            float sum = 0.0f;
            for (uint s = 0u; s < S; s++) { sum += partial[s][lane]; }
            linear_store(output, bias, aux, a, first + lane, 0u, sum);
        }
    }

    typedef decltype(xq4_split<2>) xq4_split_t;
    template [[host_name("xq4_split2")]] kernel xq4_split_t xq4_split<2>;
    template [[host_name("xq4_split4")]] kernel xq4_split_t xq4_split<4>;
    template [[host_name("xq4_split8")]] kernel xq4_split_t xq4_split<8>;

    // 8 ビット・1 トークン: SIMD グループ 1 つで R 行（アプリは Q8_ROWS 行）
    template <uint R>
    kernel void xq8_one(device const char *weights [[buffer(0)]],
                        device const float *scales [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device const float *input [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        device const float *aux [[buffer(6)]],
                        constant Args &a [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group * SIMD_GROUPS + simd) * R;
        if (first >= a.rows) { return; }
        uint rows = min(R, a.rows - first);
        uint chunks = a.columns >> 4u;
        device const uint4 *w = (device const uint4 *)(weights + a.weightSkip);
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        float acc[R];
        for (uint r = 0u; r < R; r++) { acc[r] = 0.0f; }
        for (uint c = lane; c < chunks; c += 32u) {
            float4 x0 = float4(x[c * 4u]);
            float4 x1 = float4(x[c * 4u + 1u]);
            float4 x2 = float4(x[c * 4u + 2u]);
            float4 x3 = float4(x[c * 4u + 3u]);
            for (uint r = 0u; r < R; r++) { acc[r] += dot16(w[(first + min(r, rows - 1u)) * chunks + c], x0, x1, x2, x3); }
        }
        for (uint r = 0u; r < R; r++) {
            float sum = simd_sum(acc[r]);
            if (lane == 0u && r < rows) {
                if ((a.flags & 1u) != 0u) { sum *= scales[first + r]; }
                linear_store(output, bias, aux, a, first + r, 0u, sum);
            }
        }
    }

    typedef decltype(xq8_one<2>) xq8_one_t;
    template [[host_name("xq8_one1")]] kernel xq8_one_t xq8_one<1>;
    template [[host_name("xq8_one4")]] kernel xq8_one_t xq8_one<4>;
    """
}
#endif
