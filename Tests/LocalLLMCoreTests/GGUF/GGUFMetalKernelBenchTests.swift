#if canImport(Metal)
import Metal
import XCTest
@testable import LocalLLMCore

/// 線形層の計算のしかたごとの GPU の時間を、Gemma 4 E2B と同じ大きさの行列で測る（速さを測る流れでだけ動く）。
/// アプリの計算（rows・few・matrix・legacy）と、このファイルの試しの書き方を、まとめるトークンの数ごとに比べる。
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
            Candidate(label: "app-rows", function: "linear_q4_rows", rowsPerGroup: 8, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "app-few", function: "linear_q4_tokens", rowsPerGroup: 8, tokensPerGroup: 8, threads: 64, counts: [2, 3, 4, 5, 8]),
            Candidate(label: "app-legacy", function: "linear_q4_batch", rowsPerGroup: 4, tokensPerGroup: 8, threads: 64, counts: [2, 4, 8]),
            Candidate(label: "x-one2", function: "xq4_one2", rowsPerGroup: 4, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-one8", function: "xq4_one8", rowsPerGroup: 16, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-mask4", function: "xq4_mask4", rowsPerGroup: 8, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-few2", function: "xq4_few2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2, 4]),
            Candidate(label: "x-few3", function: "xq4_few3", rowsPerGroup: 8, tokensPerGroup: 3, threads: 64, counts: [3]),
            Candidate(label: "x-few4", function: "xq4_few4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [2, 3, 4, 8]),
            Candidate(label: "x-few8", function: "xq4_few8", rowsPerGroup: 8, tokensPerGroup: 8, threads: 64, counts: [5, 8]),
            Candidate(label: "x-fewh2", function: "xq4_fewh2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2]),
            Candidate(label: "x-fewh4", function: "xq4_fewh4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [4]),
            Candidate(label: "x-fewh8", function: "xq4_fewh8", rowsPerGroup: 8, tokensPerGroup: 8, threads: 64, counts: [8]),
            Candidate(label: "x-row2", function: "xq4_row2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2]),
            Candidate(label: "x-row4", function: "xq4_row4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [4]),
            Candidate(label: "x-row8", function: "xq4_row8", rowsPerGroup: 8, tokensPerGroup: 8, threads: 64, counts: [8]),
        ] + (matrices ? [Candidate(label: "app-matrix", function: "linear_q4_mma", rowsPerGroup: 32, tokensPerGroup: 8, threads: 128,
                                   counts: [2, 4, 8])] : [])
        let q8: [Candidate] = [
            Candidate(label: "app-rows", function: "linear_q8_rows", rowsPerGroup: 8, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "app-few", function: "linear_q8_few", rowsPerGroup: 8, tokensPerGroup: 8, threads: 64, counts: [2, 3, 4, 8]),
            Candidate(label: "app-legacy", function: "linear_q8_tokens", rowsPerGroup: 4, tokensPerGroup: 8, threads: 64, counts: [2, 4]),
            Candidate(label: "x-one2", function: "xq8_one2", rowsPerGroup: 4, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-one8", function: "xq8_one8", rowsPerGroup: 16, tokensPerGroup: 0, threads: 64, counts: [1]),
            Candidate(label: "x-few2", function: "xq8_few2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2]),
            Candidate(label: "x-few3", function: "xq8_few3", rowsPerGroup: 8, tokensPerGroup: 3, threads: 64, counts: [3]),
            Candidate(label: "x-few4", function: "xq8_few4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [2, 3, 4]),
            Candidate(label: "x-few8", function: "xq8_few8", rowsPerGroup: 8, tokensPerGroup: 8, threads: 64, counts: [8]),
            Candidate(label: "x-row2", function: "xq8_row2", rowsPerGroup: 8, tokensPerGroup: 2, threads: 64, counts: [2]),
            Candidate(label: "x-row4", function: "xq8_row4", rowsPerGroup: 8, tokensPerGroup: 4, threads: 64, counts: [4]),
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
        /// 1 回の命令列で何回か計算して、1 回あたりの GPU の時間（いちばん短かったもの、秒）
        func time(_ candidate: Candidate, count: Int) throws -> Double {
            let repeats = max(2, matrix.weights.count * 2)
            var best = Double.infinity
            for _ in 0..<3 {
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

        var baseline = 0.0
        if !matrix.int8 && matrix.weights.count >= 2 {
            // FFN の gate と up を一度に（1 トークン。2 つの行列を読む）: 隣どうしの写しを gate と up にする
            let gateUp = try pipeline("gate_up_q4")
            var best = Double.infinity
            let repeats = matrix.weights.count
            for _ in 0..<3 {
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
            print(String(format: "KERNEL %@ %@ n=1 app-gateup (2 matrices): %.3f ms = %.1f GB/s", kind, shape, best * 1000,
                         Double(2 * matrix.bytes) / best / 1e9))
        }
        for candidate in candidates {
            for count in candidate.counts {
                // 前からある形（1 行ずつ入力を読み直す）の結果と比べる
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

                let seconds = try time(candidate, count: count)
                if candidate.label == "app-rows" { baseline = seconds }
                print(String(format: "KERNEL %@ %@ n=%d %@: %.3f ms = %.1f GB/s (x%.2f of app-rows n=1, per token x%.2f) diff %.1e",
                             kind, shape, count, candidate.label, seconds * 1000, Double(matrix.bytes) / seconds / 1e9,
                             seconds / max(baseline, 1e-12), seconds / max(baseline, 1e-12) / Double(count), relative))
            }
        }
    }

    /// 試しの書き方（GGUFMetalSource.code のあとにつなげる。Args・nibbles4・linear_store などはそちらのものを使う）
    static let source = """
    // ===== 速さを比べるための試しの書き方（アプリでは使わない）=====

    // 4 ビット・1 トークン: SIMD グループ 1 つで R 行（アプリは 4 行）
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

    typedef decltype(xq4_one<4>) xq4_one_t;
    template [[host_name("xq4_one2")]] kernel xq4_one_t xq4_one<2>;
    template [[host_name("xq4_one8")]] kernel xq4_one_t xq4_one<8>;

    // 組の半分（16 個）の Σ q·x を、ビットを取り出す位置に合わせて先に割っておいた入力で（シフトせずにマスクだけで q を取り出す）。
    // y = x × (1, 1/16, 1/256, 1/4096)
    inline float masked_dot(uint2 w, float4 y0, float4 y1, float4 y2, float4 y3) {
        const uint4 m = uint4(0xFu, 0xF0u, 0xF00u, 0xF000u);
        float4 a0 = float4(uint4(w.x) & m);
        float4 a1 = float4(uint4(w.x >> 16u) & m);
        float4 a2 = float4(uint4(w.y) & m);
        float4 a3 = float4(uint4(w.y >> 16u) & m);
        return dot(a0, y0) + dot(a1, y1) + dot(a2, y2) + dot(a3, y3);
    }

    // 4 ビット・1 トークン・SIMD グループ 1 つで 4 行。q の取り出しはマスクだけ
    kernel void xq4_mask4(device const uchar *nibbles [[buffer(0)]],
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
        uint first = (group * SIMD_GROUPS + simd) * 4u;
        if (first >= a.rows) { return; }
        uint rows = min(4u, a.rows - first);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        device const uint2 *words = (device const uint2 *)nibbles;
        const float4 shifts = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint h = lane; h < halves; h += 32u) {
            float4 x0 = float4(x[h * 4u]);
            float4 x1 = float4(x[h * 4u + 1u]);
            float4 x2 = float4(x[h * 4u + 2u]);
            float4 x3 = float4(x[h * 4u + 3u]);
            float xs = sum4(x0 + x1 + x2 + x3);
            float4 y0 = x0 * shifts;
            float4 y1 = x1 * shifts;
            float4 y2 = x2 * shifts;
            float4 y3 = x3 * shifts;
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            for (uint r = 0u; r < 4u; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                float2 p = float2(params[b]);
                acc[r] += p.x * masked_dot(words[b * 2u + part], y0, y1, y2, y3) + p.y * xs;
            }
        }
        for (uint r = 0u; r < 4u; r++) {
            float sum = simd_sum(acc[r]);
            if (lane == 0u && r < rows) { linear_store(output, bias, aux, a, first + r, 0u, sum); }
        }
    }

    // 4 ビット・N トークン: SIMD グループ 1 つで 4 行。組の半分の重みを 1 回だけ float にして、N トークンに使う（入力はトークンごとに 1 回読む）
    template <uint N>
    kernel void xq4_few(device const uchar *nibbles [[buffer(0)]],
                        device const half2 *params [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device const float *input [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        device const uint *high [[buffer(5)]],
                        device const float *aux [[buffer(6)]],
                        constant Args &a [[buffer(7)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * 4u;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(4u, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const uint2 *words = (device const uint2 *)nibbles;
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[4][N];
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint h = lane; h < halves; h += 32u) {
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            float4 w0[4];
            float4 w1[4];
            float4 w2[4];
            float4 w3[4];
            float2 p[4];
            for (uint r = 0u; r < 4u; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                uint2 q = words[b * 2u + part];
                w0[r] = nibbles4(q.x, 0u);
                w1[r] = nibbles4(q.x, 16u);
                w2[r] = nibbles4(q.y, 0u);
                w3[r] = nibbles4(q.y, 16u);
                p[r] = float2(params[b]);
            }
            for (uint t = 0u; t < N; t++) {
                device const packed_float4 *x = (device const packed_float4 *)(base + min(t, tokens - 1u) * a.inStride) + h * 4u;
                float4 x0 = float4(x[0]);
                float4 x1 = float4(x[1]);
                float4 x2 = float4(x[2]);
                float4 x3 = float4(x[3]);
                float xs = sum4(x0 + x1 + x2 + x3);
                for (uint r = 0u; r < 4u; r++) {
                    float qx = dot(w0[r], x0) + dot(w1[r], x1) + dot(w2[r], x2) + dot(w3[r], x3);
                    acc[r][t] += p[r].x * qx + p[r].y * xs;
                }
            }
        }
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum); }
            }
        }
    }

    typedef decltype(xq4_few<2>) xq4_few_t;
    template [[host_name("xq4_few2")]] kernel xq4_few_t xq4_few<2>;
    template [[host_name("xq4_few3")]] kernel xq4_few_t xq4_few<3>;
    template [[host_name("xq4_few4")]] kernel xq4_few_t xq4_few<4>;
    template [[host_name("xq4_few8")]] kernel xq4_few_t xq4_few<8>;

    inline half4 nibbles4h(uint word, uint shift) {
        return half4(half((word >> shift) & 15u), half((word >> (shift + 4u)) & 15u),
                     half((word >> (shift + 8u)) & 15u), half((word >> (shift + 12u)) & 15u));
    }

    // xq4_few と同じで、ほどいた重み（0 … 15 の整数なので half でちょうど表せる）を half で持つ（レジスタを減らす）
    template <uint N>
    kernel void xq4_fewh(device const uchar *nibbles [[buffer(0)]],
                         device const half2 *params [[buffer(1)]],
                         device const float *bias [[buffer(2)]],
                         device const float *input [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         device const uint *high [[buffer(5)]],
                         device const float *aux [[buffer(6)]],
                         constant Args &a [[buffer(7)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * 4u;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(4u, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const uint2 *words = (device const uint2 *)nibbles;
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[4][N];
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint h = lane; h < halves; h += 32u) {
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            half4 w0[4];
            half4 w1[4];
            half4 w2[4];
            half4 w3[4];
            float2 p[4];
            for (uint r = 0u; r < 4u; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                uint2 q = words[b * 2u + part];
                w0[r] = nibbles4h(q.x, 0u);
                w1[r] = nibbles4h(q.x, 16u);
                w2[r] = nibbles4h(q.y, 0u);
                w3[r] = nibbles4h(q.y, 16u);
                p[r] = float2(params[b]);
            }
            for (uint t = 0u; t < N; t++) {
                device const packed_float4 *x = (device const packed_float4 *)(base + min(t, tokens - 1u) * a.inStride) + h * 4u;
                float4 x0 = float4(x[0]);
                float4 x1 = float4(x[1]);
                float4 x2 = float4(x[2]);
                float4 x3 = float4(x[3]);
                float xs = sum4(x0 + x1 + x2 + x3);
                for (uint r = 0u; r < 4u; r++) {
                    float qx = dot(float4(w0[r]), x0) + dot(float4(w1[r]), x1) + dot(float4(w2[r]), x2) + dot(float4(w3[r]), x3);
                    acc[r][t] += p[r].x * qx + p[r].y * xs;
                }
            }
        }
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum); }
            }
        }
    }

    typedef decltype(xq4_fewh<2>) xq4_fewh_t;
    template [[host_name("xq4_fewh2")]] kernel xq4_fewh_t xq4_fewh<2>;
    template [[host_name("xq4_fewh4")]] kernel xq4_fewh_t xq4_fewh<4>;
    template [[host_name("xq4_fewh8")]] kernel xq4_fewh_t xq4_fewh<8>;

    // 4 ビット・N トークン: 行ごとに重みをほどいて、N トークンに使う（入力は行ごとに読み直す。レジスタは少ない）
    template <uint N>
    kernel void xq4_row(device const uchar *nibbles [[buffer(0)]],
                        device const half2 *params [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device const float *input [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        device const uint *high [[buffer(5)]],
                        device const float *aux [[buffer(6)]],
                        constant Args &a [[buffer(7)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * 4u;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(4u, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const uint2 *words = (device const uint2 *)nibbles;
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[4][N];
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint h = lane; h < halves; h += 32u) {
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            for (uint r = 0u; r < 4u; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                uint2 q = words[b * 2u + part];
                float2 p = float2(params[b]);
                float4 w0 = nibbles4(q.x, 0u);
                float4 w1 = nibbles4(q.x, 16u);
                float4 w2 = nibbles4(q.y, 0u);
                float4 w3 = nibbles4(q.y, 16u);
                for (uint t = 0u; t < N; t++) {
                    device const packed_float4 *x = (device const packed_float4 *)(base + min(t, tokens - 1u) * a.inStride) + h * 4u;
                    float4 x0 = float4(x[0]);
                    float4 x1 = float4(x[1]);
                    float4 x2 = float4(x[2]);
                    float4 x3 = float4(x[3]);
                    float xs = sum4(x0 + x1 + x2 + x3);
                    float qx = dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);
                    acc[r][t] += p.x * qx + p.y * xs;
                }
            }
        }
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum); }
            }
        }
    }

    typedef decltype(xq4_row<2>) xq4_row_t;
    template [[host_name("xq4_row2")]] kernel xq4_row_t xq4_row<2>;
    template [[host_name("xq4_row4")]] kernel xq4_row_t xq4_row<4>;
    template [[host_name("xq4_row8")]] kernel xq4_row_t xq4_row<8>;

    // 8 ビット・1 トークン: SIMD グループ 1 つで R 行（アプリは 4 行）
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

    typedef decltype(xq8_one<4>) xq8_one_t;
    template [[host_name("xq8_one2")]] kernel xq8_one_t xq8_one<2>;
    template [[host_name("xq8_one8")]] kernel xq8_one_t xq8_one<8>;

    // 8 ビット・N トークン: SIMD グループ 1 つで 4 行。16 個ずつの重みを 1 回だけ float にして N トークンに使う
    template <uint N>
    kernel void xq8_few(device const char *weights [[buffer(0)]],
                        device const float *scales [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device const float *input [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        device const float *aux [[buffer(6)]],
                        constant Args &a [[buffer(7)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * 4u;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(4u, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint chunks = a.columns >> 4u;
        device const uint4 *w = (device const uint4 *)(weights + a.weightSkip);
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[4][N];
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint c = lane; c < chunks; c += 32u) {
            float4 w0[4];
            float4 w1[4];
            float4 w2[4];
            float4 w3[4];
            for (uint r = 0u; r < 4u; r++) {
                uint4 q = w[(first + min(r, rows - 1u)) * chunks + c];
                w0[r] = float4(as_type<char4>(q.x));
                w1[r] = float4(as_type<char4>(q.y));
                w2[r] = float4(as_type<char4>(q.z));
                w3[r] = float4(as_type<char4>(q.w));
            }
            for (uint t = 0u; t < N; t++) {
                device const packed_float4 *x = (device const packed_float4 *)(base + min(t, tokens - 1u) * a.inStride) + c * 4u;
                float4 x0 = float4(x[0]);
                float4 x1 = float4(x[1]);
                float4 x2 = float4(x[2]);
                float4 x3 = float4(x[3]);
                for (uint r = 0u; r < 4u; r++) {
                    acc[r][t] += dot(w0[r], x0) + dot(w1[r], x1) + dot(w2[r], x2) + dot(w3[r], x3);
                }
            }
        }
        for (uint r = 0u; r < 4u; r++) {
            float scale = ((a.flags & 1u) != 0u) ? scales[first + min(r, rows - 1u)] : 1.0f;
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum * scale); }
            }
        }
    }

    typedef decltype(xq8_few<2>) xq8_few_t;
    template [[host_name("xq8_few2")]] kernel xq8_few_t xq8_few<2>;
    template [[host_name("xq8_few3")]] kernel xq8_few_t xq8_few<3>;
    template [[host_name("xq8_few4")]] kernel xq8_few_t xq8_few<4>;
    template [[host_name("xq8_few8")]] kernel xq8_few_t xq8_few<8>;

    // 8 ビット・N トークン: 行ごとに 16 個の重みを float にして N トークンに使う（入力は行ごとに読み直す）
    template <uint N>
    kernel void xq8_row(device const char *weights [[buffer(0)]],
                        device const float *scales [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device const float *input [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        device const float *aux [[buffer(6)]],
                        constant Args &a [[buffer(7)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * 4u;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(4u, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint chunks = a.columns >> 4u;
        device const uint4 *w = (device const uint4 *)(weights + a.weightSkip);
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[4][N];
        for (uint r = 0u; r < 4u; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint c = lane; c < chunks; c += 32u) {
            for (uint r = 0u; r < 4u; r++) {
                uint4 q = w[(first + min(r, rows - 1u)) * chunks + c];
                float4 w0 = float4(as_type<char4>(q.x));
                float4 w1 = float4(as_type<char4>(q.y));
                float4 w2 = float4(as_type<char4>(q.z));
                float4 w3 = float4(as_type<char4>(q.w));
                for (uint t = 0u; t < N; t++) {
                    device const packed_float4 *x = (device const packed_float4 *)(base + min(t, tokens - 1u) * a.inStride) + c * 4u;
                    acc[r][t] += dot(w0, float4(x[0])) + dot(w1, float4(x[1])) + dot(w2, float4(x[2])) + dot(w3, float4(x[3]));
                }
            }
        }
        for (uint r = 0u; r < 4u; r++) {
            float scale = ((a.flags & 1u) != 0u) ? scales[first + min(r, rows - 1u)] : 1.0f;
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum * scale); }
            }
        }
    }

    typedef decltype(xq8_row<2>) xq8_row_t;
    template [[host_name("xq8_row2")]] kernel xq8_row_t xq8_row<2>;
    template [[host_name("xq8_row4")]] kernel xq8_row_t xq8_row<4>;
    """
}
#endif
