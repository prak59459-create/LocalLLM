import Foundation

/// GPU（Metal）で GGUF の推論を計算するときの、計算の中身（Metal Shading Language）。
/// アプリの中で起動時にコンパイルする（Swift Playgrounds は .metal ファイルをビルドできないため）。
///
/// 1 トークンずつの生成は、ほとんどの時間が「重みを読む」こと（メモリの速さ）で決まる。そこで:
/// - 線形層は、1 つの SIMD グループ（32 スレッド）で何行かをまとめて計算し、どのスレッドにも同じだけの仕事を割り当てる
///   （4・5 ビットは組の半分 = 16 個ずつ）。隣どうしのスレッドが隣どうしの重みを読む。
/// - 正規化・softmax・注意の足し合わせは、1 行を threadgroup のスレッド全部で分けて計算する（1 スレッドで回さない）。
/// - 小さな計算はつなげて、GPU に渡す命令の数を減らす（正規化と残差の足し算、q・k の正規化と RoPE、FFN の gate と up）。
enum GGUFMetalSource {
    /// 計算の名前（kernel の関数名）
    enum Kernel: String, CaseIterable {
        case linearQ8 = "linear_q8"
        case linearQ8Batch = "linear_q8_batch"
        case linearF32 = "linear_f32"
        case linearF32Batch = "linear_f32_batch"
        case linearQ8Vector = "linear_q8_vector"
        case linearQ8Rows = "linear_q8_rows"
        case linearQ8Tokens = "linear_q8_tokens"
        case argmax = "argmax"
        case embeddingRow = "embedding_row"
        /// 少しまとめて（数字はコンパイル時に決めたトークンの数）
        case linearQ4Few2 = "linear_q4_few2"
        case linearQ4Few3 = "linear_q4_few3"
        case linearQ4Few4 = "linear_q4_few4"
        case linearQ5Few2 = "linear_q5_few2"
        case linearQ5Few3 = "linear_q5_few3"
        case linearQ5Few4 = "linear_q5_few4"
        case linearQ8Few2 = "linear_q8_few2"
        case linearQ8Few3 = "linear_q8_few3"
        case linearQ8Few4 = "linear_q8_few4"
        case linearQ4Matrix = "linear_q4_mma"
        case linearQ5Matrix = "linear_q5_mma"
        case linearQ8Matrix = "linear_q8_mma"

        /// simdgroup の行列の計算を使うもの（Apple7 以降の GPU だけ。matrixCode にある）
        static let matrixKernels: Set<Kernel> = [.linearQ4Matrix, .linearQ5Matrix, .linearQ8Matrix]

        /// 少しまとめての計算（bits は 4・5・8、tokens は 2 … fewTokens。コンパイル時にその数に決めたもの）
        static func few(bits: Int, tokens: Int) -> Kernel {
            let kernels: [Kernel]
            switch bits {
            case 4: kernels = [.linearQ4Few2, .linearQ4Few3, .linearQ4Few4]
            case 5: kernels = [.linearQ5Few2, .linearQ5Few3, .linearQ5Few4]
            default: kernels = [.linearQ8Few2, .linearQ8Few3, .linearQ8Few4]
            }
            return kernels[Swift.max(2, Swift.min(tokens, GGUFMetalSource.fewTokens)) - 2]
        }
        case linearF32Vector = "linear_f32_vector"
        case linearQ4Vector = "linear_q4_vector"
        case linearQ5Vector = "linear_q5_vector"
        case linearQ4Rows = "linear_q4_rows"
        case linearQ5Rows = "linear_q5_rows"
        case linearQ4Batch = "linear_q4_batch"
        case linearQ5Batch = "linear_q5_batch"
        case gateUpQ4 = "gate_up_q4"
        case gateUpQ5 = "gate_up_q5"
        case rmsNorm = "rms_norm"
        case headNormRope = "head_norm_rope"
        case rope = "rope"
        case attentionScores = "attention_scores"
        case attentionSoftmax = "attention_softmax"
        case attentionCombine = "attention_combine"
        case scaleAdd = "scale_add"
        case gatedActivation = "gated_activation"
        case convolutionPrepare = "convolution_prepare"
        case convolutionApply = "convolution_apply"
    }

    /// Args.flags の意味
    enum Flag {
        static let scales: UInt32 = 1
        static let bias: UInt32 = 2
        static let splitPairs: UInt32 = 4
        static let gelu: UInt32 = 8
        /// 正規化: 重みを掛ける・バイアスを足す・LayerNorm（平均も引く）
        static let normWeight: UInt32 = 16
        static let normBias: UInt32 = 32
        static let layerNorm: UInt32 = 64
        /// 活性化: up と掛けない（gate の無い FFN）・ReLU²
        static let noUp: UInt32 = 128
        static let reluSquared: UInt32 = 256
        /// scale_add: 2 つ目の入力を足す
        static let second: UInt32 = 512
        /// 出力に足す: 線形層は 出力 += 値 × extra、正規化は 出力 = (出力 + 値 × extra) × scale
        static let add: UInt32 = 1024
        /// 線形層の出力を gelu(値) × aux にする（Gemma 4 の層ごとの入力）
        static let geluTimesAux: UInt32 = 2048
        /// head_norm_rope: RoPE を先に（正規化はあと）
        static let ropeFirst: UInt32 = 4096
    }

    /// 線形層の SIMD グループ 1 つが受け持つ行（1 トークン・まとめて）と、threadgroup の SIMD グループの数
    static let vectorRows = 4
    /// 8 ビットの線形層（1 トークン）の、SIMD グループ 1 つが受け持つ行（2 行のほうが 4 行より速かった）
    static let q8Rows = 2
    static let batchRows = 2
    static let gateUpRows = 2
    static let simdGroups = 2
    /// まとめて計算するときの、1 回のトークン数
    static let batchTokens = 8
    /// 少しまとめて（先読みの確かめなど）の、SIMD グループ 1 つが受け持つ行と、1 回で受け持つトークンの数の上限
    /// （トークンの数をコンパイル時に決めた計算がある数。それより多いときは何回かに分ける）
    static let fewRows = 4
    static let fewTokens = 4
    /// 行列の掛け算（simdgroup の 8×8）でまとめて計算するときの、threadgroup 1 つが受け持つ行・トークンと、スレッドの数
    static let matrixRows = 32
    static let matrixTokens = 8
    static let matrixThreads = 128

    static let code = """
    #include <metal_stdlib>
    using namespace metal;

    // Swift 側の GGUFMetalArgs と同じ並び（すべて 4 バイト）
    struct Args {
        uint rows;
        uint columns;
        uint count;
        uint inOffset;
        uint inStride;
        uint outOffset;
        uint outStride;
        uint weightSkip;
        uint flags;
        uint heads;
        uint kvHeads;
        uint size;
        uint total;
        uint start;
        uint window;
        uint pairs;
        float epsilon;
        float scale;
        float softcap;
        float extra;
        uint auxOffset;
        uint auxStride;
    };

    #define VECTOR_ROWS 4u
    #define FEW_ROWS 4u
    #define Q8_ROWS 2u
    #define BATCH_ROWS 2u
    #define GATE_UP_ROWS 2u
    #define SIMD_GROUPS 2u
    #define BATCH_TOKENS 8u

    inline float4 load4(device const packed_char4 *w, uint i) { return float4(char4(w[i])); }
    inline float4 load4(device const packed_float4 *w, uint i) { return float4(w[i]); }

    inline float sum4(float4 v) { return v.x + v.y + v.z + v.w; }

    // 大きな値でも NaN にならない tanh（±15 より外は ±1 と見なせる）
    inline float safe_tanh(float v) { return tanh(clamp(v, -15.0f, 15.0f)); }

    inline float gelu(float g) {
        return 0.5f * g * (1.0f + safe_tanh(0.7978845608028654f * g * (1.0f + 0.044715f * g * g)));
    }

    inline float silu(float g) { return g / (1.0f + exp(min(-g, 80.0f))); }

    // FFN の活性化（flags: 8 = GELU、256 = ReLU²、ほかは SiLU）
    inline float activation(float g, uint flags) {
        if ((flags & 8u) != 0u) { return gelu(g); }
        if ((flags & 256u) != 0u) { float r = max(g, 0.0f); return r * r; }
        return silu(g);
    }

    // threadgroup のスレッド全部での合計・最大（shared は SIMD グループの数以上）
    inline float group_sum(float v, threadgroup float *shared, uint lane, uint simd, uint simds) {
        v = simd_sum(v);
        if (simds == 1u) { return v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { shared[simd] = v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < simds; i++) { total += shared[i]; }
        return total;
    }

    inline float group_max(float v, threadgroup float *shared, uint lane, uint simd, uint simds) {
        v = simd_max(v);
        if (simds == 1u) { return v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { shared[simd] = v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = shared[0];
        for (uint i = 1u; i < simds; i++) { best = max(best, shared[i]); }
        return best;
    }

    // 線形層の出力を書く（バイアス、gelu × aux、出力に足す）。token = まとめた中の何番目か
    inline void linear_store(device float *output, device const float *bias, device const float *aux,
                             constant Args &a, uint row, uint token, float sum) {
        if ((a.flags & 2u) != 0u) { sum += bias[row]; }
        if ((a.flags & 2048u) != 0u) { sum = gelu(sum) * aux[a.auxOffset + token * a.auxStride + row]; }
        uint index = a.outOffset + token * a.outStride + row;
        if ((a.flags & 1024u) != 0u) { sum = output[index] + sum * a.extra; }
        output[index] = sum;
    }

    // 1 行の重みと入力の内積（4 個ずつ）
    template <typename W>
    inline float dot_row(device const W *w, device const packed_float4 *x, uint quads) {
        float4 a0 = float4(0.0f);
        float4 a1 = float4(0.0f);
        uint i = 0;
        for (; i + 1 < quads; i += 2) {
            a0 += load4(w, i) * float4(x[i]);
            a1 += load4(w, i + 1) * float4(x[i + 1]);
        }
        if (i < quads) {
            a0 += load4(w, i) * float4(x[i]);
        }
        return sum4(a0 + a1);
    }

    // 線形層。gid.x = 出力の行、gid.y = トークン
    template <typename W>
    inline void linear_one(device const W *weights, device const float *scales, device const float *bias,
                           device const float *input, device float *output, device const float *aux,
                           constant Args &a, uint2 gid) {
        uint row = gid.x;
        uint n = gid.y;
        if (row >= a.rows || n >= a.count) { return; }
        uint quads = a.columns >> 2;
        device const W *w = weights + row * quads;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset + n * a.inStride);
        float sum = dot_row(w, x, quads);
        if ((a.flags & 1u) != 0u) { sum *= scales[row]; }
        linear_store(output, bias, aux, a, row, n, sum);
    }

    // 線形層（トークン 4 個ぶんを 1 スレッドで。重みを読むのが 1 回で済む）。gid.x = 出力の行、gid.y = トークン ÷ 4
    template <typename W>
    inline void linear_four(device const W *weights, device const float *scales, device const float *bias,
                            device const float *input, device float *output, device const float *aux,
                            constant Args &a, uint2 gid) {
        uint row = gid.x;
        uint first = gid.y * 4u;
        if (row >= a.rows || first >= a.count) { return; }
        uint last = a.count - 1u;
        uint n1 = min(first + 1u, last);
        uint n2 = min(first + 2u, last);
        uint n3 = min(first + 3u, last);
        uint quads = a.columns >> 2;
        device const W *w = weights + row * quads;
        device const float *base = input + a.inOffset;
        device const packed_float4 *x0 = (device const packed_float4 *)(base + first * a.inStride);
        device const packed_float4 *x1 = (device const packed_float4 *)(base + n1 * a.inStride);
        device const packed_float4 *x2 = (device const packed_float4 *)(base + n2 * a.inStride);
        device const packed_float4 *x3 = (device const packed_float4 *)(base + n3 * a.inStride);
        float4 s0 = float4(0.0f);
        float4 s1 = float4(0.0f);
        float4 s2 = float4(0.0f);
        float4 s3 = float4(0.0f);
        for (uint i = 0; i < quads; i++) {
            float4 v = load4(w, i);
            s0 += v * float4(x0[i]);
            s1 += v * float4(x1[i]);
            s2 += v * float4(x2[i]);
            s3 += v * float4(x3[i]);
        }
        float4 sums = float4(sum4(s0), sum4(s1), sum4(s2), sum4(s3));
        if ((a.flags & 1u) != 0u) { sums *= scales[row]; }
        linear_store(output, bias, aux, a, row, first, sums.x);
        if (first + 1u <= last) { linear_store(output, bias, aux, a, row, first + 1u, sums.y); }
        if (first + 2u <= last) { linear_store(output, bias, aux, a, row, first + 2u, sums.z); }
        if (first + 3u <= last) { linear_store(output, bias, aux, a, row, first + 3u, sums.w); }
    }

    // 線形層の入り口（どれも同じ並び: 0 重み、1 倍率、2 バイアス、3 入力、4 出力、5 使わない、6 aux、7 Args）
    kernel void linear_q8(device const char *weights [[buffer(0)]],
                          device const float *scales [[buffer(1)]],
                          device const float *bias [[buffer(2)]],
                          device const float *input [[buffer(3)]],
                          device float *output [[buffer(4)]],
                          device const float *aux [[buffer(6)]],
                          constant Args &a [[buffer(7)]],
                          uint2 gid [[thread_position_in_grid]]) {
        linear_one((device const packed_char4 *)(weights + a.weightSkip), scales, bias, input, output, aux, a, gid);
    }

    kernel void linear_q8_batch(device const char *weights [[buffer(0)]],
                                device const float *scales [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device const float *input [[buffer(3)]],
                                device float *output [[buffer(4)]],
                                device const float *aux [[buffer(6)]],
                                constant Args &a [[buffer(7)]],
                                uint2 gid [[thread_position_in_grid]]) {
        linear_four((device const packed_char4 *)(weights + a.weightSkip), scales, bias, input, output, aux, a, gid);
    }

    kernel void linear_f32(device const char *weights [[buffer(0)]],
                           device const float *scales [[buffer(1)]],
                           device const float *bias [[buffer(2)]],
                           device const float *input [[buffer(3)]],
                           device float *output [[buffer(4)]],
                           device const float *aux [[buffer(6)]],
                           constant Args &a [[buffer(7)]],
                           uint2 gid [[thread_position_in_grid]]) {
        linear_one((device const packed_float4 *)(weights + a.weightSkip), scales, bias, input, output, aux, a, gid);
    }

    kernel void linear_f32_batch(device const char *weights [[buffer(0)]],
                                 device const float *scales [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device const float *input [[buffer(3)]],
                                 device float *output [[buffer(4)]],
                                 device const float *aux [[buffer(6)]],
                                 constant Args &a [[buffer(7)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        linear_four((device const packed_float4 *)(weights + a.weightSkip), scales, bias, input, output, aux, a, gid);
    }

    // 線形層（1 トークンだけのとき）: 1 つの SIMD グループ（32 スレッド）で 1 行を計算する。
    // 隣どうしのスレッドが隣どうしの重みを読むので、メモリを続けて読める。threadgroup = 行
    template <typename W>
    inline void linear_vector(device const W *weights, device const float *scales, device const float *bias,
                              device const float *input, device float *output, device const float *aux,
                              constant Args &a, uint row, uint lane) {
        if (row >= a.rows) { return; }
        uint quads = a.columns >> 2;
        device const W *w = weights + row * quads;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        float4 acc = float4(0.0f);
        for (uint i = lane; i < quads; i += 32u) {
            acc += load4(w, i) * float4(x[i]);
        }
        float sum = simd_sum(sum4(acc));
        if (lane == 0u) {
            if ((a.flags & 1u) != 0u) { sum *= scales[row]; }
            linear_store(output, bias, aux, a, row, 0u, sum);
        }
    }

    kernel void linear_q8_vector(device const char *weights [[buffer(0)]],
                                 device const float *scales [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device const float *input [[buffer(3)]],
                                 device float *output [[buffer(4)]],
                                 device const float *aux [[buffer(6)]],
                                 constant Args &a [[buffer(7)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
        linear_vector((device const packed_char4 *)(weights + a.weightSkip), scales, bias, input, output, aux, a, row, lane);
    }

    kernel void linear_f32_vector(device const char *weights [[buffer(0)]],
                                  device const float *scales [[buffer(1)]],
                                  device const float *bias [[buffer(2)]],
                                  device const float *input [[buffer(3)]],
                                  device float *output [[buffer(4)]],
                                  device const float *aux [[buffer(6)]],
                                  constant Args &a [[buffer(7)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
        linear_vector((device const packed_float4 *)(weights + a.weightSkip), scales, bias, input, output, aux, a, row, lane);
    }

    // 8 ビットの重み（行ごとの倍率）を 16 個ずつ（16 バイト）読む。列は 16 の倍数、行の先頭は 16 バイトの境目にあること
    inline float dot16(uint4 w, float4 x0, float4 x1, float4 x2, float4 x3) {
        return dot(float4(as_type<char4>(w.x)), x0) + dot(float4(as_type<char4>(w.y)), x1)
             + dot(float4(as_type<char4>(w.z)), x2) + dot(float4(as_type<char4>(w.w)), x3);
    }

    // 1 トークン: SIMD グループ 1 つで Q8_ROWS 行。スレッドは 16 個ずつを受け持ち、入力は 1 回読んで何行にも使う
    kernel void linear_q8_rows(device const char *weights [[buffer(0)]],
                               device const float *scales [[buffer(1)]],
                               device const float *bias [[buffer(2)]],
                               device const float *input [[buffer(3)]],
                               device float *output [[buffer(4)]],
                               device const float *aux [[buffer(6)]],
                               constant Args &a [[buffer(7)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group * SIMD_GROUPS + simd) * Q8_ROWS;
        if (first >= a.rows) { return; }
        uint rows = min(Q8_ROWS, a.rows - first);
        uint chunks = a.columns >> 4u;
        device const uint4 *w = (device const uint4 *)(weights + a.weightSkip) + first * chunks;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        float acc[Q8_ROWS];
        for (uint r = 0u; r < Q8_ROWS; r++) { acc[r] = 0.0f; }
        for (uint c = lane; c < chunks; c += 32u) {
            float4 x0 = float4(x[c * 4u]);
            float4 x1 = float4(x[c * 4u + 1u]);
            float4 x2 = float4(x[c * 4u + 2u]);
            float4 x3 = float4(x[c * 4u + 3u]);
            for (uint r = 0u; r < Q8_ROWS; r++) {
                if (r < rows) { acc[r] += dot16(w[r * chunks + c], x0, x1, x2, x3); }
            }
        }
        for (uint r = 0u; r < Q8_ROWS; r++) {
            if (r < rows) {
                float sum = simd_sum(acc[r]);
                if (lane == 0u) {
                    if ((a.flags & 1u) != 0u) { sum *= scales[first + r]; }
                    linear_store(output, bias, aux, a, first + r, 0u, sum);
                }
            }
        }
    }

    // まとめて（トークン BATCH_TOKENS 個まで。先読みの確かめなど）: SIMD グループ 1 つで BATCH_ROWS 行。
    // 重みは 1 回読んで、全部のトークンに使う。threadgroup の y = トークン ÷ BATCH_TOKENS
    kernel void linear_q8_tokens(device const char *weights [[buffer(0)]],
                                 device const float *scales [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device const float *input [[buffer(3)]],
                                 device float *output [[buffer(4)]],
                                 device const float *aux [[buffer(6)]],
                                 constant Args &a [[buffer(7)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * BATCH_ROWS;
        uint t0 = group.y * BATCH_TOKENS;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(BATCH_ROWS, a.rows - first);
        uint tokens = min(BATCH_TOKENS, a.count - t0);
        uint chunks = a.columns >> 4u;
        device const uint4 *w = (device const uint4 *)(weights + a.weightSkip) + first * chunks;
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[BATCH_ROWS][BATCH_TOKENS];
        for (uint r = 0u; r < BATCH_ROWS; r++) {
            for (uint t = 0u; t < BATCH_TOKENS; t++) { acc[r][t] = 0.0f; }
        }
        for (uint c = lane; c < chunks; c += 32u) {
            float4 w0[BATCH_ROWS];
            float4 w1[BATCH_ROWS];
            float4 w2[BATCH_ROWS];
            float4 w3[BATCH_ROWS];
            for (uint r = 0u; r < BATCH_ROWS; r++) {
                uint4 q = (r < rows) ? w[r * chunks + c] : uint4(0u);
                w0[r] = float4(as_type<char4>(q.x));
                w1[r] = float4(as_type<char4>(q.y));
                w2[r] = float4(as_type<char4>(q.z));
                w3[r] = float4(as_type<char4>(q.w));
            }
            for (uint t = 0u; t < BATCH_TOKENS; t++) {
                if (t < tokens) {
                    device const packed_float4 *x = (device const packed_float4 *)(base + t * a.inStride) + c * 4u;
                    float4 x0 = float4(x[0]);
                    float4 x1 = float4(x[1]);
                    float4 x2 = float4(x[2]);
                    float4 x3 = float4(x[3]);
                    for (uint r = 0u; r < BATCH_ROWS; r++) {
                        acc[r][t] += dot(w0[r], x0) + dot(w1[r], x1) + dot(w2[r], x2) + dot(w3[r], x3);
                    }
                }
            }
        }
        for (uint r = 0u; r < BATCH_ROWS; r++) {
            if (r < rows) {
                float scale = ((a.flags & 1u) != 0u) ? scales[first + r] : 1.0f;
                for (uint t = 0u; t < BATCH_TOKENS; t++) {
                    if (t < tokens) {
                        float sum = simd_sum(acc[r][t]);
                        if (lane == 0u) { linear_store(output, bias, aux, a, first + r, t0 + t, sum * scale); }
                    }
                }
            }
        }
    }

    // 8 ビット・少しまとめて（N トークン。先読みの確かめなど）: SIMD グループ 1 つで FEW_ROWS 行。16 個ずつの重みを 1 回だけ float にして
    // N トークンに使い、入力はトークンごとに 1 回読んで全部の行に使う。N はコンパイル時に決める（linear_group_few と同じ理由）。
    // threadgroup の y = トークン ÷ N
    template <uint N>
    kernel void linear_q8_few(device const char *weights [[buffer(0)]],
                              device const float *scales [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device const float *input [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              device const float *aux [[buffer(6)]],
                              constant Args &a [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        uint first = (group.x * SIMD_GROUPS + simd) * FEW_ROWS;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(FEW_ROWS, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint chunks = a.columns >> 4u;
        device const uint4 *w = (device const uint4 *)(weights + a.weightSkip);
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[FEW_ROWS][N];
        for (uint r = 0u; r < FEW_ROWS; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint c = lane; c < chunks; c += 32u) {
            float4 w0[FEW_ROWS];
            float4 w1[FEW_ROWS];
            float4 w2[FEW_ROWS];
            float4 w3[FEW_ROWS];
            for (uint r = 0u; r < FEW_ROWS; r++) {
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
                for (uint r = 0u; r < FEW_ROWS; r++) {
                    acc[r][t] += dot(w0[r], x0) + dot(w1[r], x1) + dot(w2[r], x2) + dot(w3[r], x3);
                }
            }
        }
        for (uint r = 0u; r < FEW_ROWS; r++) {
            float scale = ((a.flags & 1u) != 0u) ? scales[first + min(r, rows - 1u)] : 1.0f;
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum * scale); }
            }
        }
    }

    typedef decltype(linear_q8_few<2>) linear_q8_few_t;
    template [[host_name("linear_q8_few2")]] kernel linear_q8_few_t linear_q8_few<2>;
    template [[host_name("linear_q8_few3")]] kernel linear_q8_few_t linear_q8_few<3>;
    template [[host_name("linear_q8_few4")]] kernel linear_q8_few_t linear_q8_few<4>;

    // いちばん大きい値の位置（同じ値なら前のもの）を result[outOffset] に書く。threadgroup 1 つで values[inOffset …] の count 個を見る
    kernel void argmax(device const float *values [[buffer(0)]],
                       device uint *result [[buffer(1)]],
                       constant Args &a [[buffer(2)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]],
                       uint simds [[simdgroups_per_threadgroup]]) {
        threadgroup float bestValues[32];
        threadgroup uint bestIndices[32];
        device const float *v = values + a.inOffset;
        float best = -3.0e38f;
        uint index = 0xFFFFFFFFu;
        for (uint i = tid; i < a.count; i += threads) {
            float x = v[i];
            if (x > best) {
                best = x;
                index = i;
            }
        }
        float top = simd_max(best);
        uint first = simd_min(best == top ? index : 0xFFFFFFFFu);
        if (lane == 0u) {
            bestValues[simd] = top;
            bestIndices[simd] = first;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            float value = lane < simds ? bestValues[lane] : -3.0e38f;
            uint position = lane < simds ? bestIndices[lane] : 0xFFFFFFFFu;
            float overall = simd_max(value);
            uint chosen = simd_min(value == overall ? position : 0xFFFFFFFFu);
            if (lane == 0u) { result[a.outOffset] = chosen == 0xFFFFFFFFu ? 0u : chosen; }
        }
    }

    // 埋め込みの表の 1 行 × scale を output[outOffset …] に書く。行の番号は GPU の tokens[start] から読む（先読みで使う）。
    // size: 32 = Float32、8 = 8 ビット（flags の 1 があれば行ごとの倍率 scales）、4・5 = 組（scales は倍率と下限、high は 5 ビット目）
    kernel void embedding_row(device const char *weights [[buffer(0)]],
                              device const float *scales [[buffer(1)]],
                              device const uint *tokens [[buffer(2)]],
                              device float *output [[buffer(3)]],
                              device const uint *high [[buffer(4)]],
                              constant Args &a [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= a.columns) { return; }
        uint row = tokens[a.start];
        device const char *base = weights + a.weightSkip;
        float value;
        if (a.size == 4u || a.size == 5u) {
            uint groups = a.columns >> 5u;
            uint g = i >> 5u;
            uint j = i & 31u;
            device const uint *words = (device const uint *)base + (row * groups + g) * 4u;
            uint q = (words[j >> 3u] >> ((j & 7u) * 4u)) & 15u;
            if (a.size == 5u) { q += ((high[row * groups + g] >> j) & 1u) * 16u; }
            float2 p = float2(((device const half2 *)scales)[row * groups + g]);
            value = (float(q) * p.x + p.y) * a.scale;
        } else if (a.size == 8u) {
            float rowScale = ((a.flags & 1u) != 0u) ? scales[row] : 1.0f;
            value = float(base[row * a.columns + i]) * (a.scale * rowScale);
        } else {
            value = ((device const float *)base)[row * a.columns + i] * a.scale;
        }
        output[a.outOffset + i] = value;
    }

    // 4・5 ビットの重み（GGUFGroupQuant）。32 個ずつの組: 下位 4 ビット 16 バイト（j 番目は j ÷ 8 番目の uint の 4 × (j % 8) ビット目）、
    // 5 ビット目 4 バイト（j 番目は j ビット目）、倍率と下限（half2）。値 = q × 倍率 + 下限
    inline float4 nibbles4(uint word, uint shift) {
        return float4(float((word >> shift) & 15u), float((word >> (shift + 4u)) & 15u),
                      float((word >> (shift + 8u)) & 15u), float((word >> (shift + 12u)) & 15u));
    }

    inline float4 high4(uint bits, uint e) {
        return float4(float((bits >> e) & 1u), float((bits >> (e + 1u)) & 1u),
                      float((bits >> (e + 2u)) & 1u), float((bits >> (e + 3u)) & 1u)) * 16.0f;
    }

    // 組の半分（16 個）の Σ q·x。w = 下位 4 ビット 8 バイト、bits = その 16 個の 5 ビット目（下から）
    inline float half_dot(uint2 w, uint bits, bool five, float4 x0, float4 x1, float4 x2, float4 x3) {
        float4 a0 = nibbles4(w.x, 0u);
        float4 a1 = nibbles4(w.x, 16u);
        float4 a2 = nibbles4(w.y, 0u);
        float4 a3 = nibbles4(w.y, 16u);
        if (five) {
            a0 += high4(bits, 0u);
            a1 += high4(bits, 4u);
            a2 += high4(bits, 8u);
            a3 += high4(bits, 12u);
        }
        return dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
    }

    // 4 ビットの組の半分の Σ q·x を、シフトせずにマスクだけで q を取り出して計算する（1 トークンのときに速い）。
    // マスクした値は q × (1, 16, 256, 4096) なので、入力は先にその分だけ割っておく: y = x × (1, 1/16, 1/256, 1/4096)。
    // 積は 2 の累乗倍しか違わないので、half_dot と同じ値になる
    #define NIBBLE_SCALES float4(1.0f, 0.0625f, 0.00390625f, 0.000244140625f)

    inline float masked_dot(uint2 w, float4 y0, float4 y1, float4 y2, float4 y3) {
        const uint4 m = uint4(0xFu, 0xF0u, 0xF00u, 0xF000u);
        float4 a0 = float4(uint4(w.x) & m);
        float4 a1 = float4(uint4(w.x >> 16u) & m);
        float4 a2 = float4(uint4(w.y) & m);
        float4 a3 = float4(uint4(w.y >> 16u) & m);
        return dot(a0, y0) + dot(a1, y1) + dot(a2, y2) + dot(a3, y3);
    }

    // 1 トークン: SIMD グループ 1 つで VECTOR_ROWS 行。スレッドは組の半分ずつを受け持ち、入力は 1 回読んで何行にも使う。
    // threadgroup = SIMD_GROUPS 個の SIMD グループ。4 ビットは masked_dot、5 ビットは half_dot
    inline void linear_group_rows(device const uchar *nibbles, device const half2 *params, device const float *bias,
                                  device const float *input, device float *output, device const uint *high,
                                  device const float *aux, constant Args &a, uint group, uint simd, uint lane, bool five) {
        uint first = (group * SIMD_GROUPS + simd) * VECTOR_ROWS;
        if (first >= a.rows) { return; }
        uint rows = min(VECTOR_ROWS, a.rows - first);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        device const uint2 *words = (device const uint2 *)nibbles;
        float acc[VECTOR_ROWS];
        for (uint r = 0u; r < VECTOR_ROWS; r++) { acc[r] = 0.0f; }
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
            for (uint r = 0u; r < VECTOR_ROWS; r++) {
                if (r < rows) {
                    uint b = block + r * groups;
                    float2 p = float2(params[b]);
                    uint2 w = words[b * 2u + part];
                    float qx = five ? half_dot(w, high[b] >> (part * 16u), true, x0, x1, x2, x3) : masked_dot(w, y0, y1, y2, y3);
                    acc[r] += p.x * qx + p.y * xs;
                }
            }
        }
        for (uint r = 0u; r < VECTOR_ROWS; r++) {
            if (r < rows) {
                float sum = simd_sum(acc[r]);
                if (lane == 0u) { linear_store(output, bias, aux, a, first + r, 0u, sum); }
            }
        }
    }

    // まとめて（トークン BATCH_TOKENS 個まで）: SIMD グループ 1 つで BATCH_ROWS 行。重みは q に直したら全部のトークンに使う。
    // gid.y（threadgroup の y）= トークン ÷ BATCH_TOKENS
    inline void linear_group_batch(device const uchar *nibbles, device const half2 *params, device const float *bias,
                                   device const float *input, device float *output, device const uint *high,
                                   device const float *aux, constant Args &a, uint group, uint chunk, uint simd, uint lane,
                                   bool five) {
        uint first = (group * SIMD_GROUPS + simd) * BATCH_ROWS;
        uint t0 = chunk * BATCH_TOKENS;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(BATCH_ROWS, a.rows - first);
        uint tokens = min(BATCH_TOKENS, a.count - t0);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const uint2 *words = (device const uint2 *)nibbles;
        float acc[BATCH_ROWS][BATCH_TOKENS];
        for (uint r = 0u; r < BATCH_ROWS; r++) {
            for (uint t = 0u; t < BATCH_TOKENS; t++) { acc[r][t] = 0.0f; }
        }
        for (uint h = lane; h < halves; h += 32u) {
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            for (uint r = 0u; r < BATCH_ROWS; r++) {
                if (r < rows) {
                    uint b = block + r * groups;
                    uint2 w = words[b * 2u + part];
                    float4 a0 = nibbles4(w.x, 0u);
                    float4 a1 = nibbles4(w.x, 16u);
                    float4 a2 = nibbles4(w.y, 0u);
                    float4 a3 = nibbles4(w.y, 16u);
                    if (five) {
                        uint bits = high[b] >> (part * 16u);
                        a0 += high4(bits, 0u);
                        a1 += high4(bits, 4u);
                        a2 += high4(bits, 8u);
                        a3 += high4(bits, 12u);
                    }
                    float2 p = float2(params[b]);
                    for (uint t = 0u; t < BATCH_TOKENS; t++) {
                        if (t < tokens) {
                            device const packed_float4 *x =
                                (device const packed_float4 *)(input + a.inOffset + (t0 + t) * a.inStride) + h * 4u;
                            float4 x0 = float4(x[0]);
                            float4 x1 = float4(x[1]);
                            float4 x2 = float4(x[2]);
                            float4 x3 = float4(x[3]);
                            float qx = dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
                            acc[r][t] += p.x * qx + p.y * sum4(x0 + x1 + x2 + x3);
                        }
                    }
                }
            }
        }
        for (uint r = 0u; r < BATCH_ROWS; r++) {
            if (r < rows) {
                for (uint t = 0u; t < BATCH_TOKENS; t++) {
                    if (t < tokens) {
                        float sum = simd_sum(acc[r][t]);
                        if (lane == 0u) { linear_store(output, bias, aux, a, first + r, t0 + t, sum); }
                    }
                }
            }
        }
    }

    // 4・5 ビットの入り口（0 下位 4 ビット、1 倍率と下限、2 バイアス、3 入力、4 出力、5 5 ビット目、6 aux、7 Args）
    kernel void linear_q4_rows(device const uchar *nibbles [[buffer(0)]],
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
        linear_group_rows(nibbles, params, bias, input, output, high, aux, a, group, simd, lane, false);
    }

    kernel void linear_q5_rows(device const uchar *nibbles [[buffer(0)]],
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
        linear_group_rows(nibbles, params, bias, input, output, high, aux, a, group, simd, lane, true);
    }

    // 4 個の q（0 … 31 の整数）を half に。half でちょうど表せるので、float のときの半分のレジスタで持てる
    inline half4 nibbles4h(uint word, uint shift) {
        return half4(half((word >> shift) & 15u), half((word >> (shift + 4u)) & 15u),
                     half((word >> (shift + 8u)) & 15u), half((word >> (shift + 12u)) & 15u));
    }

    inline half4 high4h(uint bits, uint e) {
        return half4(half((bits >> e) & 1u), half((bits >> (e + 1u)) & 1u),
                     half((bits >> (e + 2u)) & 1u), half((bits >> (e + 3u)) & 1u)) * half(16.0f);
    }

    // 少しまとめて（N トークン。先読みの確かめなど）: 1 トークンのときと同じく SIMD グループ 1 つで FEW_ROWS 行、スレッドは組の半分ずつ。
    // 組の半分の重みは 1 回だけ q に直して（half で持つ）N トークンに使い、入力はトークンごとに 1 回読んで全部の行に使う。
    // N はコンパイル時に決める（足し込む値をレジスタに置くため。実行時の数で回すと、配列がメモリに置かれて遅くなる）。
    // threadgroup の y = トークン ÷ N。残りが N より少ないときは、最後のトークンを読み直して計算し、書かない
    template <uint N, bool FIVE>
    kernel void linear_group_few(device const uchar *nibbles [[buffer(0)]],
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
        uint first = (group.x * SIMD_GROUPS + simd) * FEW_ROWS;
        uint t0 = group.y * N;
        if (first >= a.rows || t0 >= a.count) { return; }
        uint rows = min(FEW_ROWS, a.rows - first);
        uint tokens = min(N, a.count - t0);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const uint2 *words = (device const uint2 *)nibbles;
        device const float *base = input + a.inOffset + t0 * a.inStride;
        float acc[FEW_ROWS][N];
        for (uint r = 0u; r < FEW_ROWS; r++) {
            for (uint t = 0u; t < N; t++) { acc[r][t] = 0.0f; }
        }
        for (uint h = lane; h < halves; h += 32u) {
            uint part = h & 1u;
            uint block = first * groups + (h >> 1u);
            half4 w0[FEW_ROWS];
            half4 w1[FEW_ROWS];
            half4 w2[FEW_ROWS];
            half4 w3[FEW_ROWS];
            float2 p[FEW_ROWS];
            for (uint r = 0u; r < FEW_ROWS; r++) {
                uint b = block + min(r, rows - 1u) * groups;
                uint2 q = words[b * 2u + part];
                w0[r] = nibbles4h(q.x, 0u);
                w1[r] = nibbles4h(q.x, 16u);
                w2[r] = nibbles4h(q.y, 0u);
                w3[r] = nibbles4h(q.y, 16u);
                if (FIVE) {
                    uint bits = high[b] >> (part * 16u);
                    w0[r] += high4h(bits, 0u);
                    w1[r] += high4h(bits, 4u);
                    w2[r] += high4h(bits, 8u);
                    w3[r] += high4h(bits, 12u);
                }
                p[r] = float2(params[b]);
            }
            for (uint t = 0u; t < N; t++) {
                device const packed_float4 *x = (device const packed_float4 *)(base + min(t, tokens - 1u) * a.inStride) + h * 4u;
                float4 x0 = float4(x[0]);
                float4 x1 = float4(x[1]);
                float4 x2 = float4(x[2]);
                float4 x3 = float4(x[3]);
                float xs = sum4(x0 + x1 + x2 + x3);
                for (uint r = 0u; r < FEW_ROWS; r++) {
                    float qx = dot(float4(w0[r]), x0) + dot(float4(w1[r]), x1) + dot(float4(w2[r]), x2) + dot(float4(w3[r]), x3);
                    acc[r][t] += p[r].x * qx + p[r].y * xs;
                }
            }
        }
        for (uint r = 0u; r < FEW_ROWS; r++) {
            for (uint t = 0u; t < N; t++) {
                float sum = simd_sum(acc[r][t]);
                if (lane == 0u && r < rows && t < tokens) { linear_store(output, bias, aux, a, first + r, t0 + t, sum); }
            }
        }
    }

    // 入り口（並びは linear_q4_rows と同じ）。名前の数字が N
    typedef decltype(linear_group_few<2, false>) linear_group_few_t;
    template [[host_name("linear_q4_few2")]] kernel linear_group_few_t linear_group_few<2, false>;
    template [[host_name("linear_q4_few3")]] kernel linear_group_few_t linear_group_few<3, false>;
    template [[host_name("linear_q4_few4")]] kernel linear_group_few_t linear_group_few<4, false>;
    template [[host_name("linear_q5_few2")]] kernel linear_group_few_t linear_group_few<2, true>;
    template [[host_name("linear_q5_few3")]] kernel linear_group_few_t linear_group_few<3, true>;
    template [[host_name("linear_q5_few4")]] kernel linear_group_few_t linear_group_few<4, true>;

    kernel void linear_q4_batch(device const uchar *nibbles [[buffer(0)]],
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
        linear_group_batch(nibbles, params, bias, input, output, high, aux, a, group.x, group.y, simd, lane, false);
    }

    kernel void linear_q5_batch(device const uchar *nibbles [[buffer(0)]],
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
        linear_group_batch(nibbles, params, bias, input, output, high, aux, a, group.x, group.y, simd, lane, true);
    }

    // 古い GPU（SIMD グループが 32 スレッドでないもの）用: 1 スレッドで 1 行・トークン 1 つ
    inline void linear_group_thread(device const uchar *nibbles, device const half2 *params, device const float *bias,
                                    device const float *input, device float *output, device const uint *high,
                                    device const float *aux, constant Args &a, uint2 gid, bool five) {
        uint row = gid.x;
        uint n = gid.y;
        if (row >= a.rows || n >= a.count) { return; }
        uint groups = a.columns >> 5u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset + n * a.inStride);
        device const uint2 *words = (device const uint2 *)nibbles;
        float sum = 0.0f;
        for (uint h = 0u; h < groups * 2u; h++) {
            float4 x0 = float4(x[h * 4u]);
            float4 x1 = float4(x[h * 4u + 1u]);
            float4 x2 = float4(x[h * 4u + 2u]);
            float4 x3 = float4(x[h * 4u + 3u]);
            uint b = row * groups + (h >> 1u);
            uint part = h & 1u;
            uint bits = five ? (high[b] >> (part * 16u)) : 0u;
            float2 p = float2(params[b]);
            sum += p.x * half_dot(words[b * 2u + part], bits, five, x0, x1, x2, x3) + p.y * sum4(x0 + x1 + x2 + x3);
        }
        linear_store(output, bias, aux, a, row, n, sum);
    }

    kernel void linear_q4_vector(device const uchar *nibbles [[buffer(0)]],
                                 device const half2 *params [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device const float *input [[buffer(3)]],
                                 device float *output [[buffer(4)]],
                                 device const uint *high [[buffer(5)]],
                                 device const float *aux [[buffer(6)]],
                                 constant Args &a [[buffer(7)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        linear_group_thread(nibbles, params, bias, input, output, high, aux, a, gid, false);
    }

    kernel void linear_q5_vector(device const uchar *nibbles [[buffer(0)]],
                                 device const half2 *params [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device const float *input [[buffer(3)]],
                                 device float *output [[buffer(4)]],
                                 device const uint *high [[buffer(5)]],
                                 device const float *aux [[buffer(6)]],
                                 constant Args &a [[buffer(7)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        linear_group_thread(nibbles, params, bias, input, output, high, aux, a, gid, true);
    }

    // FFN の gate と up を一度に（1 トークン。入力を 1 回読んで 2 つの行列に使う）: 出力 = act(gate · x) × (up · x)。
    // SIMD グループ 1 つで GATE_UP_ROWS 行。4 ビットは masked_dot、5 ビットは half_dot
    inline void gate_up_rows(device const uchar *gateNibbles, device const half2 *gateParams,
                             device const uchar *upNibbles, device const half2 *upParams,
                             device const float *input, device float *output,
                             device const uint *gateHigh, device const uint *upHigh,
                             constant Args &a, uint group, uint simd, uint lane, bool five) {
        uint first = (group * SIMD_GROUPS + simd) * GATE_UP_ROWS;
        if (first >= a.rows) { return; }
        uint rows = min(GATE_UP_ROWS, a.rows - first);
        uint groups = a.columns >> 5u;
        uint halves = groups * 2u;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset);
        device const uint2 *gateWords = (device const uint2 *)gateNibbles;
        device const uint2 *upWords = (device const uint2 *)upNibbles;
        float g[GATE_UP_ROWS];
        float u[GATE_UP_ROWS];
        for (uint r = 0u; r < GATE_UP_ROWS; r++) {
            g[r] = 0.0f;
            u[r] = 0.0f;
        }
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
            for (uint r = 0u; r < GATE_UP_ROWS; r++) {
                if (r < rows) {
                    uint b = block + r * groups;
                    float2 pg = float2(gateParams[b]);
                    float2 pu = float2(upParams[b]);
                    uint2 gw = gateWords[b * 2u + part];
                    uint2 uw = upWords[b * 2u + part];
                    float gx = five ? half_dot(gw, gateHigh[b] >> (part * 16u), true, x0, x1, x2, x3) : masked_dot(gw, y0, y1, y2, y3);
                    float ux = five ? half_dot(uw, upHigh[b] >> (part * 16u), true, x0, x1, x2, x3) : masked_dot(uw, y0, y1, y2, y3);
                    g[r] += pg.x * gx + pg.y * xs;
                    u[r] += pu.x * ux + pu.y * xs;
                }
            }
        }
        for (uint r = 0u; r < GATE_UP_ROWS; r++) {
            if (r < rows) {
                float gs = simd_sum(g[r]);
                float us = simd_sum(u[r]);
                if (lane == 0u) { output[a.outOffset + first + r] = activation(gs, a.flags) * us; }
            }
        }
    }

    // 0 gate の下位 4 ビット、1 gate の倍率と下限、2 up の下位 4 ビット、3 up の倍率と下限、4 入力、5 出力、
    // 6 gate の 5 ビット目、7 up の 5 ビット目、8 Args
    kernel void gate_up_q4(device const uchar *gateNibbles [[buffer(0)]],
                           device const half2 *gateParams [[buffer(1)]],
                           device const uchar *upNibbles [[buffer(2)]],
                           device const half2 *upParams [[buffer(3)]],
                           device const float *input [[buffer(4)]],
                           device float *output [[buffer(5)]],
                           device const uint *gateHigh [[buffer(6)]],
                           device const uint *upHigh [[buffer(7)]],
                           constant Args &a [[buffer(8)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
        gate_up_rows(gateNibbles, gateParams, upNibbles, upParams, input, output, gateHigh, upHigh, a, group, simd, lane, false);
    }

    kernel void gate_up_q5(device const uchar *gateNibbles [[buffer(0)]],
                           device const half2 *gateParams [[buffer(1)]],
                           device const uchar *upNibbles [[buffer(2)]],
                           device const half2 *upParams [[buffer(3)]],
                           device const float *input [[buffer(4)]],
                           device float *output [[buffer(5)]],
                           device const uint *gateHigh [[buffer(6)]],
                           device const uint *upHigh [[buffer(7)]],
                           constant Args &a [[buffer(8)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
        gate_up_rows(gateNibbles, gateParams, upNibbles, upParams, input, output, gateHigh, upHigh, a, group, simd, lane, true);
    }

    // 正規化（RMSNorm か LayerNorm）。threadgroup = 行（rows 行 × columns）で、1 行をスレッド全部で計算する。
    // 入力と出力は同じ場所でもよい。heads > 1 なら、行ごとに (行 % heads) 番目の重みを使う（ヘッドごとに違う重み）。
    // flags の 1024: 出力に足す（出力 = (出力 + 値 × extra) × scale。残差の足し算をここでする）
    kernel void rms_norm(device const float *input [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device const float *bias [[buffer(2)]],
                         device float *output [[buffer(3)]],
                         constant Args &a [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint simds [[simdgroups_per_threadgroup]]) {
        threadgroup float shared[32];
        device const float *x = input + a.inOffset + row * a.columns;
        device float *y = output + a.outOffset + row * a.columns;
        uint w = (a.heads > 1u) ? (row % a.heads) * a.columns : 0u;
        float mean = 0.0f;
        if ((a.flags & 64u) != 0u) {
            float s = 0.0f;
            for (uint i = tid; i < a.columns; i += threads) { s += x[i]; }
            mean = group_sum(s, shared, lane, simd, simds) / float(a.columns);
        }
        float squares = 0.0f;
        for (uint i = tid; i < a.columns; i += threads) {
            float v = x[i] - mean;
            squares += v * v;
        }
        float factor = 1.0f / sqrt(group_sum(squares, shared, lane, simd, simds) / float(a.columns) + a.epsilon);
        bool hasWeight = (a.flags & 16u) != 0u;
        bool hasBias = (a.flags & 32u) != 0u;
        bool add = (a.flags & 1024u) != 0u;
        for (uint i = tid; i < a.columns; i += threads) {
            float v = (x[i] - mean) * factor;
            if (hasWeight) { v *= weight[w + i]; }
            if (hasBias) { v += bias[w + i]; }
            if (add) { v = (y[i] + v * a.extra) * a.scale; }
            y[i] = v;
        }
    }

    // q・k のヘッドごとの RMSNorm と RoPE を続けて（threadgroup = トークン × ヘッドの 1 行、size 個）。
    // kvHeads > 1 ならヘッドごとに違う重み（kvHeads 個）。flags: 16 重み、4 組が半分ずつ離れている、4096 RoPE が先。
    // table は [トークン][組][cos, sin]（inOffset から）。回すのは先頭の 2 × pairs 個
    kernel void head_norm_rope(device float *x [[buffer(0)]],
                               device const float *weight [[buffer(1)]],
                               device const float *table [[buffer(2)]],
                               constant Args &a [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]]) {
        threadgroup float shared[32];
        device float *v = x + a.outOffset + row * a.size;
        uint token = row / a.heads;
        uint w = (a.kvHeads > 1u) ? (row % a.kvHeads) * a.size : 0u;
        float squares = 0.0f;
        for (uint i = tid; i < a.size; i += threads) {
            float t = v[i];
            squares += t * t;
        }
        float factor = 1.0f / sqrt(group_sum(squares, shared, lane, simd, simds) / float(a.size) + a.epsilon);
        bool hasWeight = (a.flags & 16u) != 0u;
        bool split = (a.flags & 4u) != 0u;
        bool ropeFirst = (a.flags & 4096u) != 0u;
        uint pairs = a.pairs;
        device const float *cs = table + a.inOffset + token * pairs * 2u;
        for (uint p = tid; p < pairs; p += threads) {
            uint i0 = split ? p : p * 2u;
            uint i1 = split ? p + pairs : p * 2u + 1u;
            float p0 = v[i0];
            float p1 = v[i1];
            float c = cs[p * 2u];
            float s = cs[p * 2u + 1u];
            float w0 = hasWeight ? weight[w + i0] : 1.0f;
            float w1 = hasWeight ? weight[w + i1] : 1.0f;
            if (ropeFirst) {
                float r0 = p0 * c - p1 * s;
                float r1 = p0 * s + p1 * c;
                v[i0] = r0 * factor * w0;
                v[i1] = r1 * factor * w1;
            } else {
                float n0 = p0 * factor * w0;
                float n1 = p1 * factor * w1;
                v[i0] = n0 * c - n1 * s;
                v[i1] = n0 * s + n1 * c;
            }
        }
        for (uint i = 2u * pairs + tid; i < a.size; i += threads) {
            v[i] = v[i] * factor * (hasWeight ? weight[w + i] : 1.0f);
        }
    }

    // RoPE（位置に応じた回転）。gid.x = 組、gid.y = トークン × ヘッド。table は [トークン][組][cos, sin]
    kernel void rope(device float *x [[buffer(0)]],
                     device const float *table [[buffer(1)]],
                     constant Args &a [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= a.pairs || gid.y >= a.count * a.heads) { return; }
        uint index = gid.y / a.heads;
        device float *v = x + a.outOffset + gid.y * a.size;
        device const float *cs = table + a.inOffset + (index * a.pairs + gid.x) * 2u;
        bool split = (a.flags & 4u) != 0u;
        uint first = split ? gid.x : gid.x * 2u;
        uint other = split ? gid.x + a.pairs : gid.x * 2u + 1u;
        float p = v[first];
        float q = v[other];
        float c = cs[0];
        float s = cs[1];
        v[first] = p * c - q * s;
        v[other] = p * s + q * c;
    }

    // そのトークンから見える範囲の先頭（直近だけを見る層では window 個）
    inline uint first_visible(uint position, uint window) {
        return (window > 0u && position + 1u > window) ? position + 1u - window : 0u;
    }

    // 注意のスコア = q · k。gid.x = 見る位置、gid.y = トークン × ヘッド
    kernel void attention_scores(device const float *queries [[buffer(0)]],
                                 device const float *keys [[buffer(1)]],
                                 device float *scores [[buffer(2)]],
                                 constant Args &a [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= a.total || gid.y >= a.count * a.heads) { return; }
        uint index = gid.y / a.heads;
        uint head = gid.y % a.heads;
        uint position = a.start + index;
        if (gid.x > position || gid.x < first_visible(position, a.window)) { return; }
        uint group = head / (a.heads / a.kvHeads);
        uint quads = a.size >> 2;
        device const packed_float4 *q = (device const packed_float4 *)(queries + gid.y * a.size);
        device const packed_float4 *k = (device const packed_float4 *)(keys + (gid.x * a.kvHeads + group) * a.size);
        float4 sum = float4(0.0f);
        for (uint i = 0; i < quads; i++) {
            sum += float4(q[i]) * float4(k[i]);
        }
        float s = sum4(sum) * a.scale;
        if (a.softcap > 0.0f) { s = a.softcap * safe_tanh(s / a.softcap); }
        scores[gid.y * a.total + gid.x] = s;
    }

    // スコアを重み（合計 1）にする。threadgroup = トークン × ヘッドの 1 行で、見える位置をスレッド全部で分ける
    kernel void attention_softmax(device float *scores [[buffer(0)]],
                                  constant Args &a [[buffer(1)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint simds [[simdgroups_per_threadgroup]]) {
        threadgroup float shared[32];
        uint position = a.start + row / a.heads;
        uint first = first_visible(position, a.window);
        device float *weights = scores + row * a.total;
        float peak = -3.0e38f;
        for (uint t = first + tid; t <= position; t += threads) { peak = max(peak, weights[t]); }
        peak = group_max(peak, shared, lane, simd, simds);
        float sum = 0.0f;
        for (uint t = first + tid; t <= position; t += threads) {
            float e = exp(weights[t] - peak);
            weights[t] = e;
            sum += e;
        }
        float inverse = 1.0f / group_sum(sum, shared, lane, simd, simds);
        for (uint t = first + tid; t <= position; t += threads) { weights[t] *= inverse; }
    }

    // 重みを付けて v を足し合わせる。threadgroup = トークン × ヘッドの 1 行。
    // スレッド = (ヘッドの大きさ ÷ 4) × 分ける数。見る位置を分けて足し、最後にまとめる
    kernel void attention_combine(device const float *scores [[buffer(0)]],
                                  device const float *values [[buffer(1)]],
                                  device float *output [[buffer(2)]],
                                  constant Args &a [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]]) {
        threadgroup float4 partial[256];
        uint quads = a.size >> 2u;
        uint parts = max(1u, threads / quads);
        uint quad = tid % quads;
        uint part = tid / quads;
        uint head = row % a.heads;
        uint position = a.start + row / a.heads;
        uint first = first_visible(position, a.window);
        uint group = head / (a.heads / a.kvHeads);
        uint stride = a.kvHeads * quads;
        device const float *weights = scores + row * a.total;
        device const packed_float4 *v = (device const packed_float4 *)values + group * quads + quad;
        float4 sum = float4(0.0f);
        if (part < parts) {
            for (uint t = first + part; t <= position; t += parts) {
                sum += weights[t] * float4(v[t * stride]);
            }
        }
        partial[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0u) {
            for (uint p = 1u; p < parts; p++) { sum += partial[p * quads + quad]; }
            device packed_float4 *out = (device packed_float4 *)(output + row * a.size);
            out[quad] = sum;
        }
    }

    // out = scale × x + softcap × y（y は flags に 512 があるときだけ）。gid = 要素。
    // x は inOffset、y は total、out は outOffset から。out は x と同じ場所でもよい
    kernel void scale_add(device const float *x [[buffer(0)]],
                          device const float *y [[buffer(1)]],
                          device float *out [[buffer(2)]],
                          constant Args &a [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
        if (gid >= a.count) { return; }
        float v = a.scale * x[a.inOffset + gid];
        if ((a.flags & 512u) != 0u) { v += a.softcap * y[a.total + gid]; }
        out[a.outOffset + gid] = v;
    }

    // 短い畳み込み（LFM2）の 1: bx = b × x を、位置ごとの記録（history）に書く。gid.x = チャンネル、gid.y = トークン。
    // projected は [トークン][b, c, x]（それぞれ columns 個）
    kernel void convolution_prepare(device const float *projected [[buffer(0)]],
                                    device float *history [[buffer(1)]],
                                    constant Args &a [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= a.columns || gid.y >= a.count) { return; }
        device const float *row = projected + gid.y * 3u * a.columns;
        history[(a.start + gid.y) * a.columns + gid.x] = row[gid.x] * row[2u * a.columns + gid.x];
    }

    // 短い畳み込みの 2: out = c × Σ_j 重み[j] × bx[位置 − (長さ − 1) + j]。gid.x = チャンネル、gid.y = トークン
    kernel void convolution_apply(device const float *projected [[buffer(0)]],
                                  device const float *history [[buffer(1)]],
                                  device const float *kernel_weights [[buffer(2)]],
                                  device float *output [[buffer(3)]],
                                  constant Args &a [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= a.columns || gid.y >= a.count) { return; }
        uint length = a.size;
        int position = int(a.start + gid.y);
        float sum = 0.0f;
        for (uint j = 0; j < length; j++) {
            int source = position - int(length - 1u) + int(j);
            if (source >= 0) {
                sum += kernel_weights[gid.x * length + j] * history[uint(source) * a.columns + gid.x];
            }
        }
        device const float *row = projected + gid.y * 3u * a.columns;
        output[gid.y * a.columns + gid.x] = sum * row[a.columns + gid.x];
    }

    // gate = act(gate) × up（flags に 128 があれば up と掛けない）。gid = 要素。
    // inStride > 0 なら、up は columns 個ごとに inStride ずつ進む（層ごとの入力の 1 層ぶん）
    kernel void gated_activation(device float *gate [[buffer(0)]],
                                 device const float *up [[buffer(1)]],
                                 constant Args &a [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
        if (gid >= a.count) { return; }
        float value = activation(gate[gid], a.flags);
        if ((a.flags & 128u) == 0u) {
            uint index = gid;
            if (a.inStride > 0u) { index = (gid / a.columns) * a.inStride + gid % a.columns; }
            value *= up[a.inOffset + index];
        }
        gate[gid] = value;
    }
    """

    /// まとめて（トークン 2 個以上）の線形層を、simdgroup の 8×8 の行列の掛け算で計算する（Apple7 以降の GPU。code のあとにつなげる）。
    /// 1 つずつのやり方は、まとめたトークンの数だけ入力を何度も読み直すので、先読みの確かめ・プロンプトの読み込みが遅かった。
    /// threadgroup（SIMD グループ 4 つ）が MMA_ROWS 行 × MMA_TOKENS トークンを受け持ち、列を MMA_K 個ずつ進める:
    /// 重みを Float32 に直して threadgroup メモリに置き、入力も置いてから、各 SIMD グループが 8 行 × 8 トークンを足し込む。
    static let matrixCode = """
    #define MMA_ROWS 32u
    #define MMA_TOKENS 8u
    #define MMA_K 64u
    #define MMA_THREADS 128u

    // 4・5 ビットの組の重みの MMA_ROWS 行 × MMA_K 列を、Float32 にして wTile に（[行][列]）。1 スレッドで組の半分（16 個）
    inline void mma_load_groups(threadgroup float *wTile, device const uchar *nibbles, device const half2 *params,
                                device const uint *high, constant Args &a, uint row0, uint k0, uint tid, bool five) {
        uint groups = a.columns >> 5u;
        for (uint i = tid; i < MMA_ROWS * (MMA_K / 16u); i += MMA_THREADS) {
            uint r = i / (MMA_K / 16u);
            uint q = i % (MMA_K / 16u);
            uint col = k0 + q * 16u;
            float4 w0 = float4(0.0f);
            float4 w1 = float4(0.0f);
            float4 w2 = float4(0.0f);
            float4 w3 = float4(0.0f);
            if (row0 + r < a.rows && col < a.columns) {
                uint part = (col >> 4u) & 1u;
                uint b = (row0 + r) * groups + (col >> 5u);
                uint2 w = ((device const uint2 *)nibbles)[b * 2u + part];
                float4 n0 = nibbles4(w.x, 0u);
                float4 n1 = nibbles4(w.x, 16u);
                float4 n2 = nibbles4(w.y, 0u);
                float4 n3 = nibbles4(w.y, 16u);
                if (five) {
                    uint bits = high[b] >> (part * 16u);
                    n0 += high4(bits, 0u);
                    n1 += high4(bits, 4u);
                    n2 += high4(bits, 8u);
                    n3 += high4(bits, 12u);
                }
                float2 p = float2(params[b]);
                w0 = n0 * p.x + p.y;
                w1 = n1 * p.x + p.y;
                w2 = n2 * p.x + p.y;
                w3 = n3 * p.x + p.y;
            }
            threadgroup float4 *dst = (threadgroup float4 *)(wTile + r * MMA_K + q * 16u);
            dst[0] = w0;
            dst[1] = w1;
            dst[2] = w2;
            dst[3] = w3;
        }
    }

    // 8 ビットの重み（行の先頭は 16 バイトの境目）の MMA_ROWS 行 × MMA_K 列を wTile に。行ごとの倍率は最後に掛ける
    inline void mma_load_q8(threadgroup float *wTile, device const char *weights, constant Args &a, uint row0, uint k0, uint tid) {
        for (uint i = tid; i < MMA_ROWS * (MMA_K / 16u); i += MMA_THREADS) {
            uint r = i / (MMA_K / 16u);
            uint q = i % (MMA_K / 16u);
            uint col = k0 + q * 16u;
            float4 w0 = float4(0.0f);
            float4 w1 = float4(0.0f);
            float4 w2 = float4(0.0f);
            float4 w3 = float4(0.0f);
            if (row0 + r < a.rows && col < a.columns) {
                uint4 v = *((device const uint4 *)(weights + a.weightSkip + (row0 + r) * a.columns + col));
                w0 = float4(as_type<char4>(v.x));
                w1 = float4(as_type<char4>(v.y));
                w2 = float4(as_type<char4>(v.z));
                w3 = float4(as_type<char4>(v.w));
            }
            threadgroup float4 *dst = (threadgroup float4 *)(wTile + r * MMA_K + q * 16u);
            dst[0] = w0;
            dst[1] = w1;
            dst[2] = w2;
            dst[3] = w3;
        }
    }

    // 入力の MMA_TOKENS トークン × MMA_K 列を xTile に（[トークン][列]。無いところは 0）
    inline void mma_load_input(threadgroup float *xTile, device const float *input, constant Args &a, uint t0, uint k0, uint tid) {
        for (uint i = tid; i < MMA_TOKENS * (MMA_K / 4u); i += MMA_THREADS) {
            uint t = i / (MMA_K / 4u);
            uint c = (i % (MMA_K / 4u)) * 4u;
            float4 v = float4(0.0f);
            if (t0 + t < a.count && k0 + c < a.columns) {
                v = float4(*((device const packed_float4 *)(input + a.inOffset + (t0 + t) * a.inStride + k0 + c)));
            }
            *((threadgroup float4 *)(xTile + t * MMA_K + c)) = v;
        }
    }

    // この SIMD グループの 8 行 × 8 トークンに、wTile · xTile^T（MMA_K 列ぶん）を足す
    inline void mma_accumulate(threadgroup const float *wTile, threadgroup const float *xTile, uint simd, thread simdgroup_float8x8 &acc) {
        for (uint kk = 0u; kk < MMA_K; kk += 8u) {
            simdgroup_float8x8 w;
            simdgroup_float8x8 x;
            simdgroup_load(w, wTile + simd * 8u * MMA_K + kk, MMA_K);
            simdgroup_load(x, xTile + kk, MMA_K, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, w, x, acc);
        }
    }

    // 8 行 × 8 トークンの結果を書く（行ごとの倍率・バイアス・gelu × aux・出力に足す）
    inline void mma_store(threadgroup float *results, thread simdgroup_float8x8 &acc, device float *output, device const float *bias,
                          device const float *aux, device const float *scales, bool scaled, constant Args &a,
                          uint row0, uint t0, uint simd, uint lane) {
        threadgroup float *mine = results + simd * 64u;
        simdgroup_store(acc, mine, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64u; e += 32u) {
            uint row = row0 + simd * 8u + (e >> 3u);
            uint t = e & 7u;
            if (row < a.rows && t0 + t < a.count) {
                float sum = mine[e];
                if (scaled) { sum *= scales[row]; }
                linear_store(output, bias, aux, a, row, t0 + t, sum);
            }
        }
    }

    // 入り口は linear_q4_batch と同じ並び（threadgroup の x = 行 ÷ MMA_ROWS、y = トークン ÷ MMA_TOKENS）
    kernel void linear_q4_mma(device const uchar *nibbles [[buffer(0)]],
                              device const half2 *params [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device const float *input [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              device const uint *high [[buffer(5)]],
                              device const float *aux [[buffer(6)]],
                              constant Args &a [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float wTile[MMA_ROWS * MMA_K];
        threadgroup float xTile[MMA_TOKENS * MMA_K];
        threadgroup float results[(MMA_THREADS / 32u) * 64u];
        uint row0 = group.x * MMA_ROWS;
        uint t0 = group.y * MMA_TOKENS;
        if (row0 >= a.rows || t0 >= a.count) { return; }
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k0 = 0u; k0 < a.columns; k0 += MMA_K) {
            mma_load_groups(wTile, nibbles, params, high, a, row0, k0, tid, false);
            mma_load_input(xTile, input, a, t0, k0, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_accumulate(wTile, xTile, simd, acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        mma_store(results, acc, output, bias, aux, input, false, a, row0, t0, simd, lane);
    }

    kernel void linear_q5_mma(device const uchar *nibbles [[buffer(0)]],
                              device const half2 *params [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device const float *input [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              device const uint *high [[buffer(5)]],
                              device const float *aux [[buffer(6)]],
                              constant Args &a [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float wTile[MMA_ROWS * MMA_K];
        threadgroup float xTile[MMA_TOKENS * MMA_K];
        threadgroup float results[(MMA_THREADS / 32u) * 64u];
        uint row0 = group.x * MMA_ROWS;
        uint t0 = group.y * MMA_TOKENS;
        if (row0 >= a.rows || t0 >= a.count) { return; }
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k0 = 0u; k0 < a.columns; k0 += MMA_K) {
            mma_load_groups(wTile, nibbles, params, high, a, row0, k0, tid, true);
            mma_load_input(xTile, input, a, t0, k0, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_accumulate(wTile, xTile, simd, acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        mma_store(results, acc, output, bias, aux, input, false, a, row0, t0, simd, lane);
    }

    // 8 ビット（0 重み、1 行ごとの倍率、2 バイアス、3 入力、4 出力、6 aux、7 Args）
    kernel void linear_q8_mma(device const char *weights [[buffer(0)]],
                              device const float *scales [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device const float *input [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              device const float *aux [[buffer(6)]],
                              constant Args &a [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float wTile[MMA_ROWS * MMA_K];
        threadgroup float xTile[MMA_TOKENS * MMA_K];
        threadgroup float results[(MMA_THREADS / 32u) * 64u];
        uint row0 = group.x * MMA_ROWS;
        uint t0 = group.y * MMA_TOKENS;
        if (row0 >= a.rows || t0 >= a.count) { return; }
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k0 = 0u; k0 < a.columns; k0 += MMA_K) {
            mma_load_q8(wTile, weights, a, row0, k0, tid);
            mma_load_input(xTile, input, a, t0, k0, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_accumulate(wTile, xTile, simd, acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        mma_store(results, acc, output, bias, aux, scales, (a.flags & 1u) != 0u, a, row0, t0, simd, lane);
    }
    """
}
