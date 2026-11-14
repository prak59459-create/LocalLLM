import Foundation

/// GPU（Metal）で GGUF の推論を計算するときの、計算の中身（Metal Shading Language）。
/// アプリの中で起動時にコンパイルする（Swift Playgrounds は .metal ファイルをビルドできないため）。
///
/// どの計算も「1 スレッド = 出力 1 つ（か数個）」で、範囲の外のスレッドは何もしない。
enum GGUFMetalSource {
    /// 計算の名前（kernel の関数名）
    enum Kernel: String, CaseIterable {
        case linearQ8 = "linear_q8"
        case linearQ8Batch = "linear_q8_batch"
        case linearF32 = "linear_f32"
        case linearF32Batch = "linear_f32_batch"
        case rmsNorm = "rms_norm"
        case rope = "rope"
        case attentionScores = "attention_scores"
        case attentionSoftmax = "attention_softmax"
        case attentionCombine = "attention_combine"
        case addRows = "add_rows"
        case gatedActivation = "gated_activation"
    }

    /// Args.flags の意味
    enum Flag {
        static let scales: UInt32 = 1
        static let bias: UInt32 = 2
        static let splitPairs: UInt32 = 4
        static let gelu: UInt32 = 8
    }

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
        float pad;
    };

    inline float4 load4(device const packed_char4 *w, uint i) { return float4(char4(w[i])); }
    inline float4 load4(device const packed_float4 *w, uint i) { return float4(w[i]); }

    inline float sum4(float4 v) { return v.x + v.y + v.z + v.w; }

    // 大きな値でも NaN にならない tanh（±15 より外は ±1 と見なせる）
    inline float safe_tanh(float v) { return tanh(clamp(v, -15.0f, 15.0f)); }

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
                           device const float *input, device float *output, constant Args &a, uint2 gid) {
        uint row = gid.x;
        uint n = gid.y;
        if (row >= a.rows || n >= a.count) { return; }
        uint quads = a.columns >> 2;
        device const W *w = weights + row * quads;
        device const packed_float4 *x = (device const packed_float4 *)(input + a.inOffset + n * a.inStride);
        float sum = dot_row(w, x, quads);
        if ((a.flags & 1u) != 0u) { sum *= scales[row]; }
        if ((a.flags & 2u) != 0u) { sum += bias[row]; }
        output[a.outOffset + n * a.outStride + row] = sum;
    }

    // 線形層（トークン 4 個ぶんを 1 スレッドで。重みを読むのが 1 回で済む）。gid.x = 出力の行、gid.y = トークン ÷ 4
    template <typename W>
    inline void linear_four(device const W *weights, device const float *scales, device const float *bias,
                            device const float *input, device float *output, constant Args &a, uint2 gid) {
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
        if ((a.flags & 2u) != 0u) { sums += bias[row]; }
        device float *out = output + a.outOffset + row;
        out[first * a.outStride] = sums.x;
        if (first + 1u <= last) { out[(first + 1u) * a.outStride] = sums.y; }
        if (first + 2u <= last) { out[(first + 2u) * a.outStride] = sums.z; }
        if (first + 3u <= last) { out[(first + 3u) * a.outStride] = sums.w; }
    }

    kernel void linear_q8(device const char *weights [[buffer(0)]],
                          device const float *scales [[buffer(1)]],
                          device const float *bias [[buffer(2)]],
                          device const float *input [[buffer(3)]],
                          device float *output [[buffer(4)]],
                          constant Args &a [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
        linear_one((device const packed_char4 *)(weights + a.weightSkip), scales, bias, input, output, a, gid);
    }

    kernel void linear_q8_batch(device const char *weights [[buffer(0)]],
                                device const float *scales [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device const float *input [[buffer(3)]],
                                device float *output [[buffer(4)]],
                                constant Args &a [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
        linear_four((device const packed_char4 *)(weights + a.weightSkip), scales, bias, input, output, a, gid);
    }

    kernel void linear_f32(device const char *weights [[buffer(0)]],
                           device const float *scales [[buffer(1)]],
                           device const float *bias [[buffer(2)]],
                           device const float *input [[buffer(3)]],
                           device float *output [[buffer(4)]],
                           constant Args &a [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
        linear_one((device const packed_float4 *)(weights + a.weightSkip), scales, bias, input, output, a, gid);
    }

    kernel void linear_f32_batch(device const char *weights [[buffer(0)]],
                                 device const float *scales [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device const float *input [[buffer(3)]],
                                 device float *output [[buffer(4)]],
                                 constant Args &a [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        linear_four((device const packed_float4 *)(weights + a.weightSkip), scales, bias, input, output, a, gid);
    }

    // RMSNorm。gid = 行（rows 行 × columns）。入力と出力は同じ場所でもよい
    kernel void rms_norm(device const float *input [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device float *output [[buffer(2)]],
                         constant Args &a [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
        if (gid >= a.rows) { return; }
        device const float *x = input + a.inOffset + gid * a.columns;
        device float *y = output + a.outOffset + gid * a.columns;
        float sum = 0.0f;
        for (uint i = 0; i < a.columns; i++) {
            sum += x[i] * x[i];
        }
        float factor = 1.0f / sqrt(sum / float(a.columns) + a.epsilon);
        for (uint i = 0; i < a.columns; i++) {
            y[i] = x[i] * factor * weight[i];
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

    // スコアを重み（合計 1）にする。gid = トークン × ヘッド
    kernel void attention_softmax(device float *scores [[buffer(0)]],
                                  constant Args &a [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
        if (gid >= a.count * a.heads) { return; }
        uint position = a.start + gid / a.heads;
        uint first = first_visible(position, a.window);
        device float *row = scores + gid * a.total;
        float peak = row[first];
        for (uint t = first + 1u; t <= position; t++) {
            peak = max(peak, row[t]);
        }
        float sum = 0.0f;
        for (uint t = first; t <= position; t++) {
            float e = exp(row[t] - peak);
            row[t] = e;
            sum += e;
        }
        float inverse = 1.0f / sum;
        for (uint t = first; t <= position; t++) {
            row[t] *= inverse;
        }
    }

    // 重みを付けて v を足し合わせる。gid.x = ヘッドの中の位置 ÷ 4、gid.y = トークン × ヘッド
    kernel void attention_combine(device const float *scores [[buffer(0)]],
                                  device const float *values [[buffer(1)]],
                                  device float *output [[buffer(2)]],
                                  constant Args &a [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        uint quads = a.size >> 2;
        if (gid.x >= quads || gid.y >= a.count * a.heads) { return; }
        uint head = gid.y % a.heads;
        uint position = a.start + gid.y / a.heads;
        uint first = first_visible(position, a.window);
        uint group = head / (a.heads / a.kvHeads);
        uint stride = a.kvHeads * quads;
        device const float *row = scores + gid.y * a.total;
        device const packed_float4 *v = (device const packed_float4 *)values + group * quads + gid.x;
        float4 sum = float4(0.0f);
        for (uint t = first; t <= position; t++) {
            sum += row[t] * float4(v[t * stride]);
        }
        device packed_float4 *out = (device packed_float4 *)(output + gid.y * a.size);
        out[gid.x] = sum;
    }

    // x += y。gid = 要素
    kernel void add_rows(device float *x [[buffer(0)]],
                         device const float *y [[buffer(1)]],
                         constant Args &a [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
        if (gid >= a.count) { return; }
        x[gid] += y[gid];
    }

    // gate = act(gate) × up。gid = 要素
    kernel void gated_activation(device float *gate [[buffer(0)]],
                                 device const float *up [[buffer(1)]],
                                 constant Args &a [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
        if (gid >= a.count) { return; }
        float g = gate[gid];
        float value;
        if ((a.flags & 8u) != 0u) {
            value = 0.5f * g * (1.0f + safe_tanh(0.7978845608028654f * g * (1.0f + 0.044715f * g * g)));
        } else {
            value = g / (1.0f + exp(min(-g, 80.0f)));
        }
        gate[gid] = value * up[gid];
    }
    """
}
