#!/usr/bin/env python3
"""Gemma 4 の MTP（先読み）用の小さなモデル（gemma4-assistant）と、numpy で計算した正解を作る。

    python -I Tools/gguf/make_mtp_fixtures.py

- Tests/GGUFFixtures/tiny-gemma4-assistant.gguf : tiny-gemma4.gguf（本体）と組になる、乱数の重みの先読み用モデル
- Tests/GGUFFixtures/tiny-gemma4-assistant.json : 本体の隠れ状態・先読みの logits・予想したトークン（numpy で計算）

計算は llama.cpp の src/models/gemma4-assistant.cpp と同じ流れ:
入力 = [本体の埋め込み(トークン) × √(本体の大きさ), 本体の隠れ状態] → 射影 → 層（q だけを持ち、k・v は本体の
KV キャッシュ: 直近だけを見る層は本体の最後から 2 番目の層、全体を見る層は最後の層のもの）→ 正規化 →
logits（自分の埋め込みの表）と、次の隠れ状態（射影）。どの予想も、位置は同じ（本体がまだ読んでいない位置）。
"""
import base64
import json
import os
import sys

import numpy as np
import gguf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Tests", "GGUFFixtures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fixtures import VOCAB, tiny_vocab  # noqa: E402

TARGET = os.path.join(OUT, "tiny-gemma4.gguf")
DRAFTER = os.path.join(OUT, "tiny-gemma4-assistant.gguf")

DIM, LAYERS, HEADS, FF = 16, 2, 2, 32
FULL_HEAD, SWA_HEAD = 16, 8
PATTERN = [True, False]
WINDOW = 5


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode()


def build_drafter(target_dim, seed=7):
    rng = np.random.default_rng(seed)

    def mat(rows, cols, scale=0.35):
        return (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols) * 4).astype(np.float32)

    def norm(size):
        return (1.0 + 0.2 * rng.standard_normal(size)).astype(np.float32)

    arch = "gemma4-assistant"
    w = gguf.GGUFWriter(DRAFTER, arch)
    w.add_name("tiny test drafter")
    w.add_context_length(64)
    w.add_embedding_length(DIM)
    w.add_block_count(LAYERS)
    w.add_uint32(f"{arch}.feed_forward_length", FF)
    w.add_head_count(HEADS)
    w.add_head_count_kv(1)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0)
    w.add_float32(f"{arch}.rope.freq_base_swa", 10000.0)
    w.add_key_length(FULL_HEAD)
    w.add_value_length(FULL_HEAD)
    w.add_uint32(f"{arch}.attention.key_length_swa", SWA_HEAD)
    w.add_uint32(f"{arch}.attention.value_length_swa", SWA_HEAD)
    w.add_uint32(f"{arch}.rope.dimension_count", FULL_HEAD)
    w.add_uint32(f"{arch}.rope.dimension_count_swa", SWA_HEAD)
    w.add_uint32(f"{arch}.attention.sliding_window", WINDOW)
    w.add_array(f"{arch}.attention.sliding_window_pattern", PATTERN)
    w.add_uint32(f"{arch}.attention.shared_kv_layers", LAYERS)
    w.add_uint32(f"{arch}.embedding_length_per_layer_input", 0)
    w.add_uint32(f"{arch}.embedding_length_out", target_dim)
    w.add_uint32(f"{arch}.nextn_predict_layers", LAYERS)
    w.add_file_type(0)
    tiny_vocab(w)

    w.add_tensor("token_embd.weight", mat(VOCAB, DIM, 1.0))
    w.add_tensor("output_norm.weight", norm(DIM))
    w.add_tensor("rope_freqs.weight", np.array([1, 1, 1, 2, 4, 1e30, 1e30, 1e30], dtype=np.float32))
    w.add_tensor("nextn.pre_projection.weight", mat(DIM, 2 * target_dim))
    w.add_tensor("nextn.post_projection.weight", mat(target_dim, DIM))
    for i in range(LAYERS):
        p = f"blk.{i}."
        head = SWA_HEAD if PATTERN[i] else FULL_HEAD
        w.add_tensor(p + "attn_norm.weight", norm(DIM))
        w.add_tensor(p + "attn_q.weight", mat(HEADS * head, DIM))
        w.add_tensor(p + "attn_q_norm.weight", norm(head))
        w.add_tensor(p + "attn_output.weight", mat(DIM, HEADS * head))
        w.add_tensor(p + "post_attention_norm.weight", norm(DIM))
        w.add_tensor(p + "ffn_norm.weight", norm(DIM))
        w.add_tensor(p + "ffn_gate.weight", mat(FF, DIM))
        w.add_tensor(p + "ffn_up.weight", mat(FF, DIM))
        w.add_tensor(p + "ffn_down.weight", mat(DIM, FF))
        w.add_tensor(p + "post_ffw_norm.weight", norm(DIM))
        w.add_tensor(p + "layer_output_scale.weight", np.array([0.7 + 0.1 * i], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read(path):
    reader = gguf.GGUFReader(path)
    tensors = {t.name: np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}

    def field(key):
        f = reader.fields[key]
        values = [f.parts[i][0] for i in f.data]
        return values if len(values) > 1 else values[0]

    return tensors, field


def rms(x, eps, w=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (y * w if w is not None else y).astype(np.float32)


def gelu(v):
    return (0.5 * v * (1 + np.tanh(np.float32(0.7978845608028654) * v * (1 + np.float32(0.044715) * v * v)))).astype(np.float32)


def rope(v, positions, base, factors):
    # v: [n, heads, size]（NEOX: 前半と後半を組にする）、positions: [n]
    size = v.shape[-1]
    half = size // 2
    freq = base ** (-2 * np.arange(half) / size)
    if factors is not None:
        freq = freq / factors[:half]
    theta = np.asarray(positions, dtype=np.float64)[:, None] * freq[None, :]
    cos, sin = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]
    a, b = v[..., :half], v[..., half:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(np.float32)


def target_forward(tokens):
    """本体（tiny-gemma4）: 各位置の logits、最後の正規化のあとの隠れ状態、KV キャッシュ（層ごと）"""
    T, field = read(TARGET)
    arch = "gemma4"
    layers = int(field(f"{arch}.block_count"))
    d = int(field(f"{arch}.embedding_length"))
    heads = int(field(f"{arch}.attention.head_count"))
    eps = float(field(f"{arch}.attention.layer_norm_rms_epsilon"))
    pattern = [bool(v) for v in field(f"{arch}.attention.sliding_window_pattern")]
    window = int(field(f"{arch}.attention.sliding_window"))
    shared = int(field(f"{arch}.attention.shared_kv_layers"))
    ple = int(field(f"{arch}.embedding_length_per_layer_input"))
    cap = float(field(f"{arch}.final_logit_softcapping"))
    base_full = float(field(f"{arch}.rope.freq_base"))
    base_swa = float(field(f"{arch}.rope.freq_base_swa"))
    factors = T["rope_freqs.weight"]
    n = len(tokens)
    positions = np.arange(n)

    x = T["token_embd.weight"][tokens] * np.float32(np.sqrt(d))
    proj = (x @ T["per_layer_model_proj.weight"].T) / np.float32(np.sqrt(d))
    proj = rms(proj.reshape(n, layers, ple), eps, T["per_layer_proj_norm.weight"])
    table = T["per_layer_token_embd.weight"][tokens].reshape(n, layers, ple) * np.float32(np.sqrt(ple))
    inputs = (proj + table) / np.float32(np.sqrt(2))

    cache = {}
    own = layers - shared
    for i in range(layers):
        p = f"blk.{i}."
        swa = pattern[i]
        h = rms(x, eps, T[p + "attn_norm.weight"])
        q = (h @ T[p + "attn_q.weight"].T).reshape(n, heads, -1)
        size = q.shape[-1]
        q = rope(rms(q, eps, T[p + "attn_q_norm.weight"]), positions, base_swa if swa else base_full, None if swa else factors)
        if i < own:
            k = (h @ T[p + "attn_k.weight"].T).reshape(n, -1, size)
            v = (h @ T[p + "attn_v.weight"].T).reshape(n, -1, size) if (p + "attn_v.weight") in T else k.copy()
            k = rope(rms(k, eps, T[p + "attn_k_norm.weight"]), positions, base_swa if swa else base_full,
                     None if swa else factors)
            v = rms(v, eps)
            cache[i] = (k, v)
            source = i
        else:
            source = own - (2 if swa else 1)
        k, v = cache[source]
        out = np.zeros((n, heads, size), dtype=np.float32)
        for head in range(heads):
            kk, vv = k[:, 0], v[:, 0]
            scores = q[:, head] @ kk.T
            for t in range(n):
                first = max(0, t - window + 1) if swa else 0
                row = scores[t, first:t + 1]
                row = np.exp(row - row.max())
                out[t, head] = (row / row.sum()) @ vv[first:t + 1]
        attn = out.reshape(n, -1) @ T[p + "attn_output.weight"].T
        attn_out = rms(attn, eps, T[p + "post_attention_norm.weight"]) + x
        f = rms(attn_out, eps, T[p + "ffn_norm.weight"])
        g = gelu(f @ T[p + "ffn_gate.weight"].T) * (f @ T[p + "ffn_up.weight"].T)
        f = rms(g @ T[p + "ffn_down.weight"].T, eps, T[p + "post_ffw_norm.weight"])
        cur = f + attn_out
        gate = gelu(cur @ T[p + "inp_gate.weight"].T) * inputs[:, i]
        cur = cur + rms(gate @ T[p + "proj.weight"].T, eps, T[p + "post_norm.weight"])
        x = (cur * T[p + "layer_output_scale.weight"][0]).astype(np.float32)
    hidden = rms(x, eps, T["output_norm.weight"])
    logits = hidden @ T["token_embd.weight"].T
    sources = (own - 2, own - 1)   # 先読みが使う、直近だけを見る層と全体を見る層の KV
    return (cap * np.tanh(logits / cap)).astype(np.float32), hidden, cache, sources, T, d


def drafter_step(token, h, pos0, target):
    """先読み 1 回: 位置 pos0 のトークン token と、位置 pos0 − 1 の本体の隠れ状態 h → (logits, 次の隠れ状態)"""
    _, _, cache, sources, Tt, target_dim = target
    D, field = read(DRAFTER)
    arch = "gemma4-assistant"
    eps = float(field(f"{arch}.attention.layer_norm_rms_epsilon"))
    base_full = float(field(f"{arch}.rope.freq_base"))
    base_swa = float(field(f"{arch}.rope.freq_base_swa"))
    window = int(field(f"{arch}.attention.sliding_window"))
    factors = D["rope_freqs.weight"]

    x = Tt["token_embd.weight"][token] * np.float32(np.sqrt(target_dim))
    cur = np.concatenate([x, h]).astype(np.float32) @ D["nextn.pre_projection.weight"].T
    for i in range(LAYERS):
        p = f"blk.{i}."
        swa = PATTERN[i]
        hn = rms(cur, eps, D[p + "attn_norm.weight"])
        q = (hn @ D[p + "attn_q.weight"].T).reshape(1, HEADS, -1)
        q = rope(rms(q, eps, D[p + "attn_q_norm.weight"]), [pos0], base_swa if swa else base_full,
                 None if swa else factors)[0]
        k, v = cache[sources[0] if swa else sources[1]]
        # 本体の KV キャッシュは位置 0 … pos0 − 1（pos0 はまだ無い）。直近だけを見る層は、pos0 から window 未満の位置
        first = max(0, pos0 - window + 1) if swa else 0
        kk, vv = k[first:pos0, 0], v[first:pos0, 0]
        out = np.zeros_like(q)
        for head in range(HEADS):
            scores = q[head] @ kk.T
            row = np.exp(scores - scores.max())
            out[head] = (row / row.sum()) @ vv
        attn = out.reshape(-1) @ D[p + "attn_output.weight"].T
        attn_out = rms(attn, eps, D[p + "post_attention_norm.weight"]) + cur
        f = rms(attn_out, eps, D[p + "ffn_norm.weight"])
        g = gelu(f @ D[p + "ffn_gate.weight"].T) * (f @ D[p + "ffn_up.weight"].T)
        f = rms(g @ D[p + "ffn_down.weight"].T, eps, D[p + "post_ffw_norm.weight"])
        cur = ((f + attn_out) * D[p + "layer_output_scale.weight"][0]).astype(np.float32)
    final = rms(cur, eps, D["output_norm.weight"])
    return (final @ D["token_embd.weight"].T).astype(np.float32), (final @ D["nextn.post_projection.weight"].T).astype(np.float32)


def main():
    _, field = read(TARGET)
    build_drafter(int(field("gemma4.embedding_length")))
    # 本体の numpy の計算が、tiny-gemma4.json の正確な GELU の値と同じか（先読みの正解の土台）
    reference = json.load(open(os.path.join(OUT, "tiny-gemma4.json")))
    exact = np.frombuffer(base64.b64decode(reference["exact"]), dtype=np.float32).reshape(len(reference["tokens"]), -1)
    check = float(np.abs(target_forward(reference["tokens"])[0] - exact).max())
    print("本体の numpy と tiny-gemma4.json（正確な GELU）の差:", check)
    assert check < 1e-4
    tokens = [1, 5, 9, 14, 3, 22, 40, 7, 31, 2, 18, 11]
    target = target_forward(tokens)
    logits, hidden = target[0], target[1]
    cases = []
    # 本体が tokens[:n] を読んだところから: 次のトークン（本体の logits の最大）を 3 回先読みする
    for n in (4, 9, 12):
        prefix = target_forward(tokens[:n])
        first = int(np.argmax(prefix[0][n - 1]))
        h = prefix[1][n - 1]
        steps, drafts = [], []
        token = first
        for _ in range(3):
            step_logits, h = drafter_step(token, h, n, prefix)
            token = int(np.argmax(step_logits))
            steps.append(step_logits)
            drafts.append(token)
        cases.append({"prefix": n, "first": first, "drafts": drafts, "logits": b64(np.stack(steps))})
    with open(os.path.join(OUT, "tiny-gemma4-assistant.json"), "w") as f:
        json.dump({"tokens": tokens, "vocab": VOCAB, "hidden": b64(hidden), "targetLogits": b64(logits), "cases": cases}, f)
    print("tiny-gemma4-assistant:", os.path.getsize(DRAFTER), "bytes;", [c["drafts"] for c in cases])


if __name__ == "__main__":
    sys.exit(main())
