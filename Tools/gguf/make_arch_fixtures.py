#!/usr/bin/env python3
"""いろいろな構造（llama.cpp の architecture）のとても小さなモデルと、llama.cpp で計算した logits を作る。

    python Tools/gguf/make_arch_fixtures.py [名前 ...]

Tests/GGUFFixtures/tiny-<名前>.gguf と tiny-<名前>.json を書く（make_fixtures.py の tiny-* と同じ形）。
重みは乱数。構造ごとに llama.cpp が読む重みの名前・形とメタデータをそろえてある。
"""
import json
import os
import subprocess
import sys

import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fixtures import OUT, VOCAB, b64, tiny_vocab, reference_logits  # noqa: E402

DIM = 32
FF = 64
HEADS = 4
KV_HEADS = 2
CONTEXT = 64

# 構造ごとの設定
#   norm: rms / layer、norm_bias: 正規化にバイアス、qkv: split / fused、qkv_bias、o_bias
#   ffn: gated / seq / fused、ffn_bias、qk_norm: None / head / head_w（ヘッドごとの重み）/ full
#   pre_norms・ffn_norm・post_norms・output（別の出力層）・output_bias・pos（学習した位置）
#   meta: そのほかのメタデータ、layers: 層の数
BASE = dict(norm="rms", norm_bias=False, qkv="split", qkv_bias=False, o_bias=False, ffn="gated", ffn_bias=False,
            qk_norm=None, pre_norms=True, ffn_norm=True, ffn_norm_name="ffn_norm", post_norms=False, output=False,
            output_bias=False, pos=False, meta={}, layers=3, rope_dims=None, eps=None, heads=HEADS, kv_heads=KV_HEADS,
            attn_norm=True, output_norm=True)

LAYER_NORM = dict(norm="layer", norm_bias=True)

SPECS = {
    "mistral3": dict(),
    "internlm2": dict(qkv="fused", output=True),
    "xverse": dict(output=True),
    "ernie4_5": dict(o_bias=True),
    "granite": dict(meta={"logit_scale": 4.0, "residual_scale": 0.5, "embedding_scale": 3.0, "attention.scale": 0.3}),
    "minicpm": dict(),
    "arcee": dict(ffn="seq"),
    "smollm3": dict(layers=4),
    "maincoder": dict(qk_norm="head"),
    "qwen": dict(qkv="fused", qkv_bias=True, kv_heads=HEADS, ff_meta_double=True, output=True),
    "qwen2vl": dict(qkv_bias=True, meta={"rope.dimension_sections": [2, 1, 1, 0]}),
    "qwen3vl": dict(qk_norm="head", meta={"rope.dimension_sections": [2, 1, 1, 0]}),
    "exaone": dict(rope_freqs=True),
    "exaone4": dict(qk_norm="head", pre_norms=False, post_norms=True),
    "olmo": dict(norm="layer", attn_norm=False, ffn_norm=False, output_norm=False, output=True),
    "olmo2": dict(qk_norm="full", pre_norms=False, post_norms=True, output=True),
    "seed_oss": dict(ffn_norm_name="post_attention_norm", qkv_bias=True, output=True),
    "plamo": dict(ffn_norm=False, kv_heads=HEADS, output=True),
    "openelm": dict(qkv="fused", qk_norm="head", per_layer_heads=([2, 4, 4], [1, 2, 2]), ff_per_layer=[32, 64, 96]),
    "phi2": dict(LAYER_NORM, qkv="fused", qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True, ffn_norm=False,
                 output=True, output_bias=True, rope_dims=4, kv_heads=HEADS),
    "gpt2": dict(LAYER_NORM, qkv="fused", qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True, pos=True, kv_heads=HEADS),
    "starcoder": dict(LAYER_NORM, qkv="fused", qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True, pos=True, kv_heads=1),
    "starcoder2": dict(LAYER_NORM, qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True),
    "stablelm": dict(LAYER_NORM, qkv_bias=True, rope_dims=4, output=True),
    "stablelm-par": dict(LAYER_NORM, arch="stablelm", ffn_norm=False, qk_norm="head_w", rope_dims=4, output=True),
    "gptneox": dict(LAYER_NORM, qkv="fused", qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True, rope_dims=4,
                    kv_heads=HEADS, output=True, meta={"use_parallel_residual": True}),
    "gptneox-seq": dict(LAYER_NORM, arch="gptneox", qkv="fused", qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True,
                        rope_dims=4, kv_heads=HEADS, output=True, meta={"use_parallel_residual": False}),
    "falcon": dict(LAYER_NORM, qkv="fused", ffn="seq", ffn_norm=False, kv_heads=1),
    "falcon-40b": dict(LAYER_NORM, arch="falcon", qkv="fused", ffn="seq", ffn_norm=False, attn_norm_2=True, kv_heads=1),
    "codeshell": dict(LAYER_NORM, qkv="fused", qkv_bias=True, o_bias=True, ffn="seq", ffn_bias=True, output=True),
    "orion": dict(LAYER_NORM, output=True),
    "nemotron": dict(LAYER_NORM, ffn="seq", output=True, rope_dims=4),
    "jais2": dict(LAYER_NORM, o_bias=True, ffn="seq", ffn_bias=True),
    "command-r": dict(norm="layer", ffn_norm=False, meta={"logit_scale": 0.5}),
    "cohere2": dict(norm="layer", ffn_norm=False, layers=4, meta={"logit_scale": 0.5, "attention.sliding_window": 5}),
    "chatglm": dict(qkv="fused", qkv_bias=True, ffn="fused", rope_dims=4, output=True),
    "glm4": dict(qkv_bias=True, ffn="fused", post_norms=True, rope_dims=4),
    "llama-yarn": dict(arch="llama", rope_scaling=("yarn", 4.0, 16)),
    "llama-linear": dict(arch="llama", rope_scaling=("linear", 2.0, None)),
}


def build(name, spec, path, seed):
    spec = {**BASE, **spec}
    arch = spec.get("arch", name)
    rng = np.random.default_rng(seed)
    layers = spec["layers"]
    head_dim = DIM // HEADS

    def mat(rows, cols, scale=0.35):
        return (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols) * 4).astype(np.float32)

    def norm(size):
        return (1.0 + 0.2 * rng.standard_normal(size)).astype(np.float32)

    def bias(size):
        return (0.1 * rng.standard_normal(size)).astype(np.float32)

    if "per_layer_heads" in spec:
        heads, kv_heads = spec["per_layer_heads"]
    else:
        heads, kv_heads = [spec["heads"]] * layers, [spec["kv_heads"]] * layers
    ffs = spec.get("ff_per_layer", [FF] * layers)

    w = gguf.GGUFWriter(path, arch)
    w.add_name(f"tiny test model {seed}")
    w.add_context_length(CONTEXT)
    w.add_embedding_length(DIM)
    w.add_block_count(layers)
    if spec.get("ff_meta_double"):
        w.add_feed_forward_length(2 * FF)
    elif "ff_per_layer" in spec:
        w.add_array(f"{arch}.feed_forward_length", ffs)
    else:
        w.add_feed_forward_length(FF)
    if "per_layer_heads" in spec:
        w.add_array(f"{arch}.attention.head_count", heads)
        w.add_array(f"{arch}.attention.head_count_kv", kv_heads)
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
    else:
        w.add_head_count(heads[0])
        w.add_head_count_kv(kv_heads[0])
    if spec["norm"] == "rms":
        w.add_layer_norm_rms_eps(1e-6)
    else:
        w.add_layer_norm_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    if spec["rope_dims"]:
        w.add_rope_dimension_count(spec["rope_dims"])
    if spec.get("rope_scaling"):
        kind, factor, original = spec["rope_scaling"]
        w.add_string(f"{arch}.rope.scaling.type", kind)
        w.add_float32(f"{arch}.rope.scaling.factor", factor)
        if original:
            w.add_uint32(f"{arch}.rope.scaling.original_context_length", original)
    w.add_file_type(0)
    for key, value in spec["meta"].items():
        full = f"{arch}.{key}"
        if isinstance(value, bool):
            w.add_bool(full, value)
        elif isinstance(value, list):
            w.add_array(full, value)
        elif isinstance(value, int):
            w.add_uint32(full, value)
        else:
            w.add_float32(full, value)
    tiny_vocab(w)

    w.add_tensor("token_embd.weight", mat(VOCAB, DIM, 1.0))
    if spec["pos"]:
        w.add_tensor("position_embd.weight", mat(CONTEXT, DIM, 1.0))
    if spec["output_norm"]:
        w.add_tensor("output_norm.weight", norm(DIM))
        if spec["norm_bias"]:
            w.add_tensor("output_norm.bias", bias(DIM))
    if spec["output"]:
        w.add_tensor("output.weight", mat(VOCAB, DIM, 1.0))
    if spec["output_bias"]:
        w.add_tensor("output.bias", bias(VOCAB))
    if spec.get("rope_freqs"):
        w.add_tensor("rope_freqs.weight", np.array([1.0, 1.5, 3.0, 8.0], dtype=np.float32)[: (spec["rope_dims"] or head_dim) // 2])
    for i in range(layers):
        p = f"blk.{i}."
        q, kv, ff = heads[i] * head_dim, kv_heads[i] * head_dim, ffs[i]
        if spec["pre_norms"] and spec["attn_norm"]:
            w.add_tensor(p + "attn_norm.weight", norm(DIM))
            if spec["norm_bias"]:
                w.add_tensor(p + "attn_norm.bias", bias(DIM))
            if spec.get("attn_norm_2"):
                w.add_tensor(p + "attn_norm_2.weight", norm(DIM))
                w.add_tensor(p + "attn_norm_2.bias", bias(DIM))
        if spec["qkv"] == "fused":
            w.add_tensor(p + "attn_qkv.weight", mat(q + 2 * kv, DIM))
            if spec["qkv_bias"]:
                w.add_tensor(p + "attn_qkv.bias", bias(q + 2 * kv))
        else:
            w.add_tensor(p + "attn_q.weight", mat(q, DIM))
            w.add_tensor(p + "attn_k.weight", mat(kv, DIM))
            w.add_tensor(p + "attn_v.weight", mat(kv, DIM))
            if spec["qkv_bias"]:
                w.add_tensor(p + "attn_q.bias", bias(q))
                w.add_tensor(p + "attn_k.bias", bias(kv))
                w.add_tensor(p + "attn_v.bias", bias(kv))
        w.add_tensor(p + "attn_output.weight", mat(DIM, q))
        if spec["o_bias"]:
            w.add_tensor(p + "attn_output.bias", bias(DIM))
        if spec["qk_norm"] == "head":
            w.add_tensor(p + "attn_q_norm.weight", norm(head_dim))
            w.add_tensor(p + "attn_k_norm.weight", norm(head_dim))
        elif spec["qk_norm"] == "head_w":
            w.add_tensor(p + "attn_q_norm.weight", norm((heads[i], head_dim)))
            w.add_tensor(p + "attn_k_norm.weight", norm((kv_heads[i], head_dim)))
        elif spec["qk_norm"] == "full":
            w.add_tensor(p + "attn_q_norm.weight", norm(q))
            w.add_tensor(p + "attn_k_norm.weight", norm(kv))
        if spec["post_norms"]:
            w.add_tensor(p + "post_attention_norm.weight", norm(DIM))
            w.add_tensor(p + "post_ffw_norm.weight", norm(DIM))
        if spec["pre_norms"] and spec["ffn_norm"]:
            w.add_tensor(p + spec["ffn_norm_name"] + ".weight", norm(DIM))
            if spec["norm_bias"]:
                w.add_tensor(p + spec["ffn_norm_name"] + ".bias", bias(DIM))
        if spec["ffn"] == "gated":
            w.add_tensor(p + "ffn_gate.weight", mat(ff, DIM))
            w.add_tensor(p + "ffn_up.weight", mat(ff, DIM))
        elif spec["ffn"] == "fused":
            w.add_tensor(p + "ffn_up.weight", mat(2 * ff, DIM))
        else:
            w.add_tensor(p + "ffn_up.weight", mat(ff, DIM))
            if spec["ffn_bias"]:
                w.add_tensor(p + "ffn_up.bias", bias(ff))
        w.add_tensor(p + "ffn_down.weight", mat(DIM, ff))
        if spec["ffn_bias"]:
            w.add_tensor(p + "ffn_down.bias", bias(DIM))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def build_gemma4(path, seed):
    """Gemma 4（E2B / E4B の形）: 層ごとに違うヘッドの大きさ・FFN、KV の共有、層ごとの入力、v の正規化"""
    rng = np.random.default_rng(seed)
    layers, full_head, swa_head, ple = 6, 16, 8, 8
    pattern = [True, False, True, False, True, False]
    ffs = [64, 64, 64, 64, 96, 96]
    shared = 2

    def mat(rows, cols, scale=0.35):
        return (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols) * 4).astype(np.float32)

    def norm(size):
        return (1.0 + 0.2 * rng.standard_normal(size)).astype(np.float32)

    arch = "gemma4"
    w = gguf.GGUFWriter(path, arch)
    w.add_name(f"tiny test model {seed}")
    w.add_context_length(CONTEXT)
    w.add_embedding_length(DIM)
    w.add_block_count(layers)
    w.add_array(f"{arch}.feed_forward_length", ffs)
    w.add_head_count(HEADS)
    w.add_head_count_kv(1)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0)
    w.add_float32(f"{arch}.rope.freq_base_swa", 10000.0)
    w.add_key_length(full_head)
    w.add_value_length(full_head)
    w.add_uint32(f"{arch}.attention.key_length_swa", swa_head)
    w.add_uint32(f"{arch}.attention.value_length_swa", swa_head)
    w.add_uint32(f"{arch}.rope.dimension_count", full_head)
    w.add_uint32(f"{arch}.rope.dimension_count_swa", swa_head)
    w.add_float32(f"{arch}.final_logit_softcapping", 4.0)
    w.add_uint32(f"{arch}.attention.sliding_window", 5)
    w.add_array(f"{arch}.attention.sliding_window_pattern", pattern)
    w.add_uint32(f"{arch}.attention.shared_kv_layers", shared)
    w.add_uint32(f"{arch}.embedding_length_per_layer_input", ple)
    w.add_file_type(0)
    tiny_vocab(w)

    w.add_tensor("token_embd.weight", mat(VOCAB, DIM, 1.0))
    w.add_tensor("output_norm.weight", norm(DIM))
    w.add_tensor("per_layer_token_embd.weight", mat(VOCAB, ple * layers, 1.0))
    w.add_tensor("per_layer_model_proj.weight", mat(ple * layers, DIM))
    w.add_tensor("per_layer_proj_norm.weight", norm(ple))
    # 全体を見る層の周波数の割り算（一部の次元は回さない）
    w.add_tensor("rope_freqs.weight", np.array([1, 1, 1, 2, 4, 1e30, 1e30, 1e30], dtype=np.float32))
    for i in range(layers):
        p = f"blk.{i}."
        head = swa_head if pattern[i] else full_head
        q, kv = HEADS * head, head
        owns = i < layers - shared
        w.add_tensor(p + "attn_norm.weight", norm(DIM))
        w.add_tensor(p + "attn_q.weight", mat(q, DIM))
        w.add_tensor(p + "attn_q_norm.weight", norm(head))
        if owns:
            w.add_tensor(p + "attn_k.weight", mat(kv, DIM))
            w.add_tensor(p + "attn_k_norm.weight", norm(head))
            if i != 0:   # 層 0 は v が無い（k を使う）
                w.add_tensor(p + "attn_v.weight", mat(kv, DIM))
        w.add_tensor(p + "attn_output.weight", mat(DIM, q))
        w.add_tensor(p + "post_attention_norm.weight", norm(DIM))
        w.add_tensor(p + "ffn_norm.weight", norm(DIM))
        w.add_tensor(p + "ffn_gate.weight", mat(ffs[i], DIM))
        w.add_tensor(p + "ffn_up.weight", mat(ffs[i], DIM))
        w.add_tensor(p + "ffn_down.weight", mat(DIM, ffs[i]))
        w.add_tensor(p + "post_ffw_norm.weight", norm(DIM))
        w.add_tensor(p + "inp_gate.weight", mat(ple, DIM))
        w.add_tensor(p + "proj.weight", mat(DIM, ple))
        w.add_tensor(p + "post_norm.weight", norm(DIM))
        w.add_tensor(p + "layer_output_scale.weight", np.array([0.5 + 0.1 * i], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def build_lfm2(path, seed):
    """LFM2: 注意の層と、短い畳み込みの層が混ざる（KV のヘッドが 0 の層が畳み込み）"""
    rng = np.random.default_rng(seed)
    layers, length = 4, 3
    kv_heads = [0, 0, KV_HEADS, 0]
    head_dim = DIM // HEADS

    def mat(rows, cols, scale=0.35):
        return (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols) * 4).astype(np.float32)

    def norm(size):
        return (1.0 + 0.2 * rng.standard_normal(size)).astype(np.float32)

    arch = "lfm2"
    w = gguf.GGUFWriter(path, arch)
    w.add_name(f"tiny test model {seed}")
    w.add_context_length(CONTEXT)
    w.add_embedding_length(DIM)
    w.add_block_count(layers)
    w.add_feed_forward_length(FF)
    w.add_head_count(HEADS)
    w.add_array(f"{arch}.attention.head_count_kv", kv_heads)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(1000000.0)
    w.add_uint32(f"{arch}.shortconv.l_cache", length)
    w.add_file_type(0)
    tiny_vocab(w)
    w.add_tensor("token_embd.weight", mat(VOCAB, DIM, 1.0))
    w.add_tensor("token_embd_norm.weight", norm(DIM))
    for i in range(layers):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm(DIM))
        if kv_heads[i] == 0:
            w.add_tensor(p + "shortconv.conv.weight", (0.5 * rng.standard_normal((DIM, length))).astype(np.float32))
            w.add_tensor(p + "shortconv.in_proj.weight", mat(3 * DIM, DIM))
            w.add_tensor(p + "shortconv.out_proj.weight", mat(DIM, DIM))
        else:
            kv = kv_heads[i] * head_dim
            w.add_tensor(p + "attn_q.weight", mat(DIM, DIM))
            w.add_tensor(p + "attn_k.weight", mat(kv, DIM))
            w.add_tensor(p + "attn_v.weight", mat(kv, DIM))
            w.add_tensor(p + "attn_q_norm.weight", norm(head_dim))
            w.add_tensor(p + "attn_k_norm.weight", norm(head_dim))
            w.add_tensor(p + "attn_output.weight", mat(DIM, DIM))
        w.add_tensor(p + "ffn_norm.weight", norm(DIM))
        w.add_tensor(p + "ffn_gate.weight", mat(FF, DIM))
        w.add_tensor(p + "ffn_up.weight", mat(FF, DIM))
        w.add_tensor(p + "ffn_down.weight", mat(DIM, FF))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


SPECIAL = {"gemma4": build_gemma4, "lfm2": build_lfm2}


def reference_or_error(path, tokens):
    """llama.cpp で読めなければ、その理由（読み込みのログ）を返す"""
    try:
        return reference_logits(path, tokens, CONTEXT), None
    except Exception:  # noqa: BLE001
        code = ("import llama_cpp,sys; llama_cpp.Llama(sys.argv[1], n_ctx=64, verbose=True)")
        result = subprocess.run([sys.executable, "-c", code, path], capture_output=True)
        log = result.stderr.decode("utf-8", "replace")
        lines = [line for line in log.splitlines() if "error loading" in line or "missing" in line]
        return None, "\n".join(lines[-6:])


def main():
    every = list(SPECS) + list(SPECIAL)
    names = sys.argv[1:] or every
    rng = np.random.default_rng(7)
    failed = []
    for index, name in enumerate(every):
        tokens = [1] + [int(t) for t in rng.integers(3, VOCAB, size=13)]
        if name not in names:
            continue
        path = os.path.join(OUT, f"tiny-{name}.gguf")
        if name in SPECIAL:
            SPECIAL[name](path, seed=2000 + index)
        else:
            build(name, SPECS[name], path, seed=2000 + index)
        logits, error = reference_or_error(path, tokens)
        if logits is None:
            failed.append(name)
            print(f"tiny-{name}: llama.cpp failed\n{error}")
            os.remove(path)
            continue
        with open(os.path.join(OUT, f"tiny-{name}.json"), "w") as f:
            json.dump({"tokens": tokens, "vocab": VOCAB, "logits": b64(logits)}, f)
        print(f"tiny-{name}: {os.path.getsize(path)} bytes, logits std {logits.std():.3f}")
    if failed:
        print("failed:", " ".join(failed))
        return 1
    if "gemma4" in names:
        check_gemma4()
    return 0



# ---- Gemma 4 の計算を numpy で（llama.cpp の gemma4.cpp と同じ流れ）----

def gemma4_numpy(path, tokens, f16_gelu):
    """f16_gelu: llama.cpp と同じく GELU を半精度の表で計算する（False なら正確な式）"""
    reader = gguf.GGUFReader(path)
    T = {t.name: np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}

    def field(key):
        f = reader.fields[key]
        values = [f.parts[i][0] for i in f.data]
        return values if len(values) > 1 else values[0]

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

    def rms(x, w=None):
        y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return y * w if w is not None else y

    def gelu(x):
        exact = lambda v: 0.5 * v * (1 + np.tanh(np.float32(0.7978845608028654) * v * (1 + np.float32(0.044715) * v * v)))
        if not f16_gelu:
            return exact(x).astype(np.float32)
        y = exact(x.astype(np.float16).astype(np.float32)).astype(np.float16).astype(np.float32)
        y = np.where(x <= -10, 0, y)
        return np.where(x >= 10, x, y).astype(np.float32)

    def rope(v, base, use_factors):
        # v: [n, heads, size]（NEOX: 前半と後半を組にする）
        size = v.shape[-1]
        half = size // 2
        freq = base ** (-2 * np.arange(half) / size)
        if use_factors:
            freq = freq / factors[:half]
        theta = np.arange(n)[:, None] * freq[None, :]
        cos, sin = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]
        a, b = v[..., :half], v[..., half:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(np.float32)

    x = T["token_embd.weight"][tokens] * np.float32(np.sqrt(d))
    # 層ごとの入力
    proj = (x @ T["per_layer_model_proj.weight"].T) / np.float32(np.sqrt(d))
    proj = rms(proj.reshape(n, layers, ple), T["per_layer_proj_norm.weight"])
    table = T["per_layer_token_embd.weight"][tokens].reshape(n, layers, ple) * np.float32(np.sqrt(ple))
    inputs = (proj + table) / np.float32(np.sqrt(2))

    cache = {}
    own = layers - shared
    for i in range(layers):
        p = f"blk.{i}."
        swa = pattern[i]
        h = rms(x, T[p + "attn_norm.weight"])
        q = (h @ T[p + "attn_q.weight"].T).reshape(n, heads, -1)
        size = q.shape[-1]
        q = rope(rms(q, T[p + "attn_q_norm.weight"]), base_swa if swa else base_full, not swa)
        if i < own:
            k = (h @ T[p + "attn_k.weight"].T).reshape(n, -1, size)
            v = (h @ T[p + "attn_v.weight"].T).reshape(n, -1, size) if (p + "attn_v.weight") in T else k.copy()
            k = rope(rms(k, T[p + "attn_k_norm.weight"]), base_swa if swa else base_full, not swa)
            v = rms(v)
            cache[i] = (k, v)
            source = i
        else:
            source = own - (2 if swa else 1)
        k, v = cache[source]
        groups = heads // k.shape[1]
        out = np.zeros((n, heads, size), dtype=np.float32)
        for head in range(heads):
            kk, vv = k[:, head // groups], v[:, head // groups]
            scores = q[:, head] @ kk.T          # 倍率は 1
            for t in range(n):
                first = max(0, t - window + 1) if swa else 0
                mask = np.full(n, -np.inf, dtype=np.float32)
                mask[first:t + 1] = 0
                row = scores[t] + mask
                row = np.exp(row - row.max())
                out[t, head] = (row / row.sum()) @ vv
        attn = out.reshape(n, -1) @ T[p + "attn_output.weight"].T
        attn_out = rms(attn, T[p + "post_attention_norm.weight"]) + x
        f = rms(attn_out, T[p + "ffn_norm.weight"])
        g = gelu(f @ T[p + "ffn_gate.weight"].T) * (f @ T[p + "ffn_up.weight"].T)
        f = rms(g @ T[p + "ffn_down.weight"].T, T[p + "post_ffw_norm.weight"])
        cur = f + attn_out
        gate = gelu(cur @ T[p + "inp_gate.weight"].T) * inputs[:, i]
        cur = cur + rms(gate @ T[p + "proj.weight"].T, T[p + "post_norm.weight"])
        x = (cur * T[p + "layer_output_scale.weight"][0]).astype(np.float32)
    logits = rms(x, T["output_norm.weight"]) @ T["token_embd.weight"].T
    return (cap * np.tanh(logits / cap)).astype(np.float32)


def check_gemma4():
    path = os.path.join(OUT, "tiny-gemma4.gguf")
    ref = json.load(open(os.path.join(OUT, "tiny-gemma4.json")))
    tokens = ref["tokens"]
    llama = np.frombuffer(__import__("base64").b64decode(ref["logits"]), dtype=np.float32).reshape(len(tokens), -1)
    emulated = gemma4_numpy(path, tokens, f16_gelu=True)
    exact = gemma4_numpy(path, tokens, f16_gelu=False)
    print("numpy (半精度の GELU) と llama.cpp の差:", float(np.abs(emulated - llama).max()))
    print("numpy (正確な GELU) と llama.cpp の差:", float(np.abs(exact - llama).max()))
    ref["exact"] = b64(exact)
    json.dump(ref, open(os.path.join(OUT, "tiny-gemma4.json"), "w"))


if __name__ == "__main__":
    sys.exit(main())
