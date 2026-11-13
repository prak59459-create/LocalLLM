#!/usr/bin/env python3
"""GGUF のテスト用の小さなファイルと正解を作って Tests/GGUFFixtures/ に置く（リポジトリに入れておく）。

    pip install numpy gguf llama-cpp-python
    python Tools/gguf/make_fixtures.py [--real DIR]

- quant_blocks.json
    量子化の形式ごとに、乱数のブロックと、gguf パッケージで Float に戻した正解。
- tiny-<構造>.gguf / tiny-<構造>.json
    乱数の重みを持つとても小さなモデルと、llama.cpp（llama-cpp-python）で計算した各位置の logits。
- real-<名前>.json（--real で GGUF のあるフォルダを渡したとき）
    本物のモデルについて、llama.cpp の logits の上位と、いくつかの位置の値。
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q
from gguf import quants

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Tests", "GGUFFixtures")

QUANT_TYPES = [
    Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0,
    Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K,
    Q.IQ4_NL, Q.IQ4_XS, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S, Q.IQ3_XXS, Q.IQ3_S,
]


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def make_quant_blocks():
    rng = np.random.default_rng(1234)
    entries = []
    for qtype in QUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        blocks = 4096 // block_size if block_size > 1 else 512
        blocks = min(blocks, 24) if block_size > 1 else blocks
        raw = rng.integers(0, 256, size=(blocks, type_size), dtype=np.uint8)
        with np.errstate(all="ignore"):
            expected = quants.dequantize(raw.reshape(1, -1), qtype).astype(np.float32).reshape(-1)
        entries.append({
            "type": int(qtype), "name": qtype.name, "count": int(blocks * block_size),
            "bytes": b64(raw), "expected": b64(expected),
        })
    with open(os.path.join(OUT, "quant_blocks.json"), "w") as f:
        json.dump({"entries": entries}, f)
    print("quant blocks:", ", ".join(e["name"] for e in entries))


# ---- とても小さなモデル ----

VOCAB = 64
DIM = 32
FF = 64
LAYERS = 3
HEADS = 4
KV_HEADS = 2
CONTEXT = 64


def tiny_vocab(writer):
    tokens = ["<unk>", "<s>", "</s>"] + [f"▁t{i}" for i in range(VOCAB - 3)]
    scores = [0.0] * 3 + [-float(i) for i in range(VOCAB - 3)]
    types = [2, 3, 3] + [1] * (VOCAB - 3)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)


def build_tiny(arch, path, seed):
    rng = np.random.default_rng(seed)
    head_dim = DIM // HEADS
    if arch in ("gemma2", "gemma3"):
        head_dim = 16   # ヘッドの大きさが 埋め込み / ヘッド数 と違う場合
    q_size, kv_size = HEADS * head_dim, KV_HEADS * head_dim

    def mat(rows, cols, scale=0.35):
        return (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols) * 4).astype(np.float32)

    def norm(size):
        return (1.0 + 0.2 * rng.standard_normal(size)).astype(np.float32)

    def bias(size):
        return (0.1 * rng.standard_normal(size)).astype(np.float32)

    writer = gguf.GGUFWriter(path, arch)
    # 名前に "phi3" などが入ると llama.cpp が特定のトークンを要求するので、構造の名前は入れない
    writer.add_name(f"tiny test model {seed}")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(DIM)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FF)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0 if arch != "gemma3" else 1000000.0)
    writer.add_file_type(0)
    if head_dim != DIM // HEADS:
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
    if arch == "gemma2":
        writer.add_uint32("gemma2.attention.sliding_window", 5)
        writer.add_float32("gemma2.attn_logit_softcapping", 3.0)
        writer.add_float32("gemma2.final_logit_softcapping", 4.0)
    if arch == "gemma3":
        writer.add_uint32("gemma3.attention.sliding_window", 5)
        writer.add_uint32("gemma3.attention.sliding_window_pattern", 2)
    tiny_vocab(writer)

    writer.add_tensor("token_embd.weight", mat(VOCAB, DIM, 1.0))
    writer.add_tensor("output_norm.weight", norm(DIM))
    if arch in ("llama", "phi3"):
        writer.add_tensor("output.weight", mat(VOCAB, DIM, 1.0))
    for i in range(LAYERS):
        p = f"blk.{i}."
        writer.add_tensor(p + "attn_norm.weight", norm(DIM))
        if arch == "phi3":
            writer.add_tensor(p + "attn_qkv.weight", mat(q_size + 2 * kv_size, DIM))
        else:
            writer.add_tensor(p + "attn_q.weight", mat(q_size, DIM))
            writer.add_tensor(p + "attn_k.weight", mat(kv_size, DIM))
            writer.add_tensor(p + "attn_v.weight", mat(kv_size, DIM))
        if arch == "qwen2":
            writer.add_tensor(p + "attn_q.bias", bias(q_size))
            writer.add_tensor(p + "attn_k.bias", bias(kv_size))
            writer.add_tensor(p + "attn_v.bias", bias(kv_size))
        writer.add_tensor(p + "attn_output.weight", mat(DIM, q_size))
        if arch in ("qwen3", "gemma3"):
            writer.add_tensor(p + "attn_q_norm.weight", norm(head_dim))
            writer.add_tensor(p + "attn_k_norm.weight", norm(head_dim))
        if arch in ("gemma2", "gemma3"):
            writer.add_tensor(p + "post_attention_norm.weight", norm(DIM))
            writer.add_tensor(p + "post_ffw_norm.weight", norm(DIM))
        writer.add_tensor(p + "ffn_norm.weight", norm(DIM))
        if arch == "phi3":
            writer.add_tensor(p + "ffn_up.weight", mat(2 * FF, DIM))
        else:
            writer.add_tensor(p + "ffn_gate.weight", mat(FF, DIM))
            writer.add_tensor(p + "ffn_up.weight", mat(FF, DIM))
        writer.add_tensor(p + "ffn_down.weight", mat(DIM, FF))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def reference_logits(path, tokens, n_ctx):
    import llama_cpp
    # KV キャッシュは Float32 で持たせる（既定の半精度だと、その丸めの分だけ値がずれる）
    model = llama_cpp.Llama(path, n_ctx=n_ctx, n_batch=512, logits_all=True, verbose=False, n_threads=2, flash_attn=False,
                            type_k=0, type_v=0)
    model.eval(tokens)
    return np.array(model.scores[:len(tokens)], dtype=np.float32)


def make_tiny_models():
    rng = np.random.default_rng(99)
    for index, arch in enumerate(["llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3"]):
        path = os.path.join(OUT, f"tiny-{arch}.gguf")
        build_tiny(arch, path, seed=1000 + index)
        tokens = [1] + [int(t) for t in rng.integers(3, VOCAB, size=13)]
        logits = reference_logits(path, tokens, CONTEXT)
        with open(os.path.join(OUT, f"tiny-{arch}.json"), "w") as f:
            json.dump({"tokens": tokens, "vocab": VOCAB, "logits": b64(logits)}, f)
        print(f"tiny-{arch}: {os.path.getsize(path)} bytes, logits std {logits.std():.3f}")


REAL = [
    ("smollm2-135m-q4_k_m", "SmolLM2-135M-Instruct-Q4_K_M.gguf", "The capital of France is"),
    ("smollm2-135m-q8_0", "SmolLM2-135M-Instruct-Q8_0.gguf", "The capital of France is"),
    ("smollm2-135m-f16", "SmolLM2-135M-Instruct-f16.gguf", "The capital of France is"),
]


def make_real(directory):
    import llama_cpp
    for name, filename, prompt in REAL:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            print("missing", path)
            continue
        model = llama_cpp.Llama(path, n_ctx=256, logits_all=True, verbose=False, n_threads=4, flash_attn=False)
        tokens = model.tokenize(prompt.encode(), add_bos=True, special=True)
        model.eval(tokens)
        logits = np.array(model.scores[len(tokens) - 1], dtype=np.float32)
        top = np.argsort(-logits)[:32]
        probe = list(range(0, len(logits), max(1, len(logits) // 64)))
        # 続けて貪欲に 8 トークン生成したときの列
        generated = []
        current = logits
        for _ in range(8):
            token = int(np.argmax(current))
            generated.append(token)
            model.eval([token])
            current = np.array(model.scores[model.n_tokens - 1], dtype=np.float32)
        with open(os.path.join(OUT, f"real-{name}.json"), "w") as f:
            json.dump({
                "file": filename, "prompt": prompt, "tokens": [int(t) for t in tokens],
                "top": [int(t) for t in top], "topLogits": [float(logits[t]) for t in top],
                "probe": probe, "probeLogits": [float(logits[t]) for t in probe],
                "std": float(logits.std()), "greedy": generated,
            }, f)
        print(name, [model.detokenize([t]).decode(errors="replace") for t in generated])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", help="本物の GGUF を置いたフォルダ")
    parser.add_argument("--skip-tiny", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    make_quant_blocks()
    if not args.skip_tiny:
        make_tiny_models()
    if args.real:
        make_real(args.real)


if __name__ == "__main__":
    sys.exit(main())
