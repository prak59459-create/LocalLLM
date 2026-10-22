"""MiniGPT（ミニLLM工房）の正解を PyTorch の自動微分で作って JSON に書き出す。

  python -I Tools/fixtures/minigpt_expectations.py

- Tests/Fixtures/expectations/minigpt/reference.json
  小さな設定のモデルについて、重み・入力・損失・勾配・AdamW 2 ステップ後の重み・logits を入れる。
  Swift 側（Tests/LocalLLMCoreTests/MiniGPT）は同じ重みを読み込んで結果を比べる。

PyTorch が入っていなければ何もせずに終わる（テストはスキップされる）。
"""
import json
import math
import os
import sys

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    print("torch / numpy not installed; skipping MiniGPT expectations")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Tests", "Fixtures", "expectations", "minigpt")

CONFIG = {
    "vocabSize": 13,
    "contextLength": 8,
    "embeddingSize": 16,
    "layerCount": 2,
    "headCount": 2,
    "mlpMultiplier": 4,
}
HYPER = {
    "learningRate": 0.01,
    "weightDecay": 0.1,
    "beta1": 0.9,
    "beta2": 0.99,
    "epsilon": 1e-8,
    "gradientClip": 0.5,
}


def parameter_layout(cfg):
    v, t, c = cfg["vocabSize"], cfg["contextLength"], cfg["embeddingSize"]
    m = c * cfg["mlpMultiplier"]
    layout = [("wte", [v, c]), ("wpe", [t, c])]
    for i in range(cfg["layerCount"]):
        p = f"h.{i}."
        layout += [
            (p + "ln1.weight", [c]), (p + "ln1.bias", [c]),
            (p + "attn.qkv.weight", [c, 3 * c]), (p + "attn.qkv.bias", [3 * c]),
            (p + "attn.proj.weight", [c, c]), (p + "attn.proj.bias", [c]),
            (p + "ln2.weight", [c]), (p + "ln2.bias", [c]),
            (p + "mlp.fc.weight", [c, m]), (p + "mlp.fc.bias", [m]),
            (p + "mlp.proj.weight", [m, c]), (p + "mlp.proj.bias", [c]),
        ]
    layout += [("ln_f.weight", [c]), ("ln_f.bias", [c])]
    return layout


def initial_values(layout, rng):
    values = {}
    for name, shape in layout:
        if name.endswith("ln1.weight") or name.endswith("ln2.weight") or name == "ln_f.weight":
            data = 1.0 + 0.1 * rng.standard_normal(shape)
        elif len(shape) == 1:
            data = 0.05 * rng.standard_normal(shape)
        else:
            data = 0.2 * rng.standard_normal(shape)
        # Swift 側は Float32 なので、正解も Float32 に丸めた値から始める
        values[name] = data.astype(np.float32).astype(np.float64)
    return values


def forward(params, idx, cfg):
    b, s = idx.shape
    c, h = cfg["embeddingSize"], cfg["headCount"]
    hs = c // h
    x = params["wte"][idx] + params["wpe"][:s]
    mask = torch.tril(torch.ones(s, s, dtype=torch.bool))
    for i in range(cfg["layerCount"]):
        def p(name):
            return params[f"h.{i}.{name}"]
        a = F.layer_norm(x, (c,), p("ln1.weight"), p("ln1.bias"), 1e-5)
        qkv = a @ p("attn.qkv.weight") + p("attn.qkv.bias")
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, s, h, hs).transpose(1, 2)
        k = k.view(b, s, h, hs).transpose(1, 2)
        v = v.view(b, s, h, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, s, c)
        x = x + y @ p("attn.proj.weight") + p("attn.proj.bias")
        m = F.layer_norm(x, (c,), p("ln2.weight"), p("ln2.bias"), 1e-5)
        hidden = F.gelu(m @ p("mlp.fc.weight") + p("mlp.fc.bias"), approximate="tanh")
        x = x + hidden @ p("mlp.proj.weight") + p("mlp.proj.bias")
    x = F.layer_norm(x, (c,), params["ln_f.weight"], params["ln_f.bias"], 1e-5)
    return x @ params["wte"].T


def loss_of(params, inputs, targets, cfg):
    logits = forward(params, torch.tensor(inputs), cfg)
    return F.cross_entropy(logits.reshape(-1, cfg["vocabSize"]), torch.tensor(targets).reshape(-1))


def flat(tensor):
    return [float(x) for x in tensor.detach().reshape(-1).tolist()]


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(20241022)
    cfg = CONFIG
    layout = parameter_layout(cfg)
    values = initial_values(layout, rng)
    params = {name: torch.tensor(values[name], requires_grad=True) for name, _ in layout}
    ordered = [params[name] for name, _ in layout]

    # 長さ 7（文脈長 8 より短い）のバッチ 3 つ
    inputs = rng.integers(0, cfg["vocabSize"], size=(3, 7)).tolist()
    targets = rng.integers(0, cfg["vocabSize"], size=(3, 7)).tolist()
    # 長さちょうど 8 のバッチ（損失だけ）
    full_inputs = rng.integers(0, cfg["vocabSize"], size=(2, 8)).tolist()
    full_targets = rng.integers(0, cfg["vocabSize"], size=(2, 8)).tolist()
    # 文脈長より長い文脈（最後の 8 個だけ使う）
    context = rng.integers(0, cfg["vocabSize"], size=(11,)).tolist()

    with torch.no_grad():
        full_loss = float(loss_of(params, full_inputs, full_targets, cfg))
        logits = forward(params, torch.tensor([context[-cfg["contextLength"]:]]), cfg)[0, -1]
        short_logits = forward(params, torch.tensor([context[:3]]), cfg)[0, -1]

    loss = loss_of(params, inputs, targets, cfg)
    loss.backward()
    gradients = {name: flat(params[name].grad) for name, _ in layout}

    decay = [params[name] for name, shape in layout if len(shape) >= 2]
    no_decay = [params[name] for name, shape in layout if len(shape) < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": HYPER["weightDecay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=HYPER["learningRate"], betas=(HYPER["beta1"], HYPER["beta2"]), eps=HYPER["epsilon"],
    )
    norms = []
    after = []
    losses = [float(loss.detach())]
    for step in range(2):
        if step > 0:
            optimizer.zero_grad()
            loss = loss_of(params, inputs, targets, cfg)
            losses.append(float(loss.detach()))
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ordered, HYPER["gradientClip"])
        norms.append(float(norm))
        optimizer.step()
        after.append({name: flat(params[name]) for name, _ in layout})

    result = {
        "config": cfg,
        "hyperparameters": HYPER,
        "parameters": [{"name": name, "shape": shape, "value": [float(x) for x in values[name].reshape(-1)]}
                       for name, shape in layout],
        "inputs": inputs,
        "targets": targets,
        "losses": losses,
        "gradients": gradients,
        "gradientNorms": norms,
        "afterSteps": after,
        "fullInputs": full_inputs,
        "fullTargets": full_targets,
        "fullLoss": full_loss,
        "context": context,
        "logits": flat(logits),
        "shortContext": context[:3],
        "shortLogits": flat(short_logits),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "reference.json")
    with open(path, "w") as f:
        json.dump(result, f)
    print(f"wrote {path} (loss {losses[0]:.6f}, grad norm {norms[0]:.4f})")


if __name__ == "__main__":
    main()
