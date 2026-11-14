#!/usr/bin/env python3
"""画像生成のテスト用の正解を作って Tests/ImageFixtures/ に置く（リポジトリに入れておく）。

    pip install numpy torch diffusers tokenizers
    python Tools/image/make_fixtures.py

- dpm_solver.json
    DPM-Solver++（2M）のスケジューラについて、diffusers の DPMSolverMultistepScheduler が出す
    時刻の並びと、乱数のモデル出力を与えたときの各ステップの結果。
- clip_tokens.json
    CLIP のトークナイザ（Hugging Face の tokenizers）でプロンプトを分けた結果。
"""
import json
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Tests", "ImageFixtures")


def make_dpm():
    import torch
    from diffusers import DPMSolverMultistepScheduler

    cases = []
    for steps, prediction in [(1, "epsilon"), (2, "epsilon"), (8, "epsilon"), (20, "epsilon"), (25, "epsilon"), (12, "v_prediction")]:
        scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
            prediction_type=prediction)
        scheduler.set_timesteps(steps)
        rng = np.random.default_rng(steps * 7 + len(prediction))
        sample = torch.tensor(rng.standard_normal(12), dtype=torch.float64) * scheduler.init_noise_sigma
        entry = {
            "steps": steps, "prediction": prediction,
            "timesteps": [int(t) for t in scheduler.timesteps],
            "initial": sample.tolist(), "outputs": [], "samples": [],
        }
        for t in scheduler.timesteps:
            output = torch.tensor(rng.standard_normal(12), dtype=torch.float64)
            sample = scheduler.step(output, t, sample).prev_sample
            entry["outputs"].append(output.tolist())
            entry["samples"].append(sample.tolist())
        cases.append(entry)
    with open(os.path.join(OUT, "dpm_solver.json"), "w") as f:
        json.dump({"cases": cases}, f)
    print("dpm cases:", [(c["steps"], c["prediction"]) for c in cases])


PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "A cute cat, sitting on a red sofa. 4k, highly detailed!",
    "  Extra   spaces\tand\nnewlines  ",
    "don't stop, it's 3:45pm — café naïve façade",
    "富士山と桜、浮世絵風",
    "emoji 🐱🚀 and symbols #1 @home 50% off",
    "",
    "masterpiece, best quality, " + ", ".join(f"detail number {i}" for i in range(40)),
]


def make_clip():
    from tokenizers import Tokenizer

    path = os.path.join(OUT, ".clip_tokenizer.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json", path)
    tokenizer = Tokenizer.from_file(path)
    entries = [{"prompt": prompt, "ids": tokenizer.encode(prompt).ids} for prompt in PROMPTS]
    os.remove(path)
    with open(os.path.join(OUT, "clip_tokens.json"), "w") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False)
    print("clip prompts:", len(entries), [len(e["ids"]) for e in entries])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_dpm()
    make_clip()
    sys.exit(0)
