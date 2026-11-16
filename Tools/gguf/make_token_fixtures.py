#!/usr/bin/env python3
"""GGUF に入っているトークナイザで、いくつかの文を llama.cpp に分けさせた正解を作る。

    python Tools/gguf/make_token_fixtures.py 名前 path/to/model.gguf

Tests/GGUFFixtures/tokens-<名前>.json に {"texts": [...], "tokens": [[...], ...]} を書く。
"""
import json
import os
import sys

import llama_cpp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXTS = [
    "Hello world",
    " leading space and  double  spaces",
    "日本語の文章です。今日は良い天気ですね！",
    "line one\nline two\n\n\nafter blank lines\n",
    "<start_of_turn>user\nこんにちは<end_of_turn>\n<start_of_turn>model\n",
    "<|turn>user\nWrite a haiku about the sea.<turn|>\n<|turn>model\n",
    "def add(a, b):\n    return a + b  # 足し算\n",
    "Emoji 🚀🎉 and symbols ©®™ ½ ∑ ≠",
    "1234567890 3.14159 0xDEADBEEF",
    "   \t tabs\tand\r\nwindows newlines",
    "Cafe\u0301 and \u304b\u3099\u304d\u3099 (combining marks)",
]


def main():
    name, path = sys.argv[1], sys.argv[2]
    model = llama_cpp.Llama(path, vocab_only=True, verbose=False)
    tokens = [model.tokenize(text.encode(), add_bos=True, special=True) for text in TEXTS]
    out = os.path.join(ROOT, "Tests", "GGUFFixtures", f"tokens-{name}.json")
    with open(out, "w") as f:
        json.dump({"texts": TEXTS, "tokens": tokens}, f, ensure_ascii=False)
    for text, ids in zip(TEXTS, tokens):
        print(len(ids), repr(text[:30]))


if __name__ == "__main__":
    main()
