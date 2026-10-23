"""Hugging Face `tokenizers`（Python）の出力を正解として JSON に書き出す。

  python -I Tools/fixtures/tokenizer_expectations.py

Tests/Fixtures/tokenizers/<リポジトリ>/tokenizer.json ごとに
Tests/Fixtures/expectations/tokenizer/<リポジトリ>.json を作る。Swift のテストがこれと比べる。
"""
import json
import os
import random
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "Tests", "Fixtures", "tokenizers")
OUT = os.path.join(ROOT, "Tests", "Fixtures", "expectations", "tokenizer")

LONG = (
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
    "The quick brown fox jumps over the lazy dog. "
    "func fibonacci(_ n: Int) -> [Int] {\n    guard n > 0 else { return [] }\n}\n"
    "価格は1,234,567円（税込）です！ 😀👍🏽🇯🇵 カタカナ ｶﾀｶﾅ ＡＢＣ１２３\n\n"
) * 25

CORPUS = [
    "",
    " ",
    "  ",
    "\n",
    "Hello world",
    " Hello world",
    "Hello  world  ",
    "hello\nworld\n\n\nfoo",
    "\t\ttab\tseparated\t",
    "I'm here, DON'T go! You've we'll they'd it's",
    "こんにちは、世界！",
    "吾輩は猫である。名前はまだ無い。",
    "ひらがな カタカナ 漢字 ｶﾀｶﾅ 全角ＡＢＣ１２３　全角スペース",
    "日本語とEnglishのmixed文章です。2024年10月22日",
    "中文测试，你好世界。",
    "한국어 테스트 문장입니다.",
    "Ελληνικά ί ί é é",
    "emoji 😀 👍🏽 👨‍👩‍👧‍👦 🇯🇵 ❤️",
    "Numbers: 1234567890 3.14159 1,000,000 ¹²³ ½",
    "https://huggingface.co/finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit?x=1&y=2",
    "def foo(x):\n    return x * 2  # comment\n",
    "struct ContentView: View {\n    var body: some View {\n        Text(\"Hello\")\n    }\n}\n",
    "    leading and trailing spaces    ",
    "a\u000bb\u0085c　d e f",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>",
    "<s>[INST] hi [/INST] answer</s>",
    "<start_of_turn>user\nこんにちは<end_of_turn>\n<start_of_turn>model\n",
    "<|user|>\nHi<|end|>\n<|assistant|>\n",
    "<｜User｜>質問<｜Assistant｜><think>\n",
    "[|system|]sys[|endofturn|]\n[|user|]hi\n[|assistant|]",
    "<think>\nreasoning\n</think>\n\nanswer",
    "x" * 300,
    "ー" * 50,
    LONG,
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in sorted(os.listdir(SRC)):
        path = os.path.join(SRC, name, "tokenizer.json")
        if not os.path.exists(path):
            continue
        tok = Tokenizer.from_file(path)
        cases = []
        for text in CORPUS:
            plain = tok.encode(text, add_special_tokens=False).ids
            special = tok.encode(text, add_special_tokens=True).ids
            cases.append({
                "text": text,
                "ids": plain,
                "idsWithSpecial": special,
                "decoded": tok.decode(plain, skip_special_tokens=False),
                "decodedSkipSpecial": tok.decode(special, skip_special_tokens=True),
            })
        rng = random.Random(42)
        vocab = tok.get_vocab_size(with_added_tokens=True)
        decodes = []
        for length in (1, 2, 3, 5, 8, 13, 40):
            for _ in range(4):
                ids = [rng.randrange(vocab) for _ in range(length)]
                decodes.append({
                    "ids": ids,
                    "decoded": tok.decode(ids, skip_special_tokens=False),
                    "decodedSkipSpecial": tok.decode(ids, skip_special_tokens=True),
                })
        # 1 トークンずつの復号（ストリーミング表示の確認用）
        sample = tok.encode(LONG[:400], add_special_tokens=False).ids
        single = [{"id": i, "decoded": tok.decode([i], skip_special_tokens=False)} for i in sample[:120]]
        out = {
            "name": name,
            "vocabSize": vocab,
            "cases": cases,
            "randomDecodes": decodes,
            "singleTokenDecodes": single,
        }
        with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print("wrote", name, len(cases), "cases")


if __name__ == "__main__":
    sys.exit(main())
