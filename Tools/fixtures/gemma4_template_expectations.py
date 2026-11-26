"""Gemma 4 のチャットテンプレート（ツール・ツールの結果・思考を含む）を jinja2 で描画して、正解を JSON に書き出す。

  python -I Tools/fixtures/gemma4_template_expectations.py

- 入力: Tests/GGUFFixtures/gemma4-chat-template.jinja（unsloth/gemma-4-E2B-it-GGUF の GGUF に入っているもの）
- 出力: Tests/GGUFFixtures/gemma4-chat-template-expected.json

transformers の apply_chat_template と同じ設定（trim_blocks・lstrip_blocks・raise_exception）で描画する。
会話は、アプリが Jinja に渡すのと同じ形（どのメッセージにも content がある・tool の結果は name と tool_call_id）で作る。
"""
import json
import os

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(ROOT, "Tests", "GGUFFixtures")

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "ウェブを検索する",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "検索する言葉"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"},
                                                         "offset": {"type": "integer", "description": "First line"}},
                       "required": ["path"]}}},
]

CALL = {"id": "call00001", "type": "function", "function": {"name": "web_search", "arguments": {"query": "東京 天気"}}}
MIXED = {"id": "call00002", "type": "function", "function": {"name": "read_file", "arguments": {
    "path": "a \"b\" {c}, d.txt", "offset": 3, "all": True, "tags": ["x", "y"], "range": {"from": 1, "to": 2}}}}


def message(role, content="", **extra):
    item = {"role": role, "content": content}
    item.update(extra)
    return item


CASES = [
    ("plain", [message("user", "こんにちは")], False, None, True),
    ("system", [message("system", "あなたは親切です。"), message("user", "自己紹介して")], False, None, True),
    ("tools", [message("system", "You are helpful."), message("user", "東京の天気は？")], True, None, True),
    ("tools_no_system", [message("user", "東京の天気は？")], True, None, True),
    ("tool_result", [message("user", "東京の天気は？"),
                     message("assistant", "", tool_calls=[CALL]),
                     message("tool", "晴れ、25度", name="web_search", tool_call_id="call00001")], True, None, True),
    ("tool_result_thinking", [message("user", "東京の天気は？"),
                              message("assistant", "", tool_calls=[CALL]),
                              message("tool", "晴れ、25度", name="web_search", tool_call_id="call00001")], True, True, True),
    ("after_tool_answer", [message("user", "東京の天気は？"),
                           message("assistant", "調べます。", tool_calls=[CALL]),
                           message("tool", "晴れ、25度", name="web_search", tool_call_id="call00001"),
                           message("assistant", "東京は晴れで 25 度です。"),
                           message("user", "ありがとう")], True, None, True),
    ("mixed_arguments", [message("user", "読んで"),
                         message("assistant", "", tool_calls=[MIXED]),
                         message("tool", "line1\nline2", name="read_file", tool_call_id="call00002")], True, None, True),
    ("thinking", [message("user", "2+2は？")], False, True, True),
    ("thinking_off", [message("user", "2+2は？")], False, False, True),
    ("history_thinking", [message("user", "2+2は？"),
                          message("assistant", "<|channel>thought\n2 と 2 を足す<channel|>4 です。"),
                          message("user", "では 3+3 は？")], False, None, True),
    ("no_generation_prompt", [message("user", "Q"), message("assistant", "A")], False, None, False),
]


def raise_exception(text):
    raise Exception(text)


def main():
    with open(os.path.join(FIXTURES, "gemma4-chat-template.jinja"), encoding="utf-8") as f:
        template = f.read()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    compiled = env.from_string(template)
    results = []
    for name, messages, with_tools, thinking, generation in CASES:
        context = {"messages": messages, "add_generation_prompt": generation, "bos_token": "<bos>", "eos_token": "<turn|>",
                   "tools": TOOLS if with_tools else None}
        if thinking is not None:
            context["enable_thinking"] = thinking
        results.append({"name": name, "messages": messages, "tools": with_tools, "enableThinking": thinking,
                        "addGenerationPrompt": generation, "output": compiled.render(**context)})
    with open(os.path.join(FIXTURES, "gemma4-chat-template-expected.json"), "w", encoding="utf-8") as f:
        json.dump({"tools": TOOLS, "cases": results}, f, ensure_ascii=False, indent=1)
    print("wrote", len(results), "cases")


if __name__ == "__main__":
    main()
