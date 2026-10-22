"""transformers の apply_chat_template と jinja2 の結果を正解として JSON に書き出す。

  python -I Tools/fixtures/chat_template_expectations.py

- Tests/Fixtures/expectations/chat/<リポジトリ>.json : 実際のチャットテンプレートの描画結果
- Tests/Fixtures/expectations/chat/jinja_cases.json   : Jinja の小さなテンプレート集
"""
import json
import os
import sys
from datetime import datetime

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "Tests", "Fixtures", "tokenizers")
OUT = os.path.join(ROOT, "Tests", "Fixtures", "expectations", "chat")

CONVERSATIONS = {
    "single": [{"role": "user", "content": "こんにちは"}],
    "system_user": [
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "自己紹介して"},
    ],
    "multi_turn": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke.\nMake it short."},
    ],
    "with_thinking": [
        {"role": "user", "content": "2+2は?"},
        {"role": "assistant", "content": "<think>\n2 と 2 を足す\n</think>\n\n4 です。"},
        {"role": "user", "content": "では 3+3 は?"},
    ],
    "special_chars": [
        {"role": "user", "content": "Quotes \" ' and braces {{ }} {% %} backslash \\ tab\t end"},
    ],
    "assistant_last": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ],
}


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise Exception(message)


def make_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


JINJA_CASES = [
    "plain text",
    "{{ 'a' }}{{ \"b\" }}{{ 1 }}{{ 1.5 }}{{ true }}{{ none }}",
    "{% for i in [1,2] %}{% if i == 1 %}{% set x = 5 %}{% endif %}{{ x is defined }}{% endfor %}|{{ x is defined }}",
    "{% set x = 1 %}{% for i in [1,2] %}{% set x = x + 1 %}{{ x }}{% endfor %}|{{ x }}",
    "{% if true %}{% set y = 3 %}{% endif %}{{ y }}",
    "{% set g = 7 %}{% macro m(a, b=2) %}{{ a }}-{{ b }}-{{ g }}{% endmacro %}{{ m(1) }}|{{ m(1, b=5) }}|{{ m(3, 4) }}",
    "{{ {'a':1}.b is defined }}|{{ 'abc'.foo is defined }}",
    "{{ none }}|{{ true }}|{{ [1,'a',none,1.5] }}|{{ {'a': 'b', 'c': [1]} }}|{{ 1.0 }}|{{ 7/2 }}|{{ 7//2 }}|{{ -7//2 }}|{{ -7 % 3 }}|{{ 2**10 }}",
    "{{ 'it\\'s' }}|{{ ['it\\'s', 'say \"hi\"'] }}",
    "{{ [1,2,3][::-1] }}|{{ 'hello'[1:3] }}|{{ [1,2,3][-1] }}|{{ 'hello'[::-1] }}|{{ [1,2,3,4,5][1:4:2] }}|{{ 'abc'[:-1] }}|{{ 'abc'[-2:] }}",
    "{{ 'a b  c'.split() }}|{{ 'a,b,,c'.split(',') }}|{{ ' x '.strip() }}|{{ 'xxhixx'.strip('x') }}|{{ 'a</think>b</think>c'.split('</think>')[-1] }}",
    "{{ {'a': 1, 'b': [1,2]} | tojson }}|{{ 'é\\n\"' | tojson }}|{{ {'k': 1.5, 'n': none, 't': true} | tojson(indent=2) }}",
    "{%- for k, v in {'a':1,'b':2}.items() if k != 'b' %}{{ k }}={{ v }}{% endfor %}",
    "{% for x in [1,2,3] if x > 1 %}{{ loop.index }}/{{ loop.length }}{{ loop.last }} {% endfor %}",
    "{% for x in [] %}a{% else %}empty{% endfor %}",
    "{% for x in [1,2,3,4] %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}",
    "{% set ns = namespace(a=1) %}{% for i in range(3) %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }}",
    "  {% if true %}\n  hi\n  {% endif %}\n",
    "a  {%- if true -%}  b  {%- endif -%}  c",
    "{{ 'x' ~ 1 ~ none }}|{{ 'a' + 'b' }}|{{ [1] + [2] }}|{{ 'ab' * 3 }}",
    "{{ not true and false }}|{{ 1 in [1,2] }}|{{ 'a' not in 'abc' }}|{{ 1 if false }}|{{ 'y' if 0 else 'n' }}",
    "{{ [{'role':'user'},{'role':'a'}] | selectattr('role','equalto','user') | list }}",
    "{{ 'Hello World'|lower }}|{{ 'hello world'|title }}|{{ 'hello'|capitalize }}|{{ [3,1,2]|sort }}|{{ ['a','b']|join(', ') }}|{{ [1,2]|first }}|{{ none|default('d') }}|{{ ''|default('d', true) }}|{{ undefined_thing|default('u') }}",
    "{% set x %}inner {{ 1 }}{% endset %}[{{ x }}]",
    "{# comment #}\nafter",
    "{%- set a = 'x' -%}\n\n{{ a }}\n",
    "{{ messages[0]['role'] }}{{ messages[0].role }}{{ messages[-1].content }}",
    "{{ 3 > 2 > 1 }}|{{ 1 == 1.0 }}|{{ 'a' < 'b' }}|{{ [1,2] == [1,2] }}|{{ {'a':1} == {'a':1} }}",
    "{{ (1, 2) }}|{{ () }}|{{ (1,) }}",
    "{{ 10 / 4 }}|{{ 10 / 5 }}|{{ 1e3 }}|{{ 0.1 + 0.2 }}|{{ 1.5|round }}|{{ 2.5|round }}|{{ -3|abs }}|{{ '12'|int + 1 }}|{{ '1.5'|float }}",
    "{% for m in messages %}{{ loop.previtem is defined }}{{ loop.revindex }}{{ loop.first }} {% endfor %}",
    "{{ 'a\\tb' }}|{{ \"double\" 'adjacent' }}",
    "{{ x is none }}|{{ none is none }}|{{ 1 is number }}|{{ 'a' is string }}|{{ {} is mapping }}|{{ [] is iterable }}|{{ 'a' is iterable }}|{{ 1 is iterable }}|{{ true is true }}|{{ 1 is true }}|{{ 3 is odd }}|{{ 4 is divisibleby 2 }}|{{ 4 is divisibleby(3) }}",
    "{{ {'b':1,'a':2}|dictsort }}|{{ [1,2,2,3]|unique|list }}|{{ ['a','b']|reverse|list }}|{{ [1,2,3]|sum }}|{{ [3,1]|max }}|{{ [3,1]|min }}",
    "{{ 'x'|replace('x','y') }}|{{ 'line1\\nline2'|indent(2) }}|{{ 'a b c'|wordcount }}",
    "{%- for m in messages -%}{{ loop.index0 }}{%- endfor -%}",
    "{{ messages|length }}{{ messages[1:]|length }}{{ 'abc'|length }}",
    "{{ [1,2,3]|map('string')|join('-') }}|{{ messages|map(attribute='role')|join(',') }}",
    "{%- if messages[0]['role'] == 'system' -%}\n{%- set sys = messages[0]['content'] -%}\n{%- set msgs = messages[1:] -%}\n{%- else -%}\n{%- set msgs = messages -%}\n{%- endif -%}\n{{ msgs|length }}",
    "{% for message in messages %}\n{{ message.role }}: {{ message.content }}\n{% endfor %}",
    "{%- for message in messages %}\n    {%- if loop.first %}\n        first\n    {%- endif %}\n    {{- message.role }}\n{%- endfor %}",
    "{{ 'Hello' if messages|length > 1 else 'Bye' }}",
    "{% if messages[0].role == 'user' and messages|length == 2 or false %}yes{% endif %}",
    "{{ 'abc'.startswith('a') }}|{{ 'abc'.endswith(('x','c')) }}|{{ 'ABC'.lower() }}|{{ 'a-b-c'.replace('-', '+') }}|{{ 'abcabc'.count('b') }}|{{ 'abc'.find('c') }}",
    "{{ '  lead'.lstrip() }}|{{ 'trail\\n\\n'.rstrip('\\n') }}|{{ '\\n\\nx'.lstrip('\\n') }}",
    "{%- set content = '<think>\\nA\\n</think>\\n\\nB' -%}{{ content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') }}|{{ content.split('</think>')[-1].lstrip('\\n') }}",
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}{%- for message in messages[::-1] %}{%- set index = (messages|length - 1) - loop.index0 %}{%- if ns.multi_step_tool and message.role == 'user' %}{%- set ns.multi_step_tool = false %}{%- set ns.last_query_index = index %}{%- endif %}{%- endfor %}{{ ns.last_query_index }}",
    "{{ raise_exception('boom') }}",
    "{%- macro render(x) -%}[{{ x }}]{%- endmacro -%}{{ render('a') }}{{ render(1 + 2) }}",
    "{% set role_map = {'user': 'U', 'assistant': 'A'} %}{% for m in messages %}{{ role_map[m.role] }}{% endfor %}|{{ 'user' in role_map }}|{{ 'tool' not in role_map }}",
    "{{ 'a' ~ 'b' == 'ab' }}|{{ 1 + 2 * 3 }}|{{ (1 + 2) * 3 }}|{{ 2 ** 3 ** 2 }}|{{ -2 ** 2 }}|{{ 7 % 4 }}",
    "{{ [1, 2, 3] | select('odd') | list }}|{{ [1, 2, 3] | reject('odd') | list }}|{{ [none, 1] | select | list }}",
    "{%- set x = [1, 2] -%}{%- for a in x -%}{%- for b in x -%}{{ a }}{{ b }}{{ loop.index }} {% endfor -%}|{{ loop.index }} {% endfor %}",
    "{{ messages[0].content|trim }}|{{ '  x  '|trim }}|{{ 'xxaxx'|trim('x') }}",
    "{%- for message in messages %}{%- if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}{%- endif %}{%- endfor %}",
    "{{ 'x' if true }}{{ 'y' if false }}|{{ none or 'fallback' }}|{{ 'v' and 'w' }}|{{ 0 or '' }}",
    "{{ [[1, 2], [3]]|length }}|{{ {'a': {'b': 'deep'}}.a.b }}|{{ {'a': {'b': 'deep'}}['a']['b'] }}",
    "line1\n{% if true %}\nline2\n{% endif %}\nline3",
    "{% raw %}{{ not rendered }}{% endraw %}",
    "{{ \"multi\\nline\" }}|{{ 'unicode: 日本語 😀' }}|{{ '日本語'|length }}|{{ '日本語'[1] }}",
    "{{ 5 // 2 }}|{{ 5.0 // 2 }}|{{ -5 // 2 }}|{{ 5 % -3 }}|{{ 1 / 3 }}|{{ 2.0 * 3 }}|{{ 10 - 2.5 }}",
    "{% for a, b in [[1, 2], [3, 4]] %}{{ a + b }} {% endfor %}",
    "{{ dict(a=1, b='x') }}|{{ namespace(x=1).x }}",
    "{{ range(3)|list }}|{{ range(1, 4)|list }}|{{ range(0, 10, 3)|list }}|{{ range(5, 0, -2)|list }}",
    "{%- if not tools is defined %}no tools{%- endif %}|{{ tools is none }}",
    "{{ messages|selectattr('role', 'equalto', 'assistant')|list|length }}|{{ messages|rejectattr('role', 'equalto', 'assistant')|map(attribute='content')|list }}",
    "{%- set ns = namespace() %}{%- set ns.index = 0 %}{%- for m in messages %}{%- set ns.index = ns.index + 1 %}{%- endfor %}{{ ns.index }}",
    "{{ 'a' * 0 }}|{{ [0] * 3 }}|{{ 'abc' < 'abd' }}|{{ [1, 2] < [1, 3] }}",
    "{{ {'z': 1, 'a': 2}|tojson }}|{{ [1, 'two', 3.0, false, none]|tojson }}|{{ {'nested': {'x': [1]}}|tojson(indent=4) }}",
    "{{ '%s-%s'|format('a', 'b') }}|{{ '%d items' % 3 }}|{{ '%s' % 'one' }}",
    "{{ 'Mixed Case'|upper }}|{{ 'hello world foo'|title }}|{{ 'x'|center(5) }}|",
    "{%- for m in messages if m.role == 'user' -%}{{ loop.index }}{{ m.content }}{%- else -%}none{%- endfor -%}",
    "{% if messages %}has{% endif %}|{% if [] %}x{% else %}empty{% endif %}|{% if {} %}x{% else %}emptydict{% endif %}|{% if '' %}x{% else %}emptystr{% endif %}",
    "{{ 'é' == 'é' }}|{{ 'a' in ['a', 'b'] }}|{{ 2 in {1: 'x', 2: 'y'} }}",
    "{#- leading comment -#}\n   text   \n{#- trailing -#}",
    "A{{- ' B ' -}}C",
    "{% for i in range(3) %}{{ i }}{% if not loop.last %}, {% endif %}{% endfor %}",
    "{{ '[%s]' % 'x' }}|{{ 'tab\\there' }}",
    "{%- set reasoning = '' -%}{%- if reasoning -%}R{%- else -%}N{%- endif -%}",
    "{{ messages[0].missing is defined }}|{{ messages[0].missing is not defined }}|{{ messages[0].get('role') }}|{{ messages[0].get('missing', 'dflt') }}",
    "{{ [1,2,3]|batch(2)|list }}|{{ [1,2,3,4]|first }}|{{ [1,2,3,4]|last }}",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    env = make_env()
    jinja_results = []
    context = {"messages": [{"role": "user", "content": " hi "}, {"role": "assistant", "content": "yo"}], "tools": None}
    for template in JINJA_CASES:
        try:
            output = env.from_string(template).render(**context)
            jinja_results.append({"template": template, "output": output, "error": None})
        except Exception as error:  # noqa: BLE001
            jinja_results.append({"template": template, "output": None, "error": type(error).__name__ + ": " + str(error)})
    with open(os.path.join(OUT, "jinja_cases.json"), "w", encoding="utf-8") as f:
        json.dump({"context": context, "cases": jinja_results}, f, ensure_ascii=False, indent=1)
    print("wrote jinja_cases", len(jinja_results))

    from transformers import AutoTokenizer

    for name in sorted(os.listdir(SRC)):
        directory = os.path.join(SRC, name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        except Exception as error:  # noqa: BLE001
            print("skip", name, error)
            continue
        template = tokenizer.chat_template
        if not template:
            print("no template", name)
            continue
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()))
        results = []
        for key, conversation in CONVERSATIONS.items():
            for add_generation_prompt in (True, False):
                for thinking in (None, True, False):
                    extra = {"date_string": "26 Jul 2024"}
                    if thinking is not None:
                        extra["enable_thinking"] = thinking
                    item = {"conversation": key, "addGenerationPrompt": add_generation_prompt, "enableThinking": thinking}
                    try:
                        item["output"] = tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=add_generation_prompt, **extra)
                        item["error"] = None
                    except Exception as error:  # noqa: BLE001
                        item["output"] = None
                        item["error"] = str(error)
                    results.append(item)
        special = {k: v for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)}
        out = {"name": name, "template": template, "bosToken": tokenizer.bos_token, "eosToken": tokenizer.eos_token,
               "specialTokens": special, "conversations": CONVERSATIONS, "results": results}
        with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("wrote", name, len(results))


if __name__ == "__main__":
    sys.exit(main())
