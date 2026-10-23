#!/usr/bin/env bash
# テスト用の小さなファイル（tokenizer.json や Core ML の metadata.json など）を
# Hugging Face から Tests/Fixtures/ にダウンロードする。モデル本体（重み）は取得しない。
#
#   ./Tools/fetch_fixtures.sh
#
# Tests/Fixtures/ は .gitignore 済み。フィクスチャが無い場合、該当テストはスキップされる。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/Tests/Fixtures"
mkdir -p "$OUT/tokenizers" "$OUT/coreml"

TOKENIZER_REPOS=(
  "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit"
  "finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit"
  "finnvoorhees/coreml-SmolLM2-360M-Instruct-4bit"
  "finnvoorhees/coreml-DeepSeek-R1-Distill-Qwen-1.5B-4bit"
  "XDGCC/coreml-Qwen3-0.6B"
  "pywind/coreml-EXAONE-4.0-1.2B-4bit"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/Phi-3-mini-4k-instruct"
  "unsloth/gemma-2-2b-it"
  "unsloth/mistral-7b-instruct-v0.3"
)
SUPPORT_FILES=(tokenizer.json tokenizer_config.json generation_config.json config.json special_tokens_map.json chat_template.jinja chat_template.json)

for repo in "${TOKENIZER_REPOS[@]}"; do
  dir="$OUT/tokenizers/${repo//\//__}"
  mkdir -p "$dir"
  for f in "${SUPPORT_FILES[@]}"; do
    if [ -s "$dir/$f" ]; then continue; fi
    code=$(curl -sSL -w "%{http_code}" -o "$dir/$f" "https://huggingface.co/$repo/resolve/main/$f" || true)
    if [ "$code" != "200" ]; then rm -f "$dir/$f"; fi
  done
  echo "tokenizer fixture: $repo -> $(ls "$dir" | tr '\n' ' ')"
done

# Core ML の metadata.json（.mlmodelc 内）: リポジトリ, パス
METADATA=(
  "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit|Qwen2.5-0.5B-Instruct-4bit.mlmodelc/metadata.json"
  "finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit|Llama-3.2-1B-Instruct-4bit.mlmodelc/metadata.json"
  "XDGCC/coreml-Qwen3-0.6B|Qwen3-0.6B.mlmodelc/metadata.json"
  "pywind/coreml-EXAONE-4.0-1.2B-4bit|EXAONE-4.0-1.2B-4bit.mlmodelc/metadata.json"
  "aufklarer/Qwen3-0.6B-Chat-CoreML|Qwen3Chat-INT8.mlmodelc/metadata.json"
  "aufklarer/Qwen3.5-0.8B-Chat-CoreML|int8/decoder.mlmodelc/metadata.json"
)
for entry in "${METADATA[@]}"; do
  repo="${entry%%|*}"; path="${entry#*|}"
  name="${repo//\//__}.json"
  if [ -s "$OUT/coreml/$name" ]; then continue; fi
  code=$(curl -sSL -w "%{http_code}" -o "$OUT/coreml/$name" "https://huggingface.co/$repo/resolve/main/$path" || true)
  if [ "$code" != "200" ]; then rm -f "$OUT/coreml/$name"; echo "metadata missing: $repo/$path"; else echo "metadata fixture: $name"; fi
done

# リポジトリのファイル一覧（RepoAnalysis のテスト用）
TREE_REPOS=(
  "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit"
  "mlboydaisuke/qwen2.5-0.5b-coreml"
  "aufklarer/Qwen3-0.6B-Chat-CoreML"
  "apple/mistral-coreml"
  "takanori-ishikawa/Qwen3.5-ANE-CoreML"
  "anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4"
)
mkdir -p "$OUT/trees"
for repo in "${TREE_REPOS[@]}"; do
  name="${repo//\//__}.json"
  if [ -s "$OUT/trees/$name" ]; then continue; fi
  curl -sSL -o "$OUT/trees/$name" "https://huggingface.co/api/models/$repo/tree/main?recursive=true"
  echo "tree fixture: $name"
done

# ANEMLL（分割モデル）の meta.yaml
ANEMLL_REPOS=(
  "anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4"
)
for repo in "${ANEMLL_REPOS[@]}"; do
  name="${repo//\//__}.meta.yaml"
  if [ -s "$OUT/trees/$name" ]; then continue; fi
  curl -sSL -o "$OUT/trees/$name" "https://huggingface.co/$repo/resolve/main/meta.yaml"
  echo "meta fixture: $name"
done
