# LocalLLM 計画書

iPad の中だけで動く LLM チャットアプリ。Hugging Face から Core ML 形式のモデルをダウンロードして、オフラインで会話する。
Swift Playgrounds（iPad）の App プロジェクト（`.swiftpm`）で開発する。

## 1. 前提（ヒアリング結果）

| 項目 | 内容 |
|---|---|
| 端末 | iPad（RAM 4〜6GB）、iPadOS 18 以降 |
| 開発環境 | iPad の Swift Playgrounds のみ（Mac なし） |
| 目的 | 個人利用・学習 |
| モデル | Hugging Face 上の変換済み Core ML モデル。日本語必須。3B も試したい |
| MVP 機能 | チャット（ストリーミング）、アプリ内 HF 検索、会話履歴保存（SwiftData）、再開できるダウンロード |
| 設定 | 生成パラメータ、システムプロンプト、速度・メモリ表示 |
| UI 言語 | 日本語 + 英語 |
| 持ち込み方法 | GitHub から zip をダウンロードして Playgrounds で開く |

## 2. 技術的な決定

### 2-1. 外部ライブラリを使わない
最初は `swift-transformers`（Hugging Face 製）を使う予定だったが、最新版は C 言語のライブラリ
（`yyjson`、`swift-crypto`）に依存しており、**Swift Playgrounds は C を含むパッケージをビルドできない**。
そこで次の 3 つを純 Swift で自作する:

1. **トークナイザ**: `tokenizer.json`（BPE、バイトレベル、SentencePiece 風の Metaspace / byte fallback）を読む。
   Python の `tokenizers` と出力が完全一致することをテストで確認する。
2. **Jinja（チャットテンプレート）**: モデルごとの会話の書式（Qwen の ChatML、Llama 3、Gemma など）を
   `tokenizer_config.json` の `chat_template` からそのまま描画する。Python の `transformers` と一致を確認。
   失敗したときは組み込みの書式にフォールバックする。
3. **Hugging Face クライアントとダウンローダ**: 検索 API、ファイル一覧 API、Range ヘッダによる続きからのダウンロード。

### 2-2. 対応するモデル形式
Hugging Face 上の Core ML LLM を調べたところ、形式がばらばらだった。主に対応するのは
**iOS 18 のステートフル形式**（KV キャッシュを `MLState` に持つ）:

- 入力: `input_ids`（Int32, [1, n]）、`causal_mask`（Float16, [1, 1, n, 過去+n]）
  - Apple / HF exporters の命名（`inputIds` / `causalMask` / `keyCache` / `valueCache`）にも対応
- 出力: `logits`（[1, n, 語彙数]）
- 例: `finnvoorhees/coreml-*`（Qwen2.5, Llama 3.2, SmolLM2, DeepSeek R1 Distill）、`XDGCC/coreml-Qwen3-0.6B`、`pywind/coreml-EXAONE-4.0-1.2B-4bit`、`apple/mistral-coreml`

おまけで「ステートレス形式」（`input_ids` → `logits` のみ、毎回全体を再計算）にも対応する。
それ以外（複数ファイルに分割した ANE 向け形式、`position_ids` を要求する形式など）は、
ダウンロード前に `metadata.json` を読んで「非対応」と表示する。

### 2-3. 推論の工夫
- **KV キャッシュの再利用**: 前回の会話と共通の先頭部分は計算し直さない（2 ターン目以降が速い）。
- **チャンク分割の prefill**: 長いプロンプトを 128 トークンずつ流し込み、logits のメモリ（語彙 15 万 × 長さ）を抑える。
- **ストリーミング復号**: UTF-8 の途中（日本語や絵文字が複数トークンに分かれる）で文字化けしないよう、確定した文字だけ表示する。
- **コンテキスト管理**: 収まらないときは古いメッセージから省く。
- **思考モデル**: `<think>…</think>` を折りたたみ表示。Qwen3 の `enable_thinking` を設定で切り替え。

### 2-4. メモリ（RAM 4〜6GB への対策）
- ダウンロード前に「重み + KV キャッシュ + 作業領域」を見積もり、`os_proc_available_memory()` と比べて
  「余裕あり / ぎりぎり / メモリ不足の可能性大」を表示する。
- 3B（4bit で約 1.9GB）は「試験的」扱い。iOS のアプリごとのメモリ上限を超えると強制終了される。
  上限を上げる `increased-memory-limit` の Entitlement は Playgrounds では設定できないため使わない。

### 2-5. 追加機能（2 回目の要望）
- **ミニLLM工房（LLM をゼロから作る）**: 文字単位の小さな GPT（2〜6 層）を iPad の中で一から学習する。
  順伝播・逆伝播・AdamW を純 Swift で手書きし（行列積は Accelerate の BLAS）、PyTorch と勾配が一致することをテストで確認する。
  学習曲線（Swift Charts）と途中経過の文章を見ながら学習し、保存したモデルに文章の続きを書かせられる。
- **マイアシスタント（自分専用のコーディング AI など）**: 「モデル + システムプロンプト + 生成設定 + 参考資料」をひとまとめにする。
  参考資料（自分のコードや文書、フォルダごと取り込める）は BM25 で検索して、質問のたびに関係する部分をプロンプトに入れる。
  ひな形: コーディング先生 / 翻訳 / 要約 / 英会話の相手。
  ※ ダウンロードした Core ML モデル自体の追加学習（ファインチューニング）は、コンパイル済みモデルから勾配を取れないためできない。
- **コンパイルを速く・落ちないように**: Swift Playgrounds は最適化なし（Debug）でビルドするので、型チェックの重い式や巨大な View を避ける。
  CI で関数・式ごとの型チェック時間を計測し、遅いものを分割する。モデルの読み込みは既定で CPU + GPU（Neural Engine 向けの変換待ちや失敗を避ける）。

## 3. 構成

```
LocalLLM/
├── LocalLLM.swiftpm/              ← iPad の Swift Playgrounds で開くアプリ本体
│   ├── Package.swift
│   └── Sources/AppModule/
│       ├── Core/                  ← Foundation だけに依存（Linux でテスト可能）
│       │   ├── Tokenizer/         tokenizer.json の BPE トークナイザ
│       │   ├── Jinja/             チャットテンプレート用 Jinja
│       │   ├── Chat/              メッセージ、テンプレート描画、思考ブロック、プロンプト組み立て
│       │   ├── Generation/        サンプラー、ストリーミング復号、停止判定
│       │   ├── Hub/               HF API、リポジトリ解析、Core ML 形式判定、ダウンロード
│       │   ├── MiniGPT/           ゼロから学習する小さな GPT（順伝播・逆伝播・AdamW）
│       │   ├── Knowledge/         参考資料の分割と BM25 検索
│       │   └── Util/              多言語化、バイト数表示、簡易 Markdown
│       ├── Engine/                Core ML 推論（iOS のみ）
│       ├── Storage/               SwiftData の会話、インストール済みモデル、設定
│       ├── App/                   アプリの入口、チャット制御
│       └── UI/                    SwiftUI の画面
├── Package.swift                  ← 開発用（Core を Linux / Mac でテスト）
├── Tests/LocalLLMCoreTests/       Core の単体テスト
├── Tools/                         テスト用フィクスチャの取得・期待値生成スクリプト
└── docs/ARCHITECTURE.md           モジュール間の契約（型と関数の一覧）
```

## 4. マイルストーン

| # | 内容 | 状態 |
|---|---|---|
| M1 | 計画、骨組み、モジュール間の契約 | ✅ |
| M2 | Core: トークナイザ（Python と完全一致） | ✅ 10 モデルで一致 |
| M3 | Core: Jinja とチャットテンプレート（transformers と一致） | ✅ 10 モデルで一致 |
| M4 | Core: サンプラー、ストリーミング復号、HF API、再開可能ダウンロード | ✅ |
| M5 | Core ML 推論エンジン（ステートフル / ステートレス、KV 再利用、チャンク prefill） | ✅（実機確認待ち） |
| M6 | 画面: 会話一覧、チャット、モデル（おすすめ / 検索 / 詳細 / インストール済み）、設定 | ✅（実機確認待ち） |
| M7 | CI: Linux で Core のテスト、macOS で iOS SDK に対する型チェック | 🚧 |
| M8 | ミニLLM工房（MiniGPT の学習・保存・生成） | ✅ PyTorch と勾配一致（実機確認待ち） |
| M9 | マイアシスタント（ひな形、参考資料の取り込みと検索） | ✅（実機確認待ち） |
| M10 | コンパイル時間の計測と改善 | 🚧 |
| M11 | 実機（iPad）での動作確認と調整 | ⏳ ユーザー |

## 5. 今後の候補
- ファイルアプリからのモデル取り込み（自分で変換したモデルを使う）
- 音声入力・読み上げ
- PDF の参考資料（今はテキストとコードのみ）
- バックグラウンドダウンロード（URLSession のバックグラウンド構成）
