# LocalLLM

iPad の中だけで動く AI チャットアプリです。Hugging Face から Core ML 形式の言語モデルをダウンロードして、
インターネットに接続せずに会話できます。Swift Playgrounds（iPad）でそのまま開いて実行できます。

- 日本語 / 英語の UI
- Hugging Face のモデル検索とダウンロード（途中で止めても続きから再開）
- ストリーミング表示のチャット、会話履歴の保存
- 生成パラメータ（温度、top-p、top-k など）、システムプロンプト、思考モードの設定
- 速度（tok/s）とメモリ使用量の表示
- 外部ライブラリなし（トークナイザ、チャットテンプレート、ダウンローダまで純 Swift で実装）

詳しい設計は [PLAN.md](PLAN.md) と [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) を見てください。

## 必要なもの

- iPadOS 18 以降の iPad（RAM 4GB 以上。小さいモデルから試すのがおすすめ）
- [Swift Playgrounds](https://apps.apple.com/app/swift-playgrounds/id908519492)（最新版）
- モデル用の空き容量（300MB〜2GB 程度）

## iPad で開く手順

1. iPad の Safari でこのリポジトリを開き、**Code → Download ZIP** でダウンロードします。
2. **ファイル** アプリで、ダウンロードした zip をタップして展開します。
3. 展開したフォルダの中の **`LocalLLM.swiftpm`** をタップすると、Swift Playgrounds で開きます。
4. 右上の ▶︎（実行）ボタンを押すとアプリが起動します。
   - 「アプリ」ボタンから自分の iPad のホーム画面に追加することもできます。

## 使い方

1. 左下の **モデル** → **おすすめ** から、まず **Qwen2.5 0.5B Instruct (4bit)**（約 290MB）を選んで **ダウンロード** します。
2. ダウンロードが終わったら **使う** をタップします（初回の読み込みは少し時間がかかります）。
3. 左上の **新しいチャット** を押して話しかけます。

### おすすめモデル

| モデル | サイズ | 日本語 | メモ |
|---|---|---|---|
| Qwen2.5 0.5B Instruct (4bit) | 約 290MB | ○ | 最初の 1 つに |
| Llama 3.2 1B Instruct (4bit) | 約 700MB | △ | 英語が得意 |
| Qwen3 0.6B (FP16) | 約 1.2GB | ○ | 思考モード対応 |
| DeepSeek R1 Distill Qwen 1.5B (4bit) | 約 1.0GB | △ | 推論モデル |
| Qwen2.5 3B Instruct (4bit) | 約 1.9GB | ◎ | 試験的（RAM 4〜6GB では落ちることがあります） |

検索画面からは、ほかの Core ML モデルも探せます。対応している形式かどうかは、ダウンロード前に詳細画面で表示されます。

## 困ったとき

- **アプリが落ちる**: メモリ不足の可能性が高いです。もっと小さいモデルを使うか、他のアプリを閉じてください。
  設定の「プロンプトの分割サイズ」を小さくすると、少しメモリを節約できます。
- **ダウンロードが止まった**: モデル画面の **再開** を押すと続きからダウンロードします。
- **ゲート付きモデル**: Hugging Face で利用規約に同意したうえで、設定に Hugging Face のトークンを入力してください。

## 開発者向け

Core（Apple 専用フレームワークに依存しない部分）は Mac / Linux でテストできます。

```sh
./Tools/fetch_fixtures.sh                # テスト用の tokenizer.json などを取得
pip install tokenizers jinja2 transformers
for f in Tools/fixtures/*.py; do python -I "$f"; done   # Python の参照実装で期待値を生成
swift test
```

GitHub Actions では、Linux で Core のテスト、macOS で iOS SDK に対するアプリ全体の型チェックを実行します。
