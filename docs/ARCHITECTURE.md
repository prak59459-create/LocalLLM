# アーキテクチャとモジュール間の契約

この文書は、各モジュールが外部に公開する型と関数（＝契約）をまとめたものです。
モジュールを実装するときは、ここに書かれた名前・シグネチャ・意味をそのまま守ってください。
内部の補助型は `private` / `fileprivate` にするか、モジュールの接頭辞を付けて名前の衝突を避けます。

## 全体の方針

- アプリ本体は `LocalLLM.swiftpm`（Swift Playgrounds の App プロジェクト）。ターゲットは `AppModule` の 1 つだけ。
  ソースはすべて `LocalLLM.swiftpm/Sources/AppModule/` 以下に置く。
- **外部パッケージには依存しない。** Swift Playgrounds は C/C++ を含むパッケージ（例: 最新の swift-transformers が使う yyjson、swift-crypto）をビルドできないため。
  トークナイザ、Jinja（チャットテンプレート）、Hugging Face のダウンローダはすべて純 Swift で自作する。
- `Core/` 以下は **Foundation だけ** に依存する（SwiftUI / CoreML / SwiftData / UIKit / os / CryptoKit は使わない）。
  Linux でも `swift test` でコンパイル・テストできること。ネットワークは `#if canImport(FoundationNetworking)` で `import FoundationNetworking`。
  `Float16` は Core では使わない（Engine 側でだけ使う）。
- Swift 5 言語モード（`swift-tools-version: 5.9`）。`public` は付けない（同一モジュール。テストは `@testable import LocalLLMCore`）。
- ユーザーに見える文字列は `L("日本語", "English")`（`Core/Util/Localization.swift`）を使う。エラーの `errorDescription` も同様。
- iOS 18 以降が対象（Core ML のステートフルモデル `MLState` を使うため）。

## フォルダと担当

| フォルダ | 内容 | Linux でテスト |
|---|---|---|
| `Core/Tokenizer` | tokenizer.json を読む BPE トークナイザ | ✅ |
| `Core/Jinja` | チャットテンプレート用の Jinja サブセット | ✅ |
| `Core/Chat` | チャットメッセージ型、テンプレート描画、思考ブロックの分離、プロンプト組み立て | ✅ |
| `Core/Generation` | サンプラー、ストリーミング復号、停止判定、生成パラメータ | ✅ |
| `Core/Hub` | Hugging Face API、リポジトリ解析、Core ML モデルの入出力判定、再開可能ダウンロード | ✅ |
| `Core/Util` | 多言語化、バイト数表示など | ✅ |
| `Engine` | Core ML 推論（ステートフル／ステートレス） | ❌（iOS のみ） |
| `Storage` | SwiftData の会話履歴、インストール済みモデルの管理 | ❌ |
| `UI`, `App` | SwiftUI 画面、アプリ全体の状態 | ❌ |

---

## Core/Tokenizer

`TextTokenizer` プロトコルは `Core/Tokenizer/TextTokenizer.swift` に定義済み（変更しない）。

```swift
enum TokenizerError: Error, LocalizedError {
    case fileNotFound(String)          // 例: tokenizer.json が無い
    case invalidFormat(String)         // JSON の構造が想定外
    case unsupported(String)           // 未対応の model / pre_tokenizer など（例: Unigram, WordPiece）
}

final class HFTokenizer: TextTokenizer {
    /// tokenizer.json（必須）と tokenizer_config.json（任意）の中身から作る。
    init(tokenizerJSON: Data, tokenizerConfigJSON: Data?) throws
    /// ディレクトリ内の tokenizer.json / tokenizer_config.json を読む。
    static func load(directory: URL) throws -> HFTokenizer

    // TextTokenizer
    func encode(_ text: String, addSpecialTokens: Bool) -> [Int]
    func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String
    func tokenID(for token: String) -> Int?
    func token(for id: Int) -> String?

    var vocabSize: Int { get }                 // 通常語彙 + 追加トークンを含めた最大 ID + 1
    var specialTokenIDs: Set<Int> { get }       // added_tokens のうち special == true のもの
    // tokenizer_config.json から（無ければ nil）。値が {"content": "..."} 形式でも文字列に正規化する。
    var bosToken: String? { get }
    var eosToken: String? { get }
    var padToken: String? { get }
    var unkToken: String? { get }
    var addBosToken: Bool? { get }              // tokenizer_config の add_bos_token
    /// tokenizer_config.json の chat_template。文字列、または [{name, template}] の配列（name == "default" を優先）。
    var chatTemplate: String? { get }
}
```

挙動は Hugging Face `tokenizers`（Python）の `Tokenizer.from_file(...).encode(text, add_special_tokens=...)` /
`decode(ids, skip_special_tokens=...)` と **完全に一致** させる（テストで比較する）。
最低限の対応範囲:
- model: `BPE`（`byte_fallback`, `ignore_merges`, `continuing_subword_prefix`, `end_of_word_suffix`, `unk_token`, `fuse_unk`。merges は `"a b"` 形式と `["a","b"]` 形式の両方）
- normalizer: null, `NFC`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `Prepend`, `Replace`（String / Regex）, `Strip`, `Sequence`
- pre_tokenizer: null, `ByteLevel`（`add_prefix_space`, `use_regex`）, `Split`（String / Regex、behavior 全種、`invert`）, `Metaspace`（`replacement`, `prepend_scheme`, `split`）, `Digits`, `Punctuation`, `Whitespace`, `WhitespaceSplit`, `Sequence`
- post_processor: null, `TemplateProcessing`, `ByteLevel`, `Sequence`（`RobertaProcessing` / `BertProcessing` は可能なら）
- decoder: null, `ByteLevel`, `Metaspace`, `Replace`, `ByteFallback`, `Fuse`, `Strip`, `Sequence`
- added_tokens: 入力テキスト中の追加トークンを先に切り出す（`lstrip` / `rstrip` / `single_word` / `normalized` を考慮）。

## Core/Jinja

```swift
/// Jinja の値。
indirect enum JinjaValue {
    case undefined
    case none
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JinjaValue])
    case object(JinjaObject)          // 挿入順を保つ辞書（値型）
    case namespace(JinjaNamespace)    // namespace() の戻り値（参照型、属性代入が可能）
    case function(JinjaFunction)      // 組み込み関数・マクロ
    // 必要なら case を追加してよい
}

struct JinjaObject {                  // 挿入順を保つ辞書
    init()
    init(_ pairs: [(String, JinjaValue)])
    subscript(key: String) -> JinjaValue? { get set }
    var keys: [String] { get }
}

extension JinjaValue {
    /// JSON 風の Swift 値（String, Int, Double, Bool, NSNull, [Any], [String: Any]）から変換する。
    init(any value: Any?)
}

struct JinjaTemplate {
    /// テンプレートを解析する。構文エラーなら throw。
    /// trimBlocks / lstripBlocks の既定値は transformers の apply_chat_template と同じ true。
    init(_ source: String, trimBlocks: Bool = true, lstripBlocks: Bool = true) throws
    /// 変数を与えて描画する。raise_exception() が呼ばれたら JinjaError.raised。
    func render(_ context: [String: JinjaValue]) throws -> String
}

enum JinjaError: Error, LocalizedError {
    case syntax(String)
    case runtime(String)
    case raised(String)   // テンプレート内の raise_exception("...")
}
```

結果は Python の `jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` に
transformers と同じ拡張（`tojson` は `ensure_ascii=False`、`raise_exception`、`strftime_now`、`loopcontrols`）を入れたものと一致させる。

## Core/Chat

```swift
struct ChatMessage: Codable, Hashable, Sendable {
    enum Role: String, Codable, Sendable { case system, user, assistant }
    var role: Role
    var content: String
}

/// テンプレートの描画。Jinja で描画できなければ組み込みの形式（BuiltinChatFormat）にフォールバックする。
struct ChatTemplateRenderer {
    init(template: String?, bosToken: String?, eosToken: String?, fallbackFormat: BuiltinChatFormat? = nil)
    /// - Parameters:
    ///   - addGenerationPrompt: 末尾にアシスタントの発話開始を付けるか
    ///   - enableThinking: Qwen3 などの思考モード切り替え（テンプレート変数 enable_thinking に渡す。nil なら渡さない）
    func render(messages: [ChatMessage], addGenerationPrompt: Bool, enableThinking: Bool?) throws -> String
    /// 実際に使われた方式（.jinja またはフォールバック形式）
    var usesJinja: Bool { get }
    /// テンプレートから推定した「ターン終了」トークン（例: "<|im_end|>", "<|eot_id|>", "<end_of_turn>"）。停止判定に使う。
    var endOfTurnTokens: [String] { get }
}

enum BuiltinChatFormat: String, Codable, CaseIterable, Sendable {
    case chatML      // <|im_start|>role\n...<|im_end|>\n（Qwen, SmolLM2 など）
    case llama3      // <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
    case gemma       // <start_of_turn>user\n...<end_of_turn>\n
    case mistral     // [INST] ... [/INST]
    case phi3        // <|user|>\n...<|end|>\n<|assistant|>\n
    case deepseek    // <｜User｜>...<｜Assistant｜>
    case exaone      // [|system|]...[|endofturn|]
    case plain       // "User: ...\nAssistant: "
    /// テンプレート文字列や語彙に含まれるトークンから推定する。
    static func detect(template: String?, hasToken: (String) -> Bool) -> BuiltinChatFormat
}

/// 生成テキストから <think>...</think> の思考部分を分離する。
struct ThinkingSplit: Equatable {
    var thinking: String?     // 思考部分（無ければ nil）
    var answer: String        // 回答部分
    var isThinkingOpen: Bool  // まだ </think> が来ていない（思考中）
    static func split(_ text: String) -> ThinkingSplit
}

/// システムプロンプトと会話履歴から、コンテキスト長に収まるプロンプトを作る。
struct PromptAssembler {
    struct Result {
        var text: String          // 描画済みのプロンプト文字列
        var tokens: [Int]         // encode(text, addSpecialTokens: false)
        var droppedMessageCount: Int  // 収まらずに捨てた古いメッセージの数
    }
    /// 古いメッセージ（system 以外）から順に捨てて、tokens.count + reserveTokens <= contextLength になるようにする。
    /// 最後の user メッセージだけでも収まらなければ、トークン列の先頭側を切り詰める。
    /// アシスタントの過去の発話からは思考部分（<think>…</think>）を取り除いてから描画する。
    static func assemble(systemPrompt: String?, history: [ChatMessage], renderer: ChatTemplateRenderer,
                         tokenizer: TextTokenizer, contextLength: Int, reserveTokens: Int,
                         enableThinking: Bool?) throws -> Result
}
```

## Core/Generation

```swift
struct GenerationParameters: Codable, Hashable, Sendable {
    var temperature: Double = 0.7        // 0 なら貪欲法（argmax）
    var topK: Int = 40                   // 0 なら無効
    var topP: Double = 0.9               // 1.0 なら無効
    var minP: Double = 0.0               // 0 なら無効
    var repetitionPenalty: Double = 1.1  // 1.0 なら無効
    var repetitionWindow: Int = 64       // ペナルティを掛ける直近トークン数
    var maxNewTokens: Int = 1024
    var seed: UInt64? = nil              // nil なら毎回ランダム
    static let `default` = GenerationParameters()
}

struct TokenSampler {
    init(parameters: GenerationParameters)
    /// logits（語彙サイズの配列）から次のトークン ID を選ぶ。logits は書き換えてよい。
    /// previousTokens は繰り返しペナルティ用（直近 repetitionWindow 個だけ使う）。
    mutating func sample(logits: inout [Float], previousTokens: [Int]) -> Int
}

/// 1 トークンずつ追加して、確定した文字列だけを返す（UTF-8 の途中や結合文字の途中では出さない）。
final class StreamingDetokenizer {
    init(tokenizer: TextTokenizer, skipSpecialTokens: Bool = true)
    func append(_ token: Int) -> String   // 新しく確定した部分（空文字もあり）
    func flush() -> String                // 残りを全部出す
    var text: String { get }              // これまでに出力した全文
    var tokens: [Int] { get }
}

/// 停止文字列（例: "<|im_end|>" がテキストとして出てきた場合）の検出。部分一致中は出力を保留する。
struct StopSequenceMatcher {
    init(stopSequences: [String])
    /// 新しい文字列を入れる。emit は表示してよい部分、stopped は停止文字列が見つかったか。
    mutating func feed(_ chunk: String) -> (emit: String, stopped: Bool)
    mutating func flush() -> String
}

/// 停止トークン ID の集合を作る（eos、テンプレートのターン終了トークンなど）。
enum StopTokenResolver {
    static func resolve(tokenizer: TextTokenizer, eosToken: String?, generationConfigJSON: Data?,
                        configJSON: Data?, extraTokens: [String]) -> Set<Int>
}

/// 生成の統計（UI 表示用）。
struct GenerationStats: Codable, Hashable, Sendable {
    var promptTokens: Int = 0
    var reusedPromptTokens: Int = 0      // KV キャッシュを再利用できたトークン数
    var generatedTokens: Int = 0
    var prefillSeconds: Double = 0
    var decodeSeconds: Double = 0
    var prefillTokensPerSecond: Double { get }
    var decodeTokensPerSecond: Double { get }
    var stopReason: StopReason? = nil
}
enum StopReason: String, Codable, Sendable { case endOfTurn, maxTokens, contextFull, cancelled, error }
```

## Core/Hub

```swift
struct HFModelSummary: Codable, Hashable, Identifiable, Sendable {
    var id: String                // "owner/name"
    var downloads: Int?
    var likes: Int?
    var pipelineTag: String?
    var libraryName: String?
    var tags: [String]
    var lastModified: Date?
    var gated: Bool               // gated が false 以外なら true
}

struct HFFileEntry: Codable, Hashable, Sendable {
    var path: String              // リポジトリ内のパス（例: "Qwen2.5-0.5B-Instruct-4bit.mlmodelc/weights/weight.bin"）
    var size: Int64
    var lfsSHA256: String?        // LFS の oid（sha256）
}

struct HFModelInfo: Codable, Hashable, Sendable {
    var id: String
    var sha: String?
    var tags: [String]
    var pipelineTag: String?
    var libraryName: String?
    var gated: Bool
    var baseModels: [String]      // cardData.base_model（文字列または配列）
    var siblings: [String]        // ファイルパス一覧（サイズなし）
}

enum HFError: Error, LocalizedError {
    case http(status: Int, url: URL)   // 401/403 はトークンが必要、404 は見つからない、など分かりやすい説明を付ける
    case invalidResponse(String)
    case network(String)
}

final class HFClient: @unchecked Sendable {
    static let defaultEndpoint: URL      // https://huggingface.co
    init(token: String? = nil, endpoint: URL = HFClient.defaultEndpoint, session: URLSession = .shared)
    /// GET /api/models?search=...&filter=coreml&pipeline_tag=text-generation&sort=downloads&direction=-1&limit=...
    func searchModels(query: String, coreMLOnly: Bool, textGenerationOnly: Bool, limit: Int) async throws -> [HFModelSummary]
    /// GET /api/models/{repo}
    func modelInfo(repo: String) async throws -> HFModelInfo
    /// GET /api/models/{repo}/tree/{revision}?recursive=true（Link ヘッダのページングをたどる）。ファイルのみ返す。
    func listFiles(repo: String, revision: String) async throws -> [HFFileEntry]
    /// https://huggingface.co/{repo}/resolve/{revision}/{path}（パスは各要素をパーセントエンコード）
    func resolveURL(repo: String, path: String, revision: String) -> URL
    /// 小さいファイル（config.json, metadata.json など）を取得する。
    func fetchData(repo: String, path: String, revision: String) async throws -> Data
    /// 認証ヘッダなど、ダウンロードに付けるヘッダ。
    var authorizationHeaders: [String: String] { get }
}

/// Core ML のテンソルの仕様（metadata.json からも、端末上の MLModelDescription からも作れる共通形式）。
struct TensorSpec: Codable, Hashable, Sendable {
    enum DataType: String, Codable, Sendable { case int32, float16, float32, float64, other }
    var name: String
    var dataType: DataType
    var shape: [Int]                 // 既定の形状
    var shapeRange: [ClosedRange<Int>]?  // 可変形状の範囲（次元ごと）。固定なら nil
}

/// モデルの入出力から判定した推論方式。
enum ModelInterface: Hashable, Sendable {
    /// input_ids / inputIds（Int32, [1, n]）と causal_mask / causalMask（[1, 1, n, past+n]）を受け取り、
    /// KV キャッシュを MLState に持ち、logits（[1, n, vocab]）を返す形式。
    case statefulCausalLM(inputIDsName: String, causalMaskName: String, logitsName: String,
                          maskDataType: TensorSpec.DataType, maxContext: Int)
    /// input_ids（と任意の attention_mask）を受け取り logits を返すだけの形式（毎回全体を再計算）。
    case statelessCausalLM(inputIDsName: String, attentionMaskName: String?, logitsName: String,
                           fixedLength: Int?, maxContext: Int)
    case unsupported(reason: String)

    static func classify(inputs: [TensorSpec], outputs: [TensorSpec], states: [TensorSpec]) -> ModelInterface
    var isSupported: Bool { get }
    var maxContext: Int? { get }
}

/// .mlmodelc/metadata.json を解析する。
struct CoreMLMetadata: Hashable, Sendable {
    var inputs: [TensorSpec]
    var outputs: [TensorSpec]
    var states: [TensorSpec]
    var specificationVersion: Int?
    var storagePrecision: String?
    var availabilityIOS: String?     // 例: "18.0"
    static func parse(_ data: Data) throws -> CoreMLMetadata
    var interface: ModelInterface { get }
    /// KV キャッシュ（state）の合計バイト数
    var stateBytes: Int64 { get }
}

/// リポジトリのファイル一覧から、モデルの候補（.mlmodelc / .mlpackage）と付属ファイルを選ぶ。
struct ModelVariant: Codable, Hashable, Identifiable, Sendable {
    enum Kind: String, Codable, Sendable { case compiled, package }   // .mlmodelc / .mlpackage
    var id: String { path }
    var path: String           // 例: "Qwen3-0.6B.mlmodelc" や "int8/Qwen3Chat-INT8.mlmodelc"
    var kind: Kind
    var files: [HFFileEntry]   // このモデルフォルダ内のファイル
    var totalBytes: Int64
    var displayName: String    // フォルダ名から拡張子を除いたもの
}

struct RepoAnalysis: Hashable, Sendable {
    var variants: [ModelVariant]          // 優先度順（.mlmodelc を .mlpackage より先に）
    var supportFiles: [HFFileEntry]       // tokenizer.json, tokenizer_config.json, config.json, generation_config.json,
                                          // special_tokens_map.json, chat_template.jinja, chat_template.json など
    var hasTokenizer: Bool                // tokenizer.json があるか
    static func analyze(files: [HFFileEntry]) -> RepoAnalysis
    /// 選んだ variant のダウンロード対象（モデルフォルダ + そのモデルに最も近い階層の付属ファイル）。
    func filesToDownload(for variant: ModelVariant) -> [HFFileEntry]
}

/// メモリ使用量の見積もり（重み + KV キャッシュ + 作業領域）。
enum MemoryEstimator {
    static func estimateBytes(weightBytes: Int64, stateBytes: Int64, vocabSize: Int?, prefillChunk: Int) -> Int64
}

/// 1 ファイルの再開可能ダウンロード。destination + ".part" に書き込み、完了したら destination へ移動する。
/// 既に .part があれば Range ヘッダで続きから取得する（206 なら追記、200 なら最初から）。
/// Task のキャンセルで中断し、.part は残す（次回再開できる）。
final class ResumableDownloader: NSObject, @unchecked Sendable {
    init(configuration: URLSessionConfiguration = .default)
    func download(from url: URL, to destination: URL, expectedSize: Int64?, headers: [String: String],
                  progress: @escaping @Sendable (_ bytesOnDisk: Int64) -> Void) async throws
}

/// 複数ファイルを順番にダウンロードし、全体の進捗・速度・残り時間を報告する。失敗時は指数バックオフで再試行する。
struct DownloadProgress: Hashable, Sendable {
    var completedBytes: Int64
    var totalBytes: Int64
    var currentFile: String?
    var bytesPerSecond: Double
    var fractionCompleted: Double { get }
    var estimatedSecondsRemaining: Double? { get }
}

final class ModelDownloadJob: @unchecked Sendable {
    struct Item: Hashable, Sendable { var url: URL; var relativePath: String; var expectedSize: Int64? }
    init(items: [Item], destinationDirectory: URL, headers: [String: String], maxRetries: Int = 5,
         downloader: ResumableDownloader = ResumableDownloader())
    /// 全ファイルを取得する。既に完成しているファイル（サイズ一致）は飛ばす。
    func run(progress: @escaping @Sendable (DownloadProgress) -> Void) async throws
}
```

## Core/Util

```swift
func L(_ japanese: String, _ english: String) -> String    // 定義済み
enum ByteFormat { static func string(_ bytes: Int64) -> String }   // 例: "1.9 GB"
```

---

## Engine（iOS 専用、Core ML）

- `CoreMLLanguageModel`: `MLModel` と `ModelInterface` を持ち、`prefill(tokens:)` / `decode(token:)` で logits（`[Float]`）を返す。
  - ステートフル: `MLState` を保持し、キャッシュに入っているトークン列を覚える。新しいプロンプトと共通の先頭部分は再計算しない
    （マスクの最後の次元 = 過去の長さ + 今回の長さ、で位置が決まるため、共通部分の長さまで巻き戻すだけでよい）。
  - プロンプトはチャンク（既定 128 トークン）に分けて流し込み、logits のメモリを抑える。
  - ステートレス: 毎回トークン列全体を入れる（固定長なら右側をパディング）。
- `InferenceEngine`（actor）: モデルとトークナイザの読み込み、生成ループ（`AsyncThrowingStream` でイベントを返す）、統計。

## Storage / App / UI（iOS 専用）

- SwiftData: `Conversation`（タイトル、作成・更新日時、使ったモデル ID）と `StoredMessage`（役割、本文、順番、統計）。
- `ModelLibrary`（@Observable）: インストール済みモデル（`Application Support/Models/<owner>__<name>/` と `manifest.json`）、ダウンロード中のジョブ。
- `AppSettings`（@Observable、UserDefaults）: 生成パラメータ、システムプロンプト、思考モード、計算ユニット、統計表示、言語、HF トークン。
- 画面: 会話一覧（サイドバー）＋チャット、モデル（おすすめ / 検索 / インストール済み / 詳細とダウンロード）、設定。

---

## Core/MiniGPT（「ミニLLM工房」: iPad の中で小さな GPT をゼロから学習する）

純 Swift の小さな GPT-2 風モデル（文字単位）。順伝播・逆伝播を手書きし、AdamW で学習する。
行列積は `#if canImport(Accelerate)` なら `cblas_sgemm`（iPad の CPU / AMX で高速）、それ以外（Linux のテスト）は素朴なループ。
Swift Playgrounds は最適化なし（-Onone）でビルドされるので、重い処理はできるだけ BLAS / vDSP に任せること。

```swift
enum MiniGPTPreset: String, Codable, CaseIterable, Sendable {
    case tiny     // 2 層, 埋め込み 64, 2 ヘッド, 文脈 64
    case small    // 4 層, 埋め込み 128, 4 ヘッド, 文脈 128
    case medium   // 6 層, 埋め込み 192, 6 ヘッド, 文脈 128
}

struct MiniGPTConfig: Codable, Hashable, Sendable {
    var vocabSize: Int
    var contextLength: Int
    var embeddingSize: Int
    var layerCount: Int
    var headCount: Int
    var mlpMultiplier: Int = 4
    static func preset(_ preset: MiniGPTPreset, vocabSize: Int) -> MiniGPTConfig
    var parameterCount: Int { get }
}

struct TrainingHyperparameters: Codable, Hashable, Sendable {
    var learningRate: Float = 3e-3
    var minLearningRate: Float = 3e-4
    var batchSize: Int = 16
    var weightDecay: Float = 0.01
    var beta1: Float = 0.9
    var beta2: Float = 0.99
    var epsilon: Float = 1e-8
    var gradientClip: Float = 1.0       // 全体ノルムでクリップ（0 なら無効）
    var warmupSteps: Int = 50
    var totalSteps: Int = 2000          // コサイン減衰の長さ
    var seed: UInt64 = 1
}

/// 文字（Unicode スカラー）単位の語彙。ID 0 は <unk>。出現頻度の高い順に maxSize - 1 文字まで。
struct CharVocabulary: Codable, Hashable, Sendable {
    init(text: String, maxSize: Int = 4096)
    var size: Int { get }
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
}

/// 名前付きのパラメータ（値と勾配。行優先の Float 配列）
final class MiniGPTParameter {
    let name: String
    let shape: [Int]
    var value: [Float]
    var gradient: [Float]
}

final class MiniGPT {
    let config: MiniGPTConfig
    /// GPT-2 風の初期化: 重みは N(0, 0.02)、各ブロックの出力射影（attn.proj, mlp.proj）は N(0, 0.02/sqrt(2*層数))、
    /// バイアス 0、LayerNorm は weight 1 / bias 0。乱数は seed から決定的に作る。
    init(config: MiniGPTConfig, seed: UInt64)
    var parameters: [MiniGPTParameter] { get }       // 下の「パラメータの並び」の順
    func parameter(named name: String) -> MiniGPTParameter?
    /// 平均交差エントロピー（勾配は計算しない）。inputs / targets は [バッチ][長さ ≤ contextLength]
    func loss(inputs: [[Int]], targets: [[Int]]) -> Float
    /// 順伝播 + 逆伝播。勾配を gradient に「加算」して損失を返す。
    func lossAndGradients(inputs: [[Int]], targets: [[Int]]) -> Float
    func zeroGradients()
    /// 文脈（最後の contextLength トークンを使う）の次のトークンの logits
    func logits(context: [Int]) -> [Float]
    /// 温度 0 なら argmax。topK 0 なら無効。戻り値は「新しく生成したトークンだけ」（prompt は含まない）。
    /// prompt が空なら ID 0 から始める。
    func generate(prompt: [Int], maxNewTokens: Int, temperature: Float, topK: Int, seed: UInt64) -> [Int]
    /// 保存形式: JSON ヘッダ（config とパラメータ名・形状）+ Float32 リトルエンディアンの重み
    func save(to url: URL) throws
    static func load(from url: URL) throws -> MiniGPT
}

struct AdamWOptimizer {
    init(hyperparameters: TrainingHyperparameters)
    /// 勾配の全体ノルムでクリップ → AdamW 更新（バイアス補正あり）。重み減衰は 2 次元以上のパラメータだけ（PyTorch の AdamW と同じ式）。
    /// 戻り値はクリップ前の勾配ノルム。
    mutating func step(parameters: [MiniGPTParameter], learningRate: Float) -> Float
}

/// 学習ループ（データの切り出し、学習率スケジュール、評価）。
final class MiniGPTTrainer {
    init(model: MiniGPT, tokens: [Int], hyperparameters: TrainingHyperparameters, validationFraction: Double = 0.1)
    var stepIndex: Int { get }
    var currentLearningRate: Float { get }   // 線形ウォームアップ → コサインで minLearningRate まで
    /// ランダムな位置から batchSize 個の窓を切り出して 1 ステップ学習し、学習損失を返す。
    func step() -> Float
    /// 検証データでの平均損失（batches 回ぶん）
    func validationLoss(batches: Int) -> Float
    var hasValidationData: Bool { get }
}
```

**モデル構造（PyTorch の参照実装と完全に同じにする）**: 入力 x = wte[token] + wpe[位置] →（各層）
x = x + attn(ln1(x)); x = x + mlp(ln2(x)) → ln_f → logits = x · wteᵀ（重み共有）。
- 線形層は `y = x W + b`、W の形状は [入力, 出力]（行優先）。
- attn: qkv = x W_qkv + b（出力 3C: q = [0, C), k = [C, 2C), v = [2C, 3C)）、ヘッドは C/H ずつ連続に分割、
  因果マスク付き softmax(q kᵀ / sqrt(C/H)) v → ヘッドを連結 → proj。
- mlp: GELU（tanh 近似、GPT-2 と同じ）。LayerNorm の eps = 1e-5。損失は全位置の平均交差エントロピー。

**パラメータの並び（名前と形状）**: `wte` [V, C], `wpe` [T, C], 各層 i について
`h.i.ln1.weight` [C], `h.i.ln1.bias` [C], `h.i.attn.qkv.weight` [C, 3C], `h.i.attn.qkv.bias` [3C],
`h.i.attn.proj.weight` [C, C], `h.i.attn.proj.bias` [C], `h.i.ln2.weight` [C], `h.i.ln2.bias` [C],
`h.i.mlp.fc.weight` [C, 4C], `h.i.mlp.fc.bias` [4C], `h.i.mlp.proj.weight` [4C, C], `h.i.mlp.proj.bias` [C]、
最後に `ln_f.weight` [C], `ln_f.bias` [C]。

## Core/Knowledge（「マイアシスタント」の参考資料検索）

自分のコードや文書を小さな塊（チャンク）に分け、質問に関係する部分を BM25 で探してプロンプトに入れる。

```swift
struct KnowledgeChunk: Codable, Hashable, Identifiable, Sendable {
    var id: Int
    var source: String        // ファイル名など
    var startLine: Int        // 1 始まり
    var endLine: Int
    var text: String
}

enum KnowledgeChunker {
    /// 行単位で分割する。空行や行頭の宣言（func / class / struct / enum / extension / def / function / # 見出し）を区切りに優先し、
    /// 1 チャンク maxCharacters 文字以内、前のチャンクと overlapLines 行重ねる。
    static func chunks(source: String, text: String, maxCharacters: Int = 1200, overlapLines: Int = 2) -> [KnowledgeChunk]
}

enum KnowledgeTokenizer {
    /// 検索語に分ける: ASCII の英数字の語は小文字に（camelCase / snake_case は部分語と元の語の両方）、
    /// 漢字・ひらがな・カタカナは 2 文字ずつ（1 文字だけならその 1 文字）。
    static func terms(_ text: String) -> [String]
}

struct KnowledgeSearchResult: Hashable, Sendable {
    var chunk: KnowledgeChunk
    var score: Double
}

struct KnowledgeIndex: Codable, Sendable {
    init()
    mutating func add(source: String, text: String)    // 同じ source があれば置き換える
    mutating func remove(source: String)
    var sources: [String] { get }
    var chunkCount: Int { get }
    /// BM25（k1 = 1.2, b = 0.75）。スコア 0 のものは返さない。
    func search(_ query: String, limit: Int = 4) -> [KnowledgeSearchResult]
}

enum KnowledgePromptBuilder {
    /// 検索結果をプロンプトに入れる文章にする（maxCharacters を超えない範囲で上位から）。結果が空なら空文字。
    static func contextBlock(results: [KnowledgeSearchResult], maxCharacters: Int) -> String
}
```
