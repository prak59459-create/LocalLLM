import Foundation

/// GGUF のおすすめの続き（どれも Hugging Face にあり、Q4_K_M のファイルの大きさと、
/// 先頭に書かれた構造（llama・qwen3・gemma3・lfm2 など、このアプリで動かせるもの）を確かめたもの）。
/// 型の推論を軽くするため、小さな関数で作って、分野ごとに分けてある。
extension RecommendedModels {
    static var more: [RecommendedModel] {
        moreGeneral + moreGeneral2 + japanese + moreCoding + moreReasoning + moreTiny
    }

    private static func entry(_ repo: String, _ title: String, _ bytes: Int64, _ japanese: RecommendedModel.JapaneseSupport,
                             _ category: RecommendedModel.Category, ja: String, en: String) -> RecommendedModel {
        // 2GB を超えるものは、メモリの少ない iPad では止まることがあるので「試験的」
        RecommendedModel(repoID: repo, title: title, approximateBytes: bytes, japanese: japanese,
                         isExperimental: bytes > 2_000_000_000, note: L(ja, en), category: category, isGGUF: true)
    }

    private static var moreGeneral: [RecommendedModel] {[
        entry("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5 0.8B (GGUF)", 532_517_120, .good, .general,
              ja: "Qwen の新しい世代。線形の注意（Gated DeltaNet）と普通の注意を組み合わせた構造で、長い会話でもメモリが増えにくい。今は CPU で動かします。",
              en: "Qwen's new generation. Mixes linear attention (Gated DeltaNet) with regular attention, so memory grows slowly in long chats. Runs on the CPU for now."),
        entry("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5 2B (GGUF)", 1_280_835_840, .good, .general,
              ja: "Qwen3.5 の 2B。0.8B より賢い。今は CPU で動かします。", en: "Qwen3.5 2B. Smarter than 0.8B. Runs on the CPU for now."),
        entry("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5 4B (GGUF)", 2_740_937_888, .good, .reasoning,
              ja: "Qwen3.5 の 4B。この大きさでとても賢い。メモリの多い iPad 向け。今は CPU で動かします。",
              en: "Qwen3.5 4B. Very capable for its size. For iPads with more memory. Runs on the CPU for now."),
        entry("unsloth/Qwen3-1.7B-GGUF", "Qwen3 1.7B (GGUF)", 1_107_409_472, .good, .general,
             ja: "思考モードを切り替えられる Qwen3。賢さと速さのバランスがよい。", en: "Qwen3 with switchable thinking. A good balance of smarts and speed."),
        entry("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3 4B Instruct 2507 (GGUF)", 2_497_281_120, .good, .general,
             ja: "Qwen3 4B の改良版（考えずにすぐ答える）。この大きさでかなり賢い。", en: "Improved Qwen3 4B that answers without thinking. Very capable for its size."),
        entry("Qwen/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5 3B Instruct (GGUF)", 2_104_932_768, .good, .general,
             ja: "日本語も自然な Qwen2.5 の 3B。", en: "Qwen2.5 3B with natural Japanese."),
        entry("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama 3.2 3B Instruct (GGUF)", 2_019_377_696, .weak, .general,
             ja: "Meta の 3B。英語が得意。", en: "Meta's 3B. Strong in English."),
        entry("unsloth/gemma-3-4b-it-GGUF", "Gemma 3 4B (GGUF)", 2_489_894_016, .good, .general,
             ja: "Google の多言語モデルの 4B。文章がうまい。", en: "Google's multilingual 4B. Writes well."),
        entry("unsloth/Phi-4-mini-instruct-GGUF", "Phi-4 mini Instruct (GGUF)", 2_491_874_272, .weak, .general,
             ja: "Microsoft の 3.8B。英語の質問や要約に強い。", en: "Microsoft's 3.8B. Strong at English questions and summaries."),
        entry("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5 mini Instruct (GGUF)", 2_393_232_672, .weak, .general,
             ja: "Microsoft の 3.8B（ひとつ前の版）。", en: "Microsoft's 3.8B (previous version)."),
        entry("bartowski/HuggingFaceTB_SmolLM3-3B-GGUF", "SmolLM3 3B (GGUF)", 1_915_305_792, .weak, .general,
             ja: "Hugging Face の 3B。思考モード付き。", en: "Hugging Face's 3B with a thinking mode."),
        entry("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2 1.7B Instruct (GGUF)", 1_055_609_824, .none, .general,
             ja: "軽くて速い英語のモデル。", en: "A light, fast English model."),
    ]}

    private static var moreGeneral2: [RecommendedModel] {[
        entry("LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "LFM2.5 1.2B Instruct (GGUF)", 730_895_168, .fair, .general,
             ja: "畳み込みと注意を組み合わせた Liquid AI の新しいモデル。とても速い。", en: "Liquid AI's new hybrid of convolutions and attention. Very fast."),
        entry("LiquidAI/LFM2-1.2B-GGUF", "LFM2 1.2B (GGUF)", 796_412_864, .fair, .general,
             ja: "LFM2 の 1.2B。速くてメモリも少なめ。", en: "LFM2 1.2B. Fast with little memory."),
        entry("LiquidAI/LFM2-2.6B-GGUF", "LFM2 2.6B (GGUF)", 1_671_852_256, .fair, .general,
             ja: "LFM2 の大きめの版。", en: "The larger LFM2."),
        entry("LiquidAI/LFM2-700M-GGUF", "LFM2 700M (GGUF)", 513_954_752, .fair, .general,
             ja: "LFM2 の小さめの版。", en: "A smaller LFM2."),
        entry("lmstudio-community/granite-3.3-2b-instruct-GGUF", "Granite 3.3 2B Instruct (GGUF)", 1_545_303_328, .weak, .general,
             ja: "IBM の 2B。指示どおりに動くのが得意。", en: "IBM's 2B. Good at following instructions."),
        entry("unsloth/granite-4.0-1b-GGUF", "Granite 4.0 1B (GGUF)", 1_023_645_792, .weak, .general,
             ja: "IBM の新しい 1B。ツールの呼び出しにも強い。", en: "IBM's new 1B. Good at tool calling."),
        entry("unsloth/granite-4.0-micro-GGUF", "Granite 4.0 Micro (GGUF)", 2_099_502_848, .weak, .general,
             ja: "IBM の 3B。", en: "IBM's 3B."),
        entry("LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF", "EXAONE 4.0 1.2B (GGUF)", 812_437_792, .weak, .general,
             ja: "LG の 1.2B。英語と韓国語が得意。", en: "LG's 1.2B. Strong in English and Korean."),
        entry("bartowski/EXAONE-3.5-2.4B-Instruct-GGUF", "EXAONE 3.5 2.4B (GGUF)", 1_644_918_816, .weak, .general,
             ja: "LG の 2.4B。", en: "LG's 2.4B."),
        entry("bartowski/internlm2_5-1_8b-chat-GGUF", "InternLM2.5 1.8B Chat (GGUF)", 1_172_364_448, .weak, .general,
             ja: "上海 AI Lab の 1.8B。", en: "Shanghai AI Lab's 1.8B."),
        entry("bartowski/Falcon3-1B-Instruct-GGUF", "Falcon3 1B Instruct (GGUF)", 1_057_044_640, .none, .general,
             ja: "TII の 1B。", en: "TII's 1B."),
        entry("bartowski/Falcon3-3B-Instruct-GGUF", "Falcon3 3B Instruct (GGUF)", 2_005_684_480, .none, .general,
             ja: "TII の 3B。", en: "TII's 3B."),
    ]}

    private static var japanese: [RecommendedModel] {[
        entry("mmnga/sarashina2.2-1b-instruct-v0.1-gguf", "Sarashina2.2 1B Instruct (GGUF)", 931_179_392, .good, .japanese,
             ja: "SB Intuitions の日本語モデル。1B でも日本語がとても自然。", en: "SB Intuitions' Japanese model. Very natural Japanese even at 1B."),
        entry("mmnga/sarashina2.2-3b-instruct-v0.1-gguf", "Sarashina2.2 3B Instruct (GGUF)", 2_066_390_112, .good, .japanese,
             ja: "Sarashina2.2 の 3B。日本語の質問に強い。", en: "Sarashina2.2 3B. Strong at Japanese questions."),
        entry("mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf", "Sarashina2.2 0.5B Instruct (GGUF)", 528_204_672, .good, .japanese,
             ja: "Sarashina2.2 の一番小さい版。", en: "The smallest Sarashina2.2."),
        entry("SakanaAI/TinySwallow-1.5B-Instruct-GGUF", "TinySwallow 1.5B Instruct (GGUF)", 1_646_573_920, .good, .japanese,
             ja: "Sakana AI が日本語に鍛えた小さなモデル。", en: "Sakana AI's small model trained for Japanese."),
        entry("mmnga/llm-jp-3.1-1.8b-instruct4-gguf", "LLM-jp-3.1 1.8B Instruct (GGUF)", 1_164_239_136, .good, .japanese,
             ja: "国立情報学研究所（LLM-jp）の日本語モデル。", en: "LLM-jp (NII)'s Japanese model."),
        entry("mmnga/RakutenAI-2.0-mini-instruct-gguf", "RakutenAI 2.0 mini Instruct (GGUF)", 935_981_984, .good, .japanese,
             ja: "楽天の日本語モデル（1.5B）。", en: "Rakuten's Japanese model (1.5B)."),
        entry("bartowski/gemma-2-2b-jpn-it-GGUF", "Gemma 2 2B JPN (GGUF)", 1_708_583_072, .good, .japanese,
             ja: "Google が日本語向けに調整した Gemma 2。", en: "Gemma 2 tuned by Google for Japanese."),
        entry("LiquidAI/LFM2.5-1.2B-JP-GGUF", "LFM2.5 1.2B JP (GGUF)", 730_895_296, .good, .japanese,
             ja: "LFM2.5 の日本語版。速くて日本語が自然。", en: "The Japanese LFM2.5. Fast with natural Japanese."),
    ]}

    private static var moreCoding: [RecommendedModel] {[
        entry("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "Qwen2.5 Coder 1.5B (GGUF)", 1_117_320_768, .fair, .coding,
             ja: "CDWS のおすすめ（GGUF 版）。GPU で速く動く。", en: "Recommended for CDWS (GGUF). Fast on the GPU."),
        entry("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "Qwen2.5 Coder 3B (GGUF)", 2_104_932_800, .fair, .coding,
             ja: "1.5B より賢いコーディングモデル。メモリの多い iPad 向け。", en: "Smarter than 1.5B. For iPads with more memory."),
        entry("bartowski/stable-code-instruct-3b-GGUF", "Stable Code Instruct 3B (GGUF)", 1_708_595_936, .none, .coding,
             ja: "Stability AI のコーディングモデル。", en: "Stability AI's coding model."),
        entry("bartowski/Yi-Coder-1.5B-Chat-GGUF", "Yi-Coder 1.5B Chat (GGUF)", 963_674_304, .weak, .coding,
             ja: "01.AI のコーディングモデル。多くの言語に対応。", en: "01.AI's coding model for many languages."),
        entry("lmstudio-community/OpenCoder-1.5B-Instruct-GGUF", "OpenCoder 1.5B Instruct (GGUF)", 1_417_775_200, .none, .coding,
             ja: "学習データまで公開されたコーディングモデル。", en: "A coding model with open training data."),
        entry("TheBloke/deepseek-coder-1.3b-instruct-GGUF", "DeepSeek Coder 1.3B Instruct (GGUF)", 873_582_624, .none, .coding,
             ja: "DeepSeek の小さなコーディングモデル。", en: "DeepSeek's small coding model."),
    ]}

    private static var moreReasoning: [RecommendedModel] {[
        entry("unsloth/Qwen3-4B-Thinking-2507-GGUF", "Qwen3 4B Thinking 2507 (GGUF)", 2_497_281_152, .good, .reasoning,
             ja: "必ず考えてから答える Qwen3 4B。この大きさで最も賢い部類。", en: "Qwen3 4B that always thinks first. Among the smartest at this size."),
        entry("unsloth/Qwen3-4B-GGUF", "Qwen3 4B (GGUF)", 2_497_281_312, .good, .reasoning,
             ja: "思考モードを切り替えられる Qwen3 4B。", en: "Qwen3 4B with switchable thinking."),
        entry("unsloth/Phi-4-mini-reasoning-GGUF", "Phi-4 mini Reasoning (GGUF)", 2_491_875_232, .weak, .reasoning,
             ja: "数学の推論に特化した Phi-4 mini。", en: "Phi-4 mini specialized in math reasoning."),
        entry("unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "DeepSeek R1 Distill Qwen 1.5B (GGUF)", 1_117_321_312, .weak, .reasoning,
             ja: "R1 の考え方を学んだ 1.5B（GGUF 版）。", en: "A 1.5B distilled from R1 (GGUF)."),
        entry("bartowski/agentica-org_DeepScaleR-1.5B-Preview-GGUF", "DeepScaleR 1.5B (GGUF)", 1_117_322_144, .weak, .reasoning,
             ja: "数学の問題に強くなるよう強化学習した 1.5B。", en: "A 1.5B trained with RL for math problems."),
        entry("bartowski/nvidia_OpenReasoning-Nemotron-1.5B-GGUF", "OpenReasoning Nemotron 1.5B (GGUF)", 986_046_784, .weak, .reasoning,
             ja: "NVIDIA の推論モデル。数学・コード・科学に強い。", en: "NVIDIA's reasoning model for math, code and science."),
        entry("mradermacher/VibeThinker-1.5B-GGUF", "VibeThinker 1.5B (GGUF)", 1_117_321_472, .weak, .reasoning,
             ja: "数学とプログラミングの推論に特化（GGUF 版）。", en: "Specialized in math and coding reasoning (GGUF)."),
        entry("bartowski/LGAI-EXAONE_EXAONE-Deep-2.4B-GGUF", "EXAONE Deep 2.4B (GGUF)", 1_497_463_168, .weak, .reasoning,
             ja: "LG の推論モデル。", en: "LG's reasoning model."),
        entry("bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF", "Qwen2.5 Math 1.5B (GGUF)", 986_048_832, .fair, .reasoning,
             ja: "数学の問題を解くのに特化した Qwen2.5。", en: "Qwen2.5 specialized in solving math problems."),
        entry("LiquidAI/LFM2.5-1.2B-Thinking-GGUF", "LFM2.5 1.2B Thinking (GGUF)", 730_899_072, .fair, .reasoning,
             ja: "考えてから答える LFM2.5。軽くて速い。", en: "LFM2.5 that thinks before answering. Light and fast."),
        entry("bartowski/Menlo_Jan-nano-GGUF", "Jan-nano 4B (GGUF)", 2_497_280_704, .fair, .reasoning,
             ja: "調べ物（Web 検索などのツール）を使いこなすよう鍛えた Qwen3 4B。", en: "A Qwen3 4B trained to use research tools like web search."),
    ]}

    private static var moreTiny: [RecommendedModel] {[
        entry("unsloth/gemma-3-270m-it-GGUF", "Gemma 3 270M (GGUF)", 253_115_424, .weak, .tiny,
             ja: "Google のとても小さいモデル。", en: "Google's very small model."),
        entry("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5 350M (GGUF)", 229_312_224, .weak, .tiny,
             ja: "とても小さく速い LFM2.5。", en: "A tiny, fast LFM2.5."),
        entry("LiquidAI/LFM2-350M-GGUF", "LFM2 350M (GGUF)", 254_958_528, .weak, .tiny,
             ja: "とても小さい LFM2。", en: "A tiny LFM2."),
        entry("bartowski/SmolLM2-360M-Instruct-GGUF", "SmolLM2 360M (GGUF)", 270_590_880, .none, .tiny,
             ja: "とても小さい英語のモデル（GGUF 版）。", en: "A tiny English model (GGUF)."),
        entry("openbmb/MiniCPM4-0.5B-QAT-Int4-GGUF", "MiniCPM4 0.5B (GGUF)", 265_307_040, .weak, .tiny,
             ja: "OpenBMB の小さなモデル（4 ビット前提で学習）。", en: "OpenBMB's small model, trained for 4-bit."),
        entry("unsloth/ERNIE-4.5-0.3B-PT-GGUF", "ERNIE 4.5 0.3B (GGUF)", 240_645_600, .weak, .tiny,
             ja: "Baidu の小さなモデル。", en: "Baidu's small model."),
        entry("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "TinyLlama 1.1B Chat (GGUF)", 668_788_096, .none, .tiny,
             ja: "昔からある小さな Llama。", en: "The classic small Llama."),
    ]}
}
