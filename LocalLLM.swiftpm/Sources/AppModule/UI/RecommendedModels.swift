import Foundation

/// おすすめモデル（このアプリの形式で動くことを確認した Hugging Face のリポジトリ）。
struct RecommendedModel: Identifiable, Hashable {
    enum JapaneseSupport: Int, Comparable {
        case none = 0, weak, fair, good

        static func < (lhs: JapaneseSupport, rhs: JapaneseSupport) -> Bool { lhs.rawValue < rhs.rawValue }

        var label: String {
            switch self {
            case .none: return L("日本語 ×", "Japanese ×")
            case .weak: return L("日本語 △", "Japanese △")
            case .fair: return L("日本語 ○", "Japanese ○")
            case .good: return L("日本語 ◎", "Japanese ◎")
            }
        }
    }

    /// 何が得意か（おすすめ一覧の見出し）
    enum Category: String, CaseIterable, Identifiable {
        case general
        case coding
        case reasoning
        case tiny

        var id: String { rawValue }

        var title: String {
            switch self {
            case .general: return L("はじめに・会話", "Start here · Chat")
            case .coding: return L("コーディング専用", "Coding")
            case .reasoning: return L("賢さ・推論専門", "Reasoning")
            case .tiny: return L("とても軽い（動作確認・古い iPad）", "Very light (testing · older iPads)")
            }
        }

        var subtitle: String {
            switch self {
            case .general: return L("普段の会話や質問に。小さくて速いものから。", "For everyday chat and questions. Small and fast first.")
            case .coding: return L("コードを書く・直すのが得意。コーディングワークスペース（CDWS）のエージェントに。",
                                   "Good at writing and fixing code. Use them for the Coding Workspace agent.")
            case .reasoning: return L("考えてから答える（<think>）。難しい質問・数学・論理に強いが、答えるまで時間がかかる。",
                                      "Think before answering (<think>). Stronger at hard questions, math and logic, but slower to answer.")
            case .tiny: return L("メモリの少ない iPad でも動く。賢さは控えめ。", "Run even on iPads with little memory. Not very smart.")
            }
        }

        var symbolName: String {
            switch self {
            case .general: return "bubble.left.and.bubble.right.fill"
            case .coding: return "chevron.left.forwardslash.chevron.right"
            case .reasoning: return "brain.head.profile"
            case .tiny: return "hare.fill"
            }
        }
    }

    var id: String { repoID }
    var repoID: String
    var title: String
    var approximateBytes: Int64
    var japanese: JapaneseSupport
    var isExperimental: Bool
    var note: String
    var category: Category = .general
    /// Neural Engine 向けの分割モデル（ANEMLL）
    var isSplitModel = false
}

enum RecommendedModels {
    static func models(in category: RecommendedModel.Category) -> [RecommendedModel] {
        all.filter { $0.category == category }
    }

    // 言語の切り替えに追従するよう、毎回作り直す（computed）。
    static var all: [RecommendedModel] { general + coding + reasoning + tiny }

    private static var general: [RecommendedModel] {[
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit", title: "Qwen2.5 0.5B Instruct (4bit)",
            approximateBytes: 286_000_000, japanese: .fair, isExperimental: false,
            note: L("最初の 1 つにおすすめ。小さくて速く、日本語もそこそこ話せます。", "Best first pick. Small and fast, with usable Japanese.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit", title: "Llama 3.2 1B Instruct (4bit)",
            approximateBytes: 705_000_000, japanese: .weak, isExperimental: false,
            note: L("英語が得意。日本語はやや不自然なことがあります。", "Strong in English; Japanese can be awkward.")
        ),
        RecommendedModel(
            repoID: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5", title: "Gemma 3 1B (ANE · 4096)",
            approximateBytes: 1_438_000_000, japanese: .fair, isExperimental: false,
            note: L("Google の多言語モデル。Neural Engine で動く分割モデルで、4096 トークンまで扱えます。",
                    "Google's multilingual model. A split model for the Neural Engine with a 4096-token context."),
            isSplitModel: true
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-3B-Instruct-4bit", title: "Qwen2.5 3B Instruct (4bit)",
            approximateBytes: 1_938_000_000, japanese: .good, isExperimental: true,
            note: L("試験的。日本語は一番自然ですが、RAM 4〜6GB の iPad ではメモリが足りず途中で止まることがあります。",
                    "Experimental. Best Japanese, but may stop for lack of memory on 4–6 GB iPads.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Llama-3.2-3B-Instruct-4bit", title: "Llama 3.2 3B Instruct (4bit)",
            approximateBytes: 1_818_000_000, japanese: .weak, isExperimental: true,
            note: L("試験的。RAM 4〜6GB の iPad ではメモリが足りず途中で止まることがあります。",
                    "Experimental. May stop for lack of memory on 4–6 GB iPads.")
        ),
    ]}

    private static var coding: [RecommendedModel] {[
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-Coder-0.5B-Instruct-4bit", title: "Qwen2.5 Coder 0.5B Instruct (4bit)",
            approximateBytes: 286_000_000, japanese: .fair, isExperimental: false,
            note: L("一番軽いコーディング専用モデル。ツールの呼び出しにも対応。メモリの少ない iPad の CDWS に。",
                    "The lightest coding model, with tool calling. For CDWS on iPads with little memory."),
            category: .coding
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-Coder-1.5B-Instruct-4bit", title: "Qwen2.5 Coder 1.5B Instruct (4bit)",
            approximateBytes: 877_000_000, japanese: .fair, isExperimental: false,
            note: L("CDWS のおすすめ。コードの読み書き・修正とツールの呼び出しが得意です。",
                    "Recommended for CDWS. Good at reading, writing and fixing code and at calling tools."),
            category: .coding
        ),
    ]}

    private static var reasoning: [RecommendedModel] {[
        RecommendedModel(
            repoID: "finnvoorhees/coreml-DeepSeek-R1-Distill-Qwen-1.5B-4bit", title: "DeepSeek R1 Distill Qwen 1.5B (4bit)",
            approximateBytes: 1_008_000_000, japanese: .weak, isExperimental: false,
            note: L("考えてから答える推論モデル。数学や論理が得意。考える過程も表示されます。",
                    "A reasoning model that thinks before answering. Good at math and logic; the thoughts are shown."),
            category: .reasoning
        ),
        RecommendedModel(
            repoID: "anemll/anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5", title: "Qwen3 1.7B (ANE · 2048)",
            approximateBytes: 1_959_000_000, japanese: .good, isExperimental: false,
            note: L("思考モードを切り替えられる Qwen3。賢さと日本語のバランスがよい、Neural Engine 向けの分割モデル。",
                    "Qwen3 with switchable thinking. Well balanced in smarts and Japanese; a split model for the Neural Engine."),
            category: .reasoning, isSplitModel: true
        ),
        RecommendedModel(
            repoID: "anemll/anemll-WeiboAI-VibeThinker-1.5B-ctx2048_0.3.5", title: "VibeThinker 1.5B (ANE · 2048)",
            approximateBytes: 1_686_000_000, japanese: .weak, isExperimental: false,
            note: L("数学とプログラミングの推論に特化した小さなモデル。じっくり考えるぶん、答えは遅め。",
                    "A small model specialized in math and coding reasoning. Thinks carefully, so answers take longer."),
            category: .reasoning, isSplitModel: true
        ),
        RecommendedModel(
            repoID: "XDGCC/coreml-Qwen3-0.6B", title: "Qwen3 0.6B (FP16)",
            approximateBytes: 1_205_000_000, japanese: .fair, isExperimental: false,
            note: L("思考モード（<think>）に対応した小さな Qwen3。量子化していないためサイズは大きめです。",
                    "A small Qwen3 with thinking mode (<think>). Not quantized, so it is larger."),
            category: .reasoning
        ),
    ]}

    private static var tiny: [RecommendedModel] {[
        RecommendedModel(
            repoID: "finnvoorhees/coreml-SmolLM2-360M-Instruct-4bit", title: "SmolLM2 360M Instruct (4bit)",
            approximateBytes: 207_000_000, japanese: .none, isExperimental: false,
            note: L("とても小さい英語専用モデル。動作確認に。", "A tiny English-only model, good for testing."),
            category: .tiny
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-SmolLM2-135M-Instruct-4bit", title: "SmolLM2 135M Instruct (4bit)",
            approximateBytes: 79_000_000, japanese: .none, isExperimental: false,
            note: L("一番小さいモデル。どの iPad でもすぐ動きます。", "The smallest model. Runs instantly on any iPad."),
            category: .tiny
        ),
    ]}
}
