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

    var id: String { repoID }
    var repoID: String
    var title: String
    var approximateBytes: Int64
    var japanese: JapaneseSupport
    var isExperimental: Bool
    var note: String
}

enum RecommendedModels {
    // 言語の切り替えに追従するよう、毎回作り直す（computed）。
    static var all: [RecommendedModel] {[
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit",
            title: "Qwen2.5 0.5B Instruct (4bit)",
            approximateBytes: 286_000_000,
            japanese: .fair,
            isExperimental: false,
            note: L("最初の 1 つにおすすめ。小さくて速く、日本語もそこそこ話せます。",
                    "Best first pick. Small and fast, with usable Japanese.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Llama-3.2-1B-Instruct-4bit",
            title: "Llama 3.2 1B Instruct (4bit)",
            approximateBytes: 705_000_000,
            japanese: .weak,
            isExperimental: false,
            note: L("英語が得意。日本語はやや不自然なことがあります。", "Strong in English; Japanese can be awkward.")
        ),
        RecommendedModel(
            repoID: "XDGCC/coreml-Qwen3-0.6B",
            title: "Qwen3 0.6B (FP16)",
            approximateBytes: 1_205_000_000,
            japanese: .fair,
            isExperimental: false,
            note: L("思考モード（<think>）に対応。量子化していないためサイズは大きめです。",
                    "Supports thinking mode (<think>). Not quantized, so it is larger.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            title: "DeepSeek R1 Distill Qwen 1.5B (4bit)",
            approximateBytes: 1_008_000_000,
            japanese: .weak,
            isExperimental: false,
            note: L("考えてから答える推論モデル。考える過程も表示されます。",
                    "A reasoning model that thinks before answering; the thoughts are shown.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-Coder-1.5B-Instruct-4bit",
            title: "Qwen2.5 Coder 1.5B Instruct (4bit)",
            approximateBytes: 877_000_000,
            japanese: .fair,
            isExperimental: false,
            note: L("プログラミング向け。", "Tuned for programming.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-SmolLM2-360M-Instruct-4bit",
            title: "SmolLM2 360M Instruct (4bit)",
            approximateBytes: 207_000_000,
            japanese: .none,
            isExperimental: false,
            note: L("とても小さい英語専用モデル。動作確認に。", "A tiny English-only model, good for testing.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Qwen2.5-3B-Instruct-4bit",
            title: "Qwen2.5 3B Instruct (4bit)",
            approximateBytes: 1_938_000_000,
            japanese: .good,
            isExperimental: true,
            note: L("試験的。日本語は一番自然ですが、RAM 4〜6GB の iPad ではメモリ不足で落ちることがあります。",
                    "Experimental. Best Japanese, but may crash from low memory on 4–6 GB iPads.")
        ),
        RecommendedModel(
            repoID: "finnvoorhees/coreml-Llama-3.2-3B-Instruct-4bit",
            title: "Llama 3.2 3B Instruct (4bit)",
            approximateBytes: 1_818_000_000,
            japanese: .weak,
            isExperimental: true,
            note: L("試験的。RAM 4〜6GB の iPad ではメモリ不足で落ちることがあります。",
                    "Experimental. May crash from low memory on 4–6 GB iPads.")
        ),
    ]}
}
