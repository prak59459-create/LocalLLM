import Foundation
import SwiftData

/// マイアシスタント: 使うモデル・システムプロンプト・生成設定・参考資料をひとまとめにしたもの。
/// 例: 自分のコードを参考資料に入れた「自分専用コーディング AI」。
@Model
final class Assistant {
    var id: UUID = UUID()
    var name: String = ""
    /// SF Symbols の名前
    var symbolName: String = "person.crop.circle"
    var systemPrompt: String = ""
    /// このアシスタントで使いたいモデル（InstalledModel.id）。nil なら今読み込んでいるモデル
    var preferredModelID: String?
    /// true なら generationData の生成設定を使う（false なら全体の設定）
    var useCustomGeneration: Bool = false
    var generationData: Data?
    var thinkingModeRaw: String = ThinkingMode.modelDefault.rawValue
    /// 参考資料を検索してプロンプトに入れるか
    var useKnowledge: Bool = true
    /// 1 回に入れる参考資料の数
    var knowledgeResultCount: Int = 4
    var templateID: String?
    var createdAt: Date = Date()
    var updatedAt: Date = Date()

    init(name: String, symbolName: String, systemPrompt: String, templateID: String? = nil) {
        self.id = UUID()
        self.name = name
        self.symbolName = symbolName
        self.systemPrompt = systemPrompt
        self.templateID = templateID
        self.createdAt = Date()
        self.updatedAt = Date()
    }

    var generation: GenerationParameters {
        get { generationData.flatMap { try? JSONDecoder().decode(GenerationParameters.self, from: $0) } ?? .default }
        set { generationData = try? JSONEncoder().encode(newValue) }
    }

    var thinkingMode: ThinkingMode {
        get { ThinkingMode(rawValue: thinkingModeRaw) ?? .modelDefault }
        set { thinkingModeRaw = newValue.rawValue }
    }
}

/// アシスタントのひな形。
struct AssistantTemplate: Identifiable, Hashable {
    var id: String
    var name: String
    var symbolName: String
    var summary: String
    var systemPrompt: String
    var generation: GenerationParameters?
    /// おすすめのモデル（Hugging Face のリポジトリ ID の一部）
    var recommendedModelHint: String?

    /// 言語の切り替えに追従するよう、毎回作る。
    static var all: [AssistantTemplate] {
        var coding = GenerationParameters.default
        coding.temperature = 0.2
        coding.topP = 0.9
        coding.repetitionPenalty = 1.05
        coding.maxNewTokens = 2048

        var precise = GenerationParameters.default
        precise.temperature = 0.2
        precise.repetitionPenalty = 1.05

        var chatty = GenerationParameters.default
        chatty.temperature = 0.8

        return [
            AssistantTemplate(
                id: "coding",
                name: L("コーディング先生", "Coding Tutor"),
                symbolName: "chevron.left.forwardslash.chevron.right",
                summary: L("Swift / SwiftUI に詳しいプログラミングの先生。自分のコードを参考資料に入れると、それに合わせて答えます。",
                           "A programming tutor for Swift / SwiftUI. Add your own code as reference and it answers in your style."),
                systemPrompt: L("""
                あなたは Swift と SwiftUI、iPad の Swift Playgrounds に詳しいプログラミングの先生です。
                - 質問された言語で、初心者にも分かるように説明してください。
                - コードは ```swift のコードブロックで、そのまま動く完全な形で示してください。
                - 参考資料として利用者のコードが渡された場合は、その名前や書き方に合わせて答えてください。
                - 分からないことは推測せず、分からないと答えてください。
                """, """
                You are a programming tutor who knows Swift, SwiftUI and Swift Playgrounds on iPad well.
                - Answer in the language of the question, clearly enough for beginners.
                - Show code in ```swift code blocks, complete and ready to run.
                - When the user's own code is provided as reference, follow its names and style.
                - If you don't know something, say so instead of guessing.
                """),
                generation: coding,
                recommendedModelHint: "Coder"
            ),
            AssistantTemplate(
                id: "translator",
                name: L("翻訳", "Translator"),
                symbolName: "character.bubble",
                summary: L("日本語と英語を自然に翻訳します。", "Translates naturally between Japanese and English."),
                systemPrompt: L("""
                あなたはプロの翻訳者です。入力が日本語なら自然な英語に、それ以外なら自然な日本語に翻訳してください。
                説明は付けず、訳文だけを出力してください。
                """, """
                You are a professional translator. Translate Japanese input into natural English, and anything else into natural Japanese.
                Output only the translation, without explanations.
                """),
                generation: precise,
                recommendedModelHint: "Qwen2.5"
            ),
            AssistantTemplate(
                id: "summarizer",
                name: L("要約", "Summarizer"),
                symbolName: "text.badge.checkmark",
                summary: L("長い文章を短くまとめます。参考資料に入れた文書の要点も答えられます。",
                           "Summarizes long text, including the documents you add as reference."),
                systemPrompt: L("""
                あなたは要約の専門家です。渡された文章の要点を、箇条書き 3〜5 個で簡潔にまとめてください。
                事実だけを書き、文章にないことは付け加えないでください。
                """, """
                You are an expert summarizer. Summarize the key points of the given text in 3–5 concise bullet points.
                Stick to the facts in the text and do not add anything else.
                """),
                generation: precise,
                recommendedModelHint: nil
            ),
            AssistantTemplate(
                id: "english",
                name: L("英会話の相手", "English Partner"),
                symbolName: "person.2.wave.2",
                summary: L("やさしい英語で会話の練習相手になります。間違いもそっと直します。",
                           "Practices conversation in simple English and gently corrects mistakes."),
                systemPrompt: """
                You are a friendly English conversation partner for a Japanese learner.
                Reply in simple, natural English (2–4 sentences) and end with a question to keep the conversation going.
                If the user's English has a mistake, first show the corrected sentence briefly, then reply.
                """,
                generation: chatty,
                recommendedModelHint: nil
            ),
            AssistantTemplate(
                id: "blank",
                name: L("空のアシスタント", "Blank Assistant"),
                symbolName: "person.crop.circle",
                summary: L("自分で一から設定します。", "Configure everything yourself."),
                systemPrompt: "",
                generation: nil,
                recommendedModelHint: nil
            ),
        ]
    }

    /// ひな形からアシスタントを作る。
    func makeAssistant() -> Assistant {
        let assistant = Assistant(name: name, symbolName: symbolName, systemPrompt: systemPrompt, templateID: id)
        if let generation {
            assistant.useCustomGeneration = true
            assistant.generation = generation
        }
        return assistant
    }
}
