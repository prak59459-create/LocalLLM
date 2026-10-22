import SwiftData
import SwiftUI

/// 設定画面。型チェックを速くするため、セクションごとに別の View にしている。
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context
    @Environment(ChatController.self) private var chat

    @State private var confirmingDeleteChats = false

    var body: some View {
        NavigationStack {
            Form {
                GenerationSettingsSection()
                SystemPromptSection()
                ThinkingSettingsSection()
                ExecutionSettingsSection()
                DisplaySettingsSection()
                TokenSettingsSection()
                DeviceSettingsSection(onDeleteChats: { confirmingDeleteChats = true })
            }
            .navigationTitle(L("設定", "Settings"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("完了", "Done")) { dismiss() }
                }
            }
            .confirmationDialog(L("すべての会話を削除しますか？", "Delete all chats?"),
                                isPresented: $confirmingDeleteChats, titleVisibility: .visible) {
                Button(L("削除", "Delete"), role: .destructive, action: deleteAllChats)
            }
        }
    }

    private func deleteAllChats() {
        chat.stopGenerating()
        try? context.delete(model: StoredMessage.self)
        try? context.delete(model: Conversation.self)
        try? context.save()
        Task { await chat.engine.resetCache() }
    }
}

private struct GenerationSettingsSection: View {
    @Environment(AppSettings.self) private var settings
    @State private var seedText = ""

    var body: some View {
        @Bindable var settings = settings
        Section {
            LabeledSlider(title: L("温度（temperature）", "Temperature"),
                          value: $settings.generation.temperature, range: 0...2, step: 0.05,
                          hint: L("0 にすると毎回同じ答えに。上げるほど多様に。", "0 = deterministic; higher = more varied."))
            LabeledSlider(title: "top-p", value: $settings.generation.topP, range: 0...1, step: 0.01,
                          hint: L("確率の高い候補から合計 p までに絞ります（1 で無効）。", "Keeps the smallest set of tokens with total probability p (1 = off)."))
            Stepper(value: $settings.generation.topK, in: 0...200, step: 5) {
                LabeledContent("top-k", value: topKLabel)
            }
            LabeledSlider(title: "min-p", value: $settings.generation.minP, range: 0...0.5, step: 0.01,
                          hint: L("一番高い確率の p 倍未満の候補を除きます（0 で無効）。", "Drops tokens below p × the top probability (0 = off)."))
            LabeledSlider(title: L("繰り返しペナルティ", "Repetition penalty"),
                          value: $settings.generation.repetitionPenalty, range: 1...1.5, step: 0.01,
                          hint: L("同じ言葉の繰り返しを減らします（1 で無効）。", "Reduces repetition (1 = off)."))
            Stepper(value: $settings.generation.maxNewTokens, in: 32...8192, step: 32) {
                LabeledContent(L("最大生成トークン数", "Max new tokens"), value: "\(settings.generation.maxNewTokens)")
            }
            Toggle(L("シードを固定する", "Fixed seed"), isOn: fixedSeed)
            if settings.generation.seed != nil {
                TextField(L("シード", "Seed"), text: $seedText)
                    .keyboardType(.numberPad)
                    .onChange(of: seedText) { _, newValue in
                        if let value = UInt64(newValue) { settings.generation.seed = value }
                    }
            }
            Button(L("既定値に戻す", "Reset to defaults")) { settings.resetGeneration() }
        } header: {
            Text(L("生成", "Generation"))
        }
        .onAppear {
            if let seed = settings.generation.seed { seedText = String(seed) }
        }
    }

    private var topKLabel: String {
        settings.generation.topK == 0 ? L("無効", "Off") : "\(settings.generation.topK)"
    }

    private var fixedSeed: Binding<Bool> {
        Binding(get: { settings.generation.seed != nil },
                set: { settings.generation.seed = $0 ? (UInt64(seedText) ?? 42) : nil })
    }
}

private struct SystemPromptSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            TextEditor(text: $settings.systemPrompt)
                .frame(minHeight: 110)
            Button(L("既定のプロンプトに戻す", "Restore default prompt")) {
                settings.systemPrompt = AppSettings.defaultSystemPrompt
            }
        } header: {
            Text(L("システムプロンプト", "System prompt"))
        } footer: {
            Text(L("すべての会話の最初にモデルへ渡す指示です。空にすると送りません（テンプレートによっては既定の指示が入ります）。",
                   "Instructions sent at the start of every chat. Leave empty to send none (some templates add their own default)."))
        }
    }
}

private struct ThinkingSettingsSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            Picker(L("思考モード", "Thinking mode"), selection: $settings.thinkingMode) {
                ForEach(ThinkingMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
        } footer: {
            Text(L("Qwen3 など、考えてから答えるモデル向けの設定です（テンプレートの enable_thinking）。",
                   "For models that can think before answering, such as Qwen3 (template variable enable_thinking)."))
        }
    }
}

private struct ExecutionSettingsSection: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat

    var body: some View {
        @Bindable var settings = settings
        Section {
            Picker(L("計算ユニット", "Compute units"), selection: $settings.computeUnits) {
                ForEach(ComputeUnitsOption.allCases) { option in
                    Text(option.label).tag(option)
                }
            }
            Picker(L("プロンプトの分割サイズ", "Prefill chunk size"), selection: $settings.prefillChunkSize) {
                ForEach([32, 64, 128, 256, 512], id: \.self) { size in
                    Text("\(size)").tag(size)
                }
            }
            .onChange(of: settings.prefillChunkSize) { _, size in
                Task { await chat.engine.setPrefillChunkSize(size) }
            }
            Toggle(L("起動時に前回のモデルを読み込む", "Load last model at launch"), isOn: $settings.autoLoadLastModel)
            if let id = chat.loadedInfo?.modelID {
                Button(L("今のモデルを読み込み直す", "Reload current model")) {
                    chat.loadModel(id: id, library: library, settings: settings)
                }
            }
        } header: {
            Text(L("実行", "Execution"))
        } footer: {
            Text(L("計算ユニットの変更は次の読み込みから有効です。分割サイズを小さくするとメモリを節約できますが、読み込みは遅くなります。",
                   "Compute unit changes apply on the next load. A smaller chunk size saves memory but reads prompts more slowly."))
        }
    }
}

private struct DisplaySettingsSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            Toggle(L("速度とメモリを表示", "Show speed and memory"), isOn: $settings.showStats)
            Picker(L("言語", "Language"), selection: $settings.language) {
                Text(L("端末の設定に合わせる", "System")).tag(AppLanguage.system)
                Text("日本語").tag(AppLanguage.japanese)
                Text("English").tag(AppLanguage.english)
            }
        } header: {
            Text(L("表示", "Display"))
        }
    }
}

private struct TokenSettingsSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            SecureField("hf_...", text: $settings.huggingFaceToken)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        } header: {
            Text("Hugging Face")
        } footer: {
            Text(L("ゲート付き（利用規約への同意が必要）のモデルを使うときだけ必要です。トークンはキーチェーンに保存されます。",
                   "Only needed for gated models. The token is stored in the Keychain."))
        }
    }
}

private struct DeviceSettingsSection: View {
    @Environment(ModelLibrary.self) private var library
    var onDeleteChats: () -> Void

    var body: some View {
        Section {
            LabeledContent(L("端末のメモリ", "Device memory"), value: ByteFormat.string(DeviceMemory.physicalBytes))
            LabeledContent(L("アプリが使える上限（目安）", "App memory limit (approx.)"), value: ByteFormat.string(DeviceMemory.appLimitBytes))
            LabeledContent(L("保存したモデル", "Saved models"), value: ByteFormat.string(library.totalBytesOnDisk))
            if let free = library.availableStorageBytes {
                LabeledContent(L("空き容量", "Free storage"), value: ByteFormat.string(free))
            }
            Button(L("すべての会話を削除", "Delete all chats"), role: .destructive, action: onDeleteChats)
        } header: {
            Text(L("端末", "Device"))
        }
    }
}

/// 値の表示付きスライダー。
private struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    var hint: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: "%.2f", value))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: $value, in: range, step: step)
            if let hint {
                Text(hint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
