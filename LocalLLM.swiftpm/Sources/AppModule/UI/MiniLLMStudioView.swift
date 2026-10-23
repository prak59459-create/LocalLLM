import Charts
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// ミニLLM工房: 小さな GPT をゼロから学習する画面。
struct MiniLLMStudioView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(MiniLLMStudio.self) private var studio

    var body: some View {
        NavigationStack {
            Form {
                StudioIntroSection()
                StudioDatasetSection()
                StudioModelSection()
                StudioTrainingSection()
                StudioSavedModelsSection()
            }
            .navigationTitle(L("ミニLLM工房", "Mini LLM Studio"))
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: UUID.self) { id in
                MiniModelPlaygroundView(modelID: id)
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close")) { dismiss() }
                }
            }
        }
    }
}

private struct StudioIntroSection: View {
    var body: some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                Label(L("LLM をゼロから作ってみよう", "Build an LLM from scratch"), systemImage: "hammer")
                    .font(.headline)
                Text(L("文章を 1 文字ずつ読んで「次の文字」を当てる小さな GPT を、この iPad の中だけで学習させます。大きなモデルほど賢くはなりませんが、学習が進むにつれて文体や書き方を真似していく様子を観察できます。自分のコードで学習すると、小さな「自分専用のコード補完 AI」になります。",
                       "Train a tiny GPT that reads text character by character and learns to predict the next one, entirely on this iPad. It won't be as smart as big models, but you can watch it pick up the style as it trains. Train it on your own code for a tiny personal code-completion model."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        }
    }
}

/// 1. 学習データ
private struct StudioDatasetSection: View {
    @Environment(MiniLLMStudio.self) private var studio
    @State private var importing = false
    @State private var message: String?
    @State private var editing = false

    var body: some View {
        Section {
            DatasetInfoRows()
            DatasetSourceMenu(message: $message, importing: $importing, editing: $editing)
            if studio.isDownloadingDataset {
                ProgressView()
            }
            if let message {
                Text(message).font(.caption).foregroundStyle(.secondary)
            }
        } header: {
            Text(L("1. 学習データ", "1. Training data"))
        } footer: {
            Text(L("最低でも数千文字、できれば数十万文字あると上達が分かりやすくなります。",
                   "At least a few thousand characters; hundreds of thousands make progress easier to see."))
        }
        .fileImporter(isPresented: $importing,
                      allowedContentTypes: [.folder, .text, .sourceCode],
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                Task { message = await studio.importDataset(from: urls) }
            }
        }
        .sheet(isPresented: $editing) {
            DatasetEditorSheet()
        }
    }
}

/// いまの学習データの情報（型チェックを速くするため分けている）。
private struct DatasetInfoRows: View {
    @Environment(MiniLLMStudio.self) private var studio

    private var preview: String {
        String(studio.datasetText.prefix(300)) + (studio.datasetCharacterCount > 300 ? "…" : "")
    }

    var body: some View {
        LabeledContent(L("データ", "Dataset"), value: studio.datasetName)
        LabeledContent(L("文字数", "Characters"), value: "\(studio.datasetCharacterCount)")
        LabeledContent(L("文字の種類", "Unique characters"), value: "\(studio.datasetUniqueCharacters)")
        Text(preview)
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)
            .lineLimit(6)
    }
}

/// 学習データの選び方のメニュー。
private struct DatasetSourceMenu: View {
    @Environment(MiniLLMStudio.self) private var studio
    @Binding var message: String?
    @Binding var importing: Bool
    @Binding var editing: Bool

    var body: some View {
        Menu {
            Button(L("吾輩は猫である（冒頭）", "Wagahai wa Neko de Aru (opening)")) {
                studio.useSample(SampleCorpora.wagahai, name: L("吾輩は猫である（冒頭）", "Wagahai wa Neko de Aru (opening)"))
            }
            Button(L("Swift のコード例", "Swift code samples")) {
                studio.useSample(SampleCorpora.swiftCode, name: L("Swift のコード例", "Swift code samples"))
            }
            Button(L("Tiny Shakespeare をダウンロード（1.1MB）", "Download Tiny Shakespeare (1.1 MB)"), action: downloadShakespeare)
            Button(L("クリップボードから貼り付け", "Paste from clipboard"), action: paste)
            Button(L("ファイル・フォルダから読み込む…", "Load from files or folders…")) { importing = true }
            Button(L("自分で書く・編集する…", "Write or edit…")) { editing = true }
        } label: {
            Label(L("データを選ぶ", "Choose data"), systemImage: "doc.text")
        }
        .disabled(studio.status == .training)
    }

    private func downloadShakespeare() {
        Task { message = await studio.downloadTinyShakespeare() }
    }

    private func paste() {
        if let text = UIPasteboard.general.string, !text.isEmpty {
            studio.useSample(text, name: L("貼り付けた文章", "Pasted text"))
        }
    }
}

private struct DatasetEditorSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(MiniLLMStudio.self) private var studio
    @State private var text = ""

    var body: some View {
        NavigationStack {
            TextEditor(text: $text)
                .font(.body.monospaced())
                .padding(.horizontal)
                .navigationTitle(L("学習データを編集", "Edit training data"))
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button(L("キャンセル", "Cancel")) { dismiss() }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button(L("使う", "Use")) {
                            studio.useSample(text, name: L("自分で書いた文章", "My text"))
                            dismiss()
                        }
                    }
                }
        }
        .onAppear {
            // 大きすぎるとエディタが重くなるので先頭だけ
            text = String(studio.datasetText.prefix(200_000))
        }
    }
}

/// 2. モデルの大きさ
private struct StudioModelSection: View {
    @Environment(MiniLLMStudio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        let locked = studio.status == .training || studio.status == .paused
        let config = studio.plannedConfig
        Section {
            Picker(L("大きさ", "Size"), selection: $studio.preset) {
                ForEach(MiniGPTPreset.allCases, id: \.self) { preset in
                    Text(Self.label(for: preset)).tag(preset)
                }
            }
            .pickerStyle(.segmented)
            LabeledContent(L("構成", "Layout"),
                           value: L("\(config.layerCount) 層・\(config.embeddingSize) 次元・\(config.headCount) ヘッド・文脈 \(config.contextLength) 文字",
                                    "\(config.layerCount) layers · \(config.embeddingSize) dims · \(config.headCount) heads · \(config.contextLength) context"))
            LabeledContent(L("パラメータ数", "Parameters"), value: Self.formatCount(config.parameterCount))
            LabeledContent(L("学習に使うメモリ（目安）", "Training memory (approx.)"), value: ByteFormat.string(studio.plannedMemoryBytes))
            Stepper(value: $studio.totalSteps, in: 100...50_000, step: 100) {
                LabeledContent(L("学習ステップ数", "Training steps"), value: "\(studio.totalSteps)")
            }
            Picker(L("学習率", "Learning rate"), selection: $studio.learningRate) {
                Text("1e-3").tag(Float(1e-3))
                Text("3e-3").tag(Float(3e-3))
                Text("6e-3").tag(Float(6e-3))
            }
            Picker(L("バッチサイズ", "Batch size"), selection: $studio.batchSize) {
                ForEach([4, 8, 16, 32], id: \.self) { size in
                    Text("\(size)").tag(size)
                }
            }
        } header: {
            Text(L("2. モデル", "2. Model"))
        } footer: {
            Text(locked
                 ? L("学習中・一時停止中は変更できません。変えるときは「最初から」を押してください。",
                     "Locked while training or paused. Tap \"Start over\" to change.")
                 : L("まずは「極小」で試すのがおすすめです。大きいほど上手になりますが、時間がかかります。",
                     "Try Tiny first. Larger models learn better but take longer."))
        }
        .disabled(locked)
    }

    static func label(for preset: MiniGPTPreset) -> String {
        switch preset {
        case .tiny: return L("極小", "Tiny")
        case .small: return L("小", "Small")
        case .medium: return L("中", "Medium")
        }
    }

    static func formatCount(_ count: Int) -> String {
        if count >= 1_000_000 { return String(format: "%.1fM", Double(count) / 1_000_000) }
        if count >= 1_000 { return String(format: "%.0fK", Double(count) / 1_000) }
        return "\(count)"
    }
}

/// 3. 学習
private struct StudioTrainingSection: View {
    @Environment(MiniLLMStudio.self) private var studio
    @State private var saveName = ""
    @State private var saving = false
    @State private var saveMessage: String?

    var body: some View {
        @Bindable var studio = studio
        Section {
            controls
            if studio.step > 0 || studio.status == .training {
                ProgressView(value: Double(studio.step), total: Double(max(studio.totalSteps, 1)))
                StudioStatsView()
                LossChartView(points: studio.lossPoints)
                    .frame(height: 180)
            }
            if case .failed(let message) = studio.status {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            TextField(L("文章の例の書き出し（空でも可）", "Sample prompt (optional)"), text: $studio.samplePrompt)
            if !studio.sampleText.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L("いまのモデルが書いた文章", "What the model writes now"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(studio.sampleText)
                        .font(.callout.monospaced())
                        .textSelection(.enabled)
                }
            }
            if studio.hasTrainedModel && studio.status != .training {
                HStack {
                    TextField(L("保存する名前", "Name to save as"), text: $saveName)
                    Button(L("保存", "Save")) {
                        saving = true
                        Task {
                            saveMessage = await studio.saveCurrentModel(name: saveName) ?? L("保存しました", "Saved")
                            saving = false
                        }
                    }
                    .disabled(saving)
                }
                if let saveMessage {
                    Text(saveMessage).font(.caption).foregroundStyle(.secondary)
                }
            }
        } header: {
            Text(L("3. 学習", "3. Training"))
        } footer: {
            Text(L("学習中は画面が消えないようにしています。アプリを閉じると学習は止まります（保存していないモデルは消えます）。",
                   "The screen stays on while training. Closing the app stops training (unsaved models are lost)."))
        }
    }

    @ViewBuilder
    private var controls: some View {
        HStack {
            switch studio.status {
            case .training:
                Button {
                    studio.pause()
                } label: {
                    Label(L("一時停止", "Pause"), systemImage: "pause.fill")
                }
                .buttonStyle(.bordered)
            case .paused:
                Button {
                    studio.startOrResume()
                } label: {
                    Label(L("続きから", "Resume"), systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
            default:
                Button {
                    studio.startOrResume()
                } label: {
                    Label(studio.status == .finished ? L("もう一度学習", "Train again") : L("学習を始める", "Start training"),
                          systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!studio.canStart)
            }
            Spacer()
            if studio.status == .paused || studio.status == .finished {
                Button(role: .destructive) {
                    studio.resetTraining()
                } label: {
                    Label(L("最初から", "Start over"), systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(.bordered)
            }
        }
    }
}

private struct StudioStatsView: View {
    @Environment(MiniLLMStudio.self) private var studio

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
            GridRow {
                Text(L("ステップ", "Step"))
                Text("\(studio.step) / \(studio.totalSteps)").monospacedDigit()
            }
            GridRow {
                Text(L("損失（学習 / 検証）", "Loss (train / val)"))
                Text("\(Self.format(studio.latestTrainLoss)) / \(Self.format(studio.latestValidationLoss))").monospacedDigit()
            }
            GridRow {
                Text(L("速さ", "Speed"))
                Text(String(format: L("%.0f 文字/秒", "%.0f chars/s"), studio.tokensPerSecond)).monospacedDigit()
            }
            if let remaining = studio.estimatedSecondsRemaining {
                GridRow {
                    Text(L("残り時間", "Time left"))
                    Text(DownloadStatusView.formatDuration(remaining))
                }
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    static func format(_ value: Float?) -> String {
        guard let value else { return "–" }
        return String(format: "%.3f", value)
    }
}

/// 損失のグラフ（Swift Charts）。
private struct LossChartView: View {
    let points: [LossPoint]

    var body: some View {
        Chart(points) { point in
            LineMark(
                x: .value(L("ステップ", "Step"), point.step),
                y: .value(L("損失", "Loss"), point.loss)
            )
            .foregroundStyle(by: .value(L("種類", "Kind"), point.isValidation ? L("検証", "Validation") : L("学習", "Train")))
        }
        .chartYAxisLabel {
            Text(L("損失（小さいほど良い）", "Loss (lower is better)"))
        }
    }
}

/// 4. 保存したモデル
private struct StudioSavedModelsSection: View {
    @Environment(MiniLLMStudio.self) private var studio

    var body: some View {
        Section(L("4. 保存したモデル", "4. Saved models")) {
            if studio.savedModels.isEmpty {
                Text(L("まだありません。学習したら名前を付けて保存してください。", "None yet. Save a model after training."))
                    .foregroundStyle(.secondary)
            }
            ForEach(studio.savedModels) { info in
                NavigationLink(value: info.id) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(info.name)
                        Text(L("\(StudioModelSection.label(for: info.preset))・\(info.trainedSteps) ステップ・\(info.datasetName)",
                               "\(StudioModelSection.label(for: info.preset)) · \(info.trainedSteps) steps · \(info.datasetName)"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .swipeActions {
                    Button(role: .destructive) {
                        studio.deleteSavedModel(info.id)
                    } label: {
                        Label(L("削除", "Delete"), systemImage: "trash")
                    }
                }
            }
        }
    }
}

/// 保存したミニモデルで文章の続きを書かせる画面。
struct MiniModelPlaygroundView: View {
    @Environment(MiniLLMStudio.self) private var studio
    let modelID: UUID

    @State private var loaded: LoadedMiniModel?
    @State private var errorMessage: String?
    @State private var prompt = ""
    @State private var output = ""
    @State private var temperature: Double = 0.8
    @State private var length: Int = 300
    @State private var generating = false
    @State private var task: Task<Void, Never>?

    var body: some View {
        Form {
            if let info = studio.savedModels.first(where: { $0.id == modelID }) {
                Section {
                    LabeledContent(L("名前", "Name"), value: info.name)
                    LabeledContent(L("学習データ", "Dataset"), value: info.datasetName)
                    LabeledContent(L("ステップ", "Steps"), value: "\(info.trainedSteps)")
                    LabeledContent(L("検証損失", "Validation loss"), value: StudioStatsView.format(info.validationLoss))
                }
            }
            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
            Section(L("続きを書かせる", "Let it continue")) {
                TextField(L("書き出し", "Beginning"), text: $prompt, axis: .vertical)
                    .lineLimit(1...4)
                HStack {
                    Text(L("温度", "Temperature"))
                    Slider(value: $temperature, in: 0...1.5, step: 0.05)
                    Text(String(format: "%.2f", temperature)).monospacedDigit()
                }
                Stepper(value: $length, in: 50...2000, step: 50) {
                    LabeledContent(L("文字数", "Length"), value: "\(length)")
                }
                Button {
                    generating ? stop() : generate()
                } label: {
                    Label(generating ? L("止める", "Stop") : L("書かせる", "Generate"),
                          systemImage: generating ? "stop.fill" : "sparkles")
                }
                .disabled(loaded == nil)
            }
            if !output.isEmpty {
                Section(L("出力", "Output")) {
                    Text(prompt + output)
                        .font(.callout.monospaced())
                        .textSelection(.enabled)
                }
            }
        }
        .navigationTitle(L("ミニモデルで生成", "Generate"))
        .navigationBarTitleDisplayMode(.inline)
        .task {
            do {
                loaded = try await studio.loadSavedModel(modelID)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        .onDisappear { stop() }
    }

    private func generate() {
        guard let loaded else { return }
        output = ""
        generating = true
        let prompt = self.prompt
        let length = self.length
        let temperature = Float(self.temperature)
        let seed = UInt64.random(in: 1...UInt64.max)
        // complete は actor に属さない async 関数なので、メインスレッドの外で動く（キャンセルもそのまま伝わる）
        let stream = AsyncStream<String> { continuation in
            let worker = Task {
                await loaded.complete(prompt: prompt, length: length, temperature: temperature, topK: 40, seed: seed) { chunk in
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in worker.cancel() }
        }
        task = Task {
            for await chunk in stream {
                output += chunk
            }
            generating = false
        }
    }

    private func stop() {
        task?.cancel()
        task = nil
        generating = false
    }
}
