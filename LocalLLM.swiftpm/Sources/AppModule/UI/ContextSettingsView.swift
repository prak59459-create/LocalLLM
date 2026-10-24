import SwiftUI

/// モデルのコンテキスト長を変える画面（インストール済みのモデルから開く）。
struct ContextSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    @Environment(AppSettings.self) private var settings

    let modelID: String

    @State private var analysis: ContextResizer.Analysis?
    @State private var trainedContext: Int?
    @State private var target = 8192
    @State private var loadError: String?
    @State private var message: String?
    @State private var isWorking = false

    private var model: InstalledModel? { library.model(withID: modelID) }

    var body: some View {
        NavigationStack {
            Form {
                if let model {
                    ContextSummarySection(model: model, analysis: analysis, trainedContext: trainedContext)
                    if let analysis {
                        ContextChoiceSection(model: model, analysis: analysis, trainedContext: trainedContext, target: $target)
                        actionSection(model: model)
                    } else if let loadError {
                        Section {
                            Label(loadError, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                        }
                    } else {
                        Section {
                            HStack {
                                ProgressView()
                                Text(L("モデルを調べています…", "Inspecting the model…")).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle(L("コンテキスト長", "Context Length"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close")) { dismiss() }
                        .disabled(isWorking)
                }
            }
            .interactiveDismissDisabled(isWorking)
            .task { await inspect() }
        }
    }

    @ViewBuilder
    private func actionSection(model: InstalledModel) -> some View {
        Section {
            Button {
                Task { await apply(target) }
            } label: {
                if isWorking {
                    HStack {
                        ProgressView()
                        Text(L("書き換えて確かめています…", "Rewriting and verifying…"))
                    }
                } else {
                    Label(L("\(target.formatted()) トークンにする", "Set to \(target.formatted()) tokens"), systemImage: "arrow.up.left.and.arrow.down.right")
                }
            }
            .disabled(isWorking || target == analysis?.context)
            if let original = model.originalContext {
                Button(role: .destructive) {
                    Task { await restore() }
                } label: {
                    Label(L("元の長さ（\(original.formatted())）に戻す", "Restore the original length (\(original.formatted()))"),
                          systemImage: "arrow.uturn.backward")
                }
                .disabled(isWorking)
            }
            if let message {
                Text(message).font(.callout)
            }
        } footer: {
            Text(L("重みはそのままで、モデルの中の「形」の情報だけを書き換えて読み込み直します（数秒〜数十秒）。書き換えたあと Core ML で読み込めるか確かめ、だめなら自動で元に戻します。",
                   "The weights stay as they are; only the shape information inside the model is rewritten (a few seconds to a minute). The model is test-loaded with Core ML afterwards and restored automatically if it fails."))
        }
    }

    private func inspect() async {
        do {
            let result = try await library.contextAnalysis(for: modelID)
            analysis = result
            if target <= result.context { target = min(result.context * 4, 131_072) }
            trainedContext = await Self.trainedContext(for: model, library: library, settings: settings)
        } catch {
            loadError = error.localizedDescription
        }
    }

    private func apply(_ newContext: Int) async {
        isWorking = true
        message = nil
        defer { isWorking = false }
        let wasLoaded = chat.loadedInfo?.modelID == modelID
        // 読み込み中のモデルは先に解放する（同じモデルを 2 つ読むとメモリが足りなくなる）
        if wasLoaded { await chat.unloadModelAndWait() }
        do {
            try await library.resizeContext(id: modelID, to: newContext)
            analysis = try await library.contextAnalysis(for: modelID)
            message = L("✅ \(newContext.formatted()) トークンになりました。", "✅ The context is now \(newContext.formatted()) tokens.")
        } catch {
            message = "⚠️ " + error.localizedDescription
        }
        if wasLoaded { chat.loadModel(id: modelID, library: library, settings: settings) }
    }

    private func restore() async {
        isWorking = true
        defer { isWorking = false }
        let wasLoaded = chat.loadedInfo?.modelID == modelID
        if wasLoaded { await chat.unloadModelAndWait() }
        do {
            try await library.restoreContext(id: modelID)
            analysis = try await library.contextAnalysis(for: modelID)
            message = L("元の長さに戻しました。", "Restored the original length.")
        } catch {
            message = "⚠️ " + error.localizedDescription
        }
        if wasLoaded { chat.loadModel(id: modelID, library: library, settings: settings) }
    }

    /// 学習したときの最大の長さ（config.json の max_position_embeddings。無ければ元のモデルのものを取ってくる）
    static func trainedContext(for model: InstalledModel?, library: ModelLibrary, settings: AppSettings) async -> Int? {
        guard let model else { return nil }
        if let path = model.supportFilePath(named: "config.json"),
           let data = try? Data(contentsOf: library.directory(for: model).appendingPathComponent(model.localPath(for: path))),
           let value = ModelConfigInfo.maxPositions(from: data) {
            return value
        }
        let client = settings.makeHFClient()
        guard let info = try? await client.modelInfo(repo: model.repoID), let base = info.baseModels.first,
              let data = try? await client.fetchData(repo: base, path: "config.json", revision: "main") else { return nil }
        return ModelConfigInfo.maxPositions(from: data)
    }
}

/// 今のコンテキスト長と、KV キャッシュの大きさ。
private struct ContextSummarySection: View {
    let model: InstalledModel
    let analysis: ContextResizer.Analysis?
    let trainedContext: Int?

    var body: some View {
        Section {
            LabeledContent(L("モデル", "Model"), value: model.displayName)
            if let analysis {
                LabeledContent(L("今のコンテキスト長", "Current context"), value: L("\(analysis.context.formatted()) トークン", "\(analysis.context.formatted()) tokens"))
                LabeledContent(L("KV キャッシュ", "KV cache"),
                               value: "\(ByteFormat.string(analysis.stateBytes))（\(ByteFormat.string(analysis.kvBytesPerToken))/\(L("トークン", "token"))）")
                LabeledContent(L("形式", "Format"), value: analysis.fixedShapes ? L("固定長（ANEMLL）", "Fixed length (ANEMLL)") : L("可変長", "Flexible"))
            }
            if let original = model.originalContext {
                LabeledContent(L("元の長さ", "Original"), value: original.formatted())
            }
            if let trainedContext {
                LabeledContent(L("学習時の最大の長さ", "Trained maximum"), value: trainedContext.formatted())
            }
        } footer: {
            Text(L("コンテキスト長は、モデルが一度に覚えておける会話の長さです。長くするほど KV キャッシュのメモリが増えます。",
                   "The context length is how much conversation the model can keep in mind at once. Longer contexts need more memory for the KV cache."))
        }
    }
}

/// 新しい長さを選ぶ。長さごとに必要なメモリと、この iPad に収まるかを表示する。
private struct ContextChoiceSection: View {
    let model: InstalledModel
    let analysis: ContextResizer.Analysis
    let trainedContext: Int?
    @Binding var target: Int

    private static let presets = [2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072]

    private var choices: [Int] {
        var values = Set(Self.presets)
        values.insert(analysis.context)
        if let original = model.originalContext { values.insert(original) }
        values.insert(target)
        return values.sorted()
    }

    var body: some View {
        Section {
            ForEach(choices, id: \.self) { value in
                ContextChoiceRow(value: value, isSelected: value == target, isCurrent: value == analysis.context,
                                 estimate: estimate(value))
                    .contentShape(Rectangle())
                    .onTapGesture { target = value }
            }
            Stepper(value: $target, in: 256...262_144, step: 1024) {
                LabeledContent(L("自分で決める", "Custom"), value: target.formatted())
            }
            warnings
        } header: {
            Text(L("新しい長さ", "New length"))
        }
    }

    private func estimate(_ context: Int) -> (bytes: Int64, fit: DeviceMemory.Fit) {
        let weights = model.weightBytes ?? model.totalBytes
        let required = MemoryEstimator.estimateBytes(weightBytes: weights, stateBytes: analysis.kvBytesPerToken * Int64(context),
                                                     vocabSize: nil, prefillChunk: 128)
        return (required, DeviceMemory.fit(requiredBytes: required))
    }

    @ViewBuilder
    private var warnings: some View {
        if let trainedContext, target > trainedContext {
            Label(L("学習時の長さ（\(trainedContext.formatted())）を超えると、遠くの内容を覚えきれず文章の質が落ちることがあります。",
                    "Beyond the trained length (\(trainedContext.formatted())), the model may lose track of distant text and quality can drop."),
                  systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
        }
        if analysis.fixedShapes {
            Label(L("固定長のモデルは、いつも最大の長さぶん計算します。長くするほど 1 トークンごとの速度が落ちます。",
                    "Fixed-length models always compute over the full length, so longer contexts make every token slower."),
                  systemImage: "tortoise")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        if estimate(target).fit == .tooLarge {
            Label(L("この iPad ではメモリが足りず、読み込みや長い会話の途中でアプリが終了されるおそれがあります。",
                    "This iPad probably doesn't have enough memory; the app may be terminated while loading or during long chats."),
                  systemImage: "memorychip")
                .font(.caption)
                .foregroundStyle(.red)
        }
    }
}

private struct ContextChoiceRow: View {
    let value: Int
    let isSelected: Bool
    let isCurrent: Bool
    let estimate: (bytes: Int64, fit: DeviceMemory.Fit)

    private var fitColor: Color {
        switch estimate.fit {
        case .comfortable: return .green
        case .tight: return .yellow
        case .tooLarge: return .red
        }
    }

    private var title: String {
        value >= 1024 && value % 1024 == 0 ? "\(value / 1024)k" : value.formatted()
    }

    var body: some View {
        HStack {
            Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(title).font(.headline.monospacedDigit())
                    if isCurrent { Badge(text: L("今", "Now"), color: .blue) }
                }
                Text(L("必要メモリの目安 \(ByteFormat.string(estimate.bytes))", "Estimated memory \(ByteFormat.string(estimate.bytes))"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(estimate.fit.label)
                .font(.caption)
                .foregroundStyle(fitColor)
        }
    }
}
