import SwiftUI

/// リポジトリの詳細: 含まれるモデルの候補、対応状況、メモリの見積もり、ダウンロード。
struct ModelDetailView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat

    let repoID: String
    var onUse: () -> Void

    @State private var info: HFModelInfo?
    @State private var analysis: RepoAnalysis?
    /// variant.path → metadata.json の解析結果（.mlmodelc のみ。ダウンロード前に取得できる）
    @State private var metadata: [String: CoreMLMetadata] = [:]
    @State private var isLoading = true
    @State private var loadError: String?
    @State private var actionError: String?

    // 型チェックを速くするため、状態ごとの中身を分けている
    var body: some View {
        List {
            header
            content
        }
        .navigationTitle(repoID.split(separator: "/").last.map(String.init) ?? repoID)
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .refreshable { await load() }
    }

    @ViewBuilder
    private var content: some View {
        if isLoading {
            HStack {
                ProgressView()
                Text(L("リポジトリを調べています…", "Inspecting the repository…"))
                    .foregroundStyle(.secondary)
            }
        } else if let loadError {
            Label(loadError, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
            Button(L("もう一度", "Retry")) { Task { await load() } }
        } else if let analysis {
            warnings(analysis)
            VariantsSection(repoID: repoID, analysis: analysis, metadata: metadata, onUse: onUse,
                            onError: { actionError = $0 })
            if let actionError {
                Section {
                    Label(actionError, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
            }
            SupportFilesSection(files: analysis.supportFiles)
        }
    }

    @ViewBuilder
    private var header: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text(repoID)
                    .font(.headline)
                    .textSelection(.enabled)
                if let info {
                    HStack(spacing: 8) {
                        if let tag = info.pipelineTag { Badge(text: tag) }
                        if let library = info.libraryName { Badge(text: library, color: .purple) }
                        if info.gated { Badge(text: L("要トークン", "Gated"), color: .orange) }
                    }
                    if !info.baseModels.isEmpty {
                        Text(L("元のモデル: ", "Base model: ") + info.baseModels.joined(separator: ", "))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let url = URL(string: "https://huggingface.co/\(repoID)") {
                    Link(L("Hugging Face で開く", "Open on Hugging Face"), destination: url)
                        .font(.footnote)
                }
            }
        }
    }

    @ViewBuilder
    private func warnings(_ analysis: RepoAnalysis) -> some View {
        let needsToken = (info?.gated ?? false) && settings.huggingFaceToken.isEmpty
        if !analysis.hasTokenizer || needsToken {
            Section {
                if !analysis.hasTokenizer {
                    Label(L("tokenizer.json が含まれていません。このままでは使えません。",
                            "tokenizer.json is not included, so this repository cannot be used as is."),
                          systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
                if needsToken {
                    Label(L("このモデルは利用規約への同意が必要です。設定で Hugging Face のトークンを入力してください。",
                            "This model is gated. Enter your Hugging Face token in Settings."),
                          systemImage: "lock")
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    private func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        let client = settings.makeHFClient()
        do {
            async let infoTask = client.modelInfo(repo: repoID)
            async let filesTask = client.listFiles(repo: repoID, revision: "main")
            let (info, files) = try await (infoTask, filesTask)
            // 分割モデルの説明（meta.yaml）は小さいので先に取ってくる
            var metaFiles: [String: String] = [:]
            for path in RepoAnalysis.metaFilePaths(in: files).prefix(8) {
                if let data = try? await client.fetchData(repo: repoID, path: path, revision: "main"),
                   let text = String(data: data, encoding: .utf8) {
                    metaFiles[path] = text
                }
            }
            let repoName = repoID.split(separator: "/").last.map(String.init) ?? "model"
            let analysis = RepoAnalysis.analyze(files: files, metaFiles: metaFiles, repoName: repoName)
            self.info = info
            self.analysis = analysis
            // .mlmodelc の metadata.json は小さいので、先に取得して対応形式か調べる
            var found: [String: CoreMLMetadata] = [:]
            for variant in analysis.variants.prefix(8) where variant.kind == .compiled {
                let path = variant.path + "/metadata.json"
                guard variant.files.contains(where: { $0.path == path }) else { continue }
                if let data = try? await client.fetchData(repo: repoID, path: path, revision: "main"),
                   let parsed = try? CoreMLMetadata.parse(data) {
                    found[variant.path] = parsed
                }
            }
            metadata = found
        } catch {
            loadError = error.localizedDescription
        }
    }
}

/// モデルの候補の一覧。
private struct VariantsSection: View {
    let repoID: String
    let analysis: RepoAnalysis
    let metadata: [String: CoreMLMetadata]
    var onUse: () -> Void
    var onError: (String) -> Void

    var body: some View {
        if analysis.variants.isEmpty {
            Text(L("このリポジトリには Core ML のモデル（.mlmodelc / .mlpackage / 分割モデル / ZIP）が見つかりませんでした。",
                   "No Core ML models (.mlmodelc / .mlpackage / split models / ZIP) were found in this repository."))
                .foregroundStyle(.secondary)
        } else {
            Section(L("モデル", "Models")) {
                ForEach(analysis.variants) { variant in
                    VariantRow(repoID: repoID, variant: variant, analysis: analysis,
                               metadata: metadata[variant.path], onUse: onUse, onError: onError)
                }
            }
        }
    }
}

/// 付属ファイル（tokenizer.json など）の一覧。
private struct SupportFilesSection: View {
    let files: [HFFileEntry]

    var body: some View {
        Section(L("付属ファイル", "Support files")) {
            ForEach(files, id: \.path) { file in
                HStack {
                    Text(file.path).font(.caption.monospaced())
                    Spacer()
                    Text(ByteFormat.string(file.size)).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// コンテキスト長（一度に覚えておける長さ）と、あとから広げられることの案内。
private struct ContextInfoLine: View {
    let context: Int?
    let stateBytes: Int64?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let context {
                HStack(spacing: 6) {
                    Image(systemName: "text.alignleft")
                    Text(L("コンテキスト長 \(context.formatted()) トークン", "Context length \(context.formatted()) tokens"))
                        .fontWeight(.semibold)
                    if let stateBytes, stateBytes > 0 {
                        Text(L("（KV キャッシュ \(ByteFormat.string(stateBytes))）", "(KV cache \(ByteFormat.string(stateBytes)))"))
                            .foregroundStyle(.secondary)
                    }
                }
                Text(L("ダウンロード後に「コンテキスト長…」から、最大 128k トークンまで広げられます（メモリが足りる範囲で）。",
                       "After downloading, you can extend it up to 128k tokens from \"Context…\" (as far as memory allows)."))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Label(L("コンテキスト長はダウンロード後に表示します", "The context length is shown after download"), systemImage: "text.alignleft")
                    .foregroundStyle(.secondary)
            }
        }
        .font(.caption)
    }
}

/// 分割モデルの部品の一覧。
private struct PipelinePartsView: View {
    let pipeline: PipelineDescriptor

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(pipeline.parts.enumerated()), id: \.offset) { _, part in
                Label((part as NSString).lastPathComponent, systemImage: icon(for: part))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Text(details)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var details: String {
        var items: [String] = []
        if let batch = pipeline.batchSize { items.append(L("バッチ \(batch)", "batch \(batch)")) }
        if let window = pipeline.slidingWindow { items.append(L("スライディング窓 \(window)", "sliding window \(window)")) }
        if pipeline.argmaxInModel { items.append(L("モデル内で argmax", "argmax in model")) }
        if let architecture = pipeline.architecture { items.append(architecture) }
        return items.joined(separator: " · ")
    }

    private func icon(for part: String) -> String {
        if part == pipeline.embeddings { return "textformat.abc" }
        if part == pipeline.lmHead { return "text.cursor" }
        return "square.stack.3d.up"
    }
}

/// モデルの候補 1 つ分の行。
private struct VariantRow: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat

    let repoID: String
    let variant: ModelVariant
    let analysis: RepoAnalysis
    let metadata: CoreMLMetadata?
    var onUse: () -> Void
    var onError: (String) -> Void

    private var installedID: String { InstalledModel.makeID(repoID: repoID, variantPath: variant.path) }
    private var installed: InstalledModel? { library.model(withID: installedID) }

    private var weightBytes: Int64 {
        let weights = variant.files.filter { $0.path.hasSuffix(".bin") }.reduce(Int64(0)) { $0 + $1.size }
        return weights > 0 ? weights : variant.totalBytes
    }

    private var downloadBytes: Int64 {
        analysis.filesToDownload(for: variant).reduce(Int64(0)) { $0 + $1.size }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(variant.displayName).font(.headline)
                Spacer()
                Text(ByteFormat.string(downloadBytes))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(kindDescription)
                .font(.caption)
                .foregroundStyle(.secondary)
            if let pipeline = variant.pipeline {
                PipelinePartsView(pipeline: pipeline)
            }

            compatibility
            ContextInfoLine(context: metadata?.interface.maxContext ?? variant.pipeline?.contextLength,
                            stateBytes: metadata?.stateBytes)
            memoryEstimate
            actions
        }
        .padding(.vertical, 4)
    }

    private var kindDescription: String {
        switch variant.kind {
        case .compiled:
            return L("コンパイル済み（.mlmodelc）", "Compiled (.mlmodelc)")
        case .package:
            return L(".mlpackage（初回の読み込み時に端末でコンパイルします）", ".mlpackage (compiled on device at first load)")
        case .pipeline:
            return L("分割モデル（埋め込み・本体のチャンク・出力層をつないで動かします。Neural Engine 向け）",
                     "Split model (embeddings, chunks and LM head run in sequence; made for the Neural Engine)")
        case .archive:
            return L("ZIP（ダウンロード後に端末で展開して、中のモデルを調べます）",
                     "ZIP (expanded on device after download, then inspected)")
        }
    }

    /// ZIP は中に tokenizer.json が入っていることがあるので、付属ファイルが無くてもダウンロードできる
    private var canDownload: Bool {
        if variant.kind == .archive { return true }
        return analysis.hasTokenizer && metadata?.interface.isSupported != false
    }

    @ViewBuilder
    private var compatibility: some View {
        if let pipeline = variant.pipeline {
            Label(pipeline.contextLength.map { L("対応: 分割モデル・最大 \($0) トークン", "Supported: split model · up to \($0) tokens") }
                  ?? L("対応: 分割モデル", "Supported: split model"),
                  systemImage: "checkmark.seal.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else if let metadata {
            switch metadata.interface {
            case .statefulCausalLM(_, _, _, _, let maxContext):
                Label(L("対応: ステートフル・最大 \(maxContext) トークン", "Supported: stateful · up to \(maxContext) tokens"),
                      systemImage: "checkmark.seal.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            case .statelessCausalLM(_, _, _, _, let maxContext):
                Label(L("対応（低速）: ステートレス・最大 \(maxContext) トークン", "Supported (slow): stateless · up to \(maxContext) tokens"),
                      systemImage: "checkmark.seal")
                    .font(.caption)
                    .foregroundStyle(.yellow)
            case .unsupported(let reason):
                Label(L("非対応の形式: \(reason)", "Unsupported format: \(reason)"), systemImage: "xmark.octagon")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            if let iOS = metadata.availabilityIOS {
                Text(L("必要な iOS: \(iOS) 以降", "Requires iOS \(iOS)+"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        } else {
            Label(L("形式はダウンロード後に確認します", "Format will be checked after download"), systemImage: "questionmark.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var memoryEstimate: some View {
        let required = MemoryEstimator.estimateBytes(weightBytes: weightBytes,
                                                     stateBytes: metadata?.stateBytes ?? 0,
                                                     vocabSize: nil,
                                                     prefillChunk: settings.prefillChunkSize)
        let fit = DeviceMemory.fit(requiredBytes: required)
        HStack(spacing: 6) {
            Image(systemName: "memorychip")
            Text(L("必要メモリの目安 \(ByteFormat.string(required))", "Estimated memory \(ByteFormat.string(required))"))
            Text("·")
            Text(fit.label)
                .foregroundStyle(Self.color(for: fit))
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private static func color(for fit: DeviceMemory.Fit) -> Color {
        switch fit {
        case .comfortable: return .green
        case .tight: return .yellow
        case .tooLarge: return .red
        }
    }

    @ViewBuilder
    private var actions: some View {
        if let installed {
            if installed.isComplete {
                HStack {
                    Button {
                        chat.loadModel(id: installed.id, library: library, settings: settings)
                        onUse()
                    } label: {
                        Label(chat.loadedInfo?.modelID == installed.id ? L("使用中", "In use") : L("使う", "Use"),
                              systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(chat.loadedInfo?.modelID == installed.id)
                    Spacer()
                    Label(L("インストール済み", "Installed"), systemImage: "checkmark.circle")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            } else {
                if let state = library.downloads[installed.id] {
                    DownloadStatusView(state: state)
                }
                HStack {
                    DownloadControls(modelID: installed.id)
                    Spacer()
                    Button(role: .destructive) {
                        library.delete(id: installed.id)
                    } label: {
                        Label(L("中止して削除", "Cancel & Delete"), systemImage: "xmark")
                    }
                    .buttonStyle(.bordered)
                }
            }
        } else {
            Button {
                do {
                    try library.startDownload(repoID: repoID, variant: variant, analysis: analysis,
                                              metadata: metadata, client: settings.makeHFClient())
                } catch {
                    onError(error.localizedDescription)
                }
            } label: {
                Label(L("ダウンロード", "Download"), systemImage: "arrow.down.circle.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canDownload)
        }
    }
}
