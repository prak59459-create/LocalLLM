import SwiftUI

/// モデル画面（おすすめ / 検索 / インストール済み）。シートで表示する。
struct ModelsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ModelLibrary.self) private var library

    enum Tab: String, CaseIterable, Identifiable {
        case recommended, search, installed
        var id: String { rawValue }
        var label: String {
            switch self {
            case .recommended: return L("おすすめ", "Featured")
            case .search: return L("検索", "Search")
            case .installed: return L("インストール済み", "Installed")
            }
        }
    }

    @State private var tab: Tab = .recommended
    @State private var path: [String] = []

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 0) {
                Picker(L("表示", "View"), selection: $tab) {
                    ForEach(Tab.allCases) { tab in
                        Text(tab.label).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.vertical, 8)

                switch tab {
                case .recommended: RecommendedModelsList()
                case .search: ModelSearchView()
                case .installed: InstalledModelsList(onDone: { dismiss() })
                }
            }
            .navigationTitle(L("モデル", "Models"))
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: String.self) { repoID in
                ModelDetailView(repoID: repoID, onUse: { dismiss() })
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close")) { dismiss() }
                }
            }
        }
        .onAppear {
            if !library.models.isEmpty && library.completedModels.count == library.models.count {
                tab = .installed
            }
        }
    }
}

/// おすすめ一覧。
private struct RecommendedModelsList: View {
    @Environment(ModelLibrary.self) private var library

    var body: some View {
        List {
            Section {
                ForEach(RecommendedModels.all) { model in
                    NavigationLink(value: model.repoID) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(model.title).font(.headline)
                                if model.isExperimental {
                                    Badge(text: L("試験的", "Experimental"), color: .orange)
                                }
                                if library.models.contains(where: { $0.repoID == model.repoID && $0.isComplete }) {
                                    Badge(text: L("入手済み", "Installed"), color: .green)
                                }
                            }
                            HStack(spacing: 10) {
                                Text(ByteFormat.string(model.approximateBytes))
                                Text(model.japanese.label)
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            Text(model.note)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
            } footer: {
                Text(L("どれも Hugging Face で公開されている Core ML 形式（iOS 18 のステートフルモデル）です。RAM が少ない iPad では小さいモデルから試してください。",
                       "All are Core ML models (iOS 18 stateful format) published on Hugging Face. On iPads with less RAM, start with a small one."))
            }
        }
    }
}

/// Hugging Face の検索。1 ページ 50 件で、下の「さらに表示」で続きを読み込む。
private struct ModelSearchView: View {
    @Environment(AppSettings.self) private var settings

    @State private var query = ""
    @State private var source: HFSearchSource = .coreML
    @State private var sort: HFSearchSort = .downloads
    @State private var textGenerationOnly = true
    @State private var results: [HFModelSummary] = []
    @State private var nextPage: URL?
    @State private var isSearching = false
    @State private var isLoadingMore = false
    @State private var errorMessage: String?
    @State private var directRepo = ""

    var body: some View {
        List {
            SearchFormSection(query: $query, source: $source, sort: $sort, textGenerationOnly: $textGenerationOnly,
                              isSearching: isSearching, onSearch: startSearch)
            DirectRepoSection(directRepo: $directRepo)

            if let errorMessage {
                Section {
                    Label(errorMessage, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
            }

            Section {
                ForEach(results) { model in
                    NavigationLink(value: model.id) {
                        SearchResultRow(model: model)
                    }
                }
                if nextPage != nil {
                    loadMoreRow
                }
            } header: {
                if !results.isEmpty {
                    Text(nextPage == nil ? L("\(results.count) 件", "\(results.count) results")
                                         : L("\(results.count) 件（続きあり）", "\(results.count) results (more available)"))
                }
            }
        }
        .task {
            if results.isEmpty { await search() }
        }
        .onChange(of: source) { _, _ in startSearch() }
        .onChange(of: sort) { _, _ in startSearch() }
        .onChange(of: textGenerationOnly) { _, _ in startSearch() }
    }

    private var loadMoreRow: some View {
        Button {
            Task { await loadMore() }
        } label: {
            HStack {
                Spacer()
                if isLoadingMore {
                    ProgressView()
                } else {
                    Label(L("さらに表示", "Show more"), systemImage: "chevron.down")
                }
                Spacer()
            }
        }
        .disabled(isLoadingMore)
        // 一番下まで来たら自動で次のページを読む
        .onAppear { Task { await loadMore() } }
    }

    private func startSearch() {
        Task { await search() }
    }

    private func search() async {
        isSearching = true
        errorMessage = nil
        defer { isSearching = false }
        do {
            let client = settings.makeHFClient()
            let page = try await client.searchModels(query: query, source: source, textGenerationOnly: textGenerationOnly,
                                                     sort: sort, limit: 50)
            results = page.models
            nextPage = page.next
            if results.isEmpty {
                errorMessage = L("見つかりませんでした。キーワードや絞り込みを変えてみてください。",
                                 "Nothing found. Try other keywords or filters.")
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func loadMore() async {
        guard let url = nextPage, !isLoadingMore, !isSearching else { return }
        isLoadingMore = true
        defer { isLoadingMore = false }
        do {
            let page = try await settings.makeHFClient().searchPage(url)
            let known = Set(results.map(\.id))
            results.append(contentsOf: page.models.filter { !known.contains($0.id) })
            nextPage = page.next
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

private struct SearchFormSection: View {
    @Binding var query: String
    @Binding var source: HFSearchSource
    @Binding var sort: HFSearchSort
    @Binding var textGenerationOnly: Bool
    var isSearching: Bool
    var onSearch: () -> Void

    var body: some View {
        Section {
            HStack {
                TextField(L("キーワード（例: qwen）", "Keywords (e.g. qwen)"), text: $query)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .onSubmit(onSearch)
                Button(action: onSearch) {
                    if isSearching { ProgressView() } else { Image(systemName: "magnifyingglass") }
                }
                .disabled(isSearching)
            }
            Picker(L("探す場所", "Source"), selection: $source) {
                ForEach(HFSearchSource.allCases) { source in
                    Text(source.label).tag(source)
                }
            }
            Picker(L("並べ順", "Sort"), selection: $sort) {
                ForEach(HFSearchSort.allCases) { sort in
                    Text(sort.label).tag(sort)
                }
            }
            if source != .anemll {
                Toggle(L("テキスト生成だけ", "Text generation only"), isOn: $textGenerationOnly)
            }
        } footer: {
            if source == .anemll {
                Text(L("ANEMLL のモデルは、埋め込み・本体（複数のチャンク）・出力層に分かれています。まとめてダウンロードして Neural Engine で動かします。",
                       "ANEMLL models are split into embeddings, body chunks and an output head. They are downloaded together and run on the Neural Engine."))
            }
        }
    }
}

private struct DirectRepoSection: View {
    @Binding var directRepo: String

    var body: some View {
        Section(L("リポジトリ ID を直接入力", "Enter a repository ID")) {
            HStack {
                TextField("owner/name", text: $directRepo)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                NavigationLink(value: directRepo.trimmingCharacters(in: .whitespaces)) {
                    Text(L("開く", "Open"))
                }
                .disabled(!directRepo.contains("/"))
            }
        }
    }
}

private struct SearchResultRow: View {
    let model: HFModelSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(model.id)
                .font(.callout.weight(.medium))
                .lineLimit(2)
            HStack(spacing: 10) {
                if let downloads = model.downloads {
                    Label("\(downloads)", systemImage: "arrow.down.circle")
                }
                if let likes = model.likes {
                    Label("\(likes)", systemImage: "heart")
                }
                if let tag = model.pipelineTag {
                    Text(tag)
                }
                if model.gated {
                    Label(L("要トークン", "Gated"), systemImage: "lock")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}

/// インストール済みのモデル一覧。
struct InstalledModelsList: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    var onDone: () -> Void

    @State private var deleting: InstalledModel?
    @State private var contextModelID: ContextTarget?

    struct ContextTarget: Identifiable {
        var id: String
    }

    var body: some View {
        List {
            if library.models.isEmpty {
                Text(L("まだモデルがありません。「おすすめ」から入手してください。",
                       "No models yet. Get one from Featured."))
                    .foregroundStyle(.secondary)
            }
            ForEach(library.models) { model in
                InstalledModelRow(model: model, onDone: onDone, onDelete: { deleting = model },
                                  onContext: { contextModelID = ContextTarget(id: model.id) })
            }

            Section {
                LabeledContent(L("モデルの合計", "Total models"), value: ByteFormat.string(library.totalBytesOnDisk))
                if let available = library.availableStorageBytes {
                    LabeledContent(L("空き容量", "Free storage"), value: ByteFormat.string(available))
                }
            }
        }
        .confirmationDialog(L("このモデルを削除しますか？", "Delete this model?"), isPresented: Binding(
            get: { deleting != nil },
            set: { if !$0 { deleting = nil } }
        ), titleVisibility: .visible) {
            Button(L("削除", "Delete"), role: .destructive) {
                if let model = deleting {
                    if chat.loadedInfo?.modelID == model.id { chat.unloadModel() }
                    library.delete(id: model.id)
                }
                deleting = nil
            }
        } message: {
            Text(deleting.map { "\($0.displayName)（\(ByteFormat.string($0.totalBytes))）" } ?? "")
        }
        .sheet(item: $contextModelID) { target in
            ContextSettingsView(modelID: target.id)
        }
    }
}

/// インストール済みのモデル 1 つ分の行。
private struct InstalledModelRow: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat
    let model: InstalledModel
    var onDone: () -> Void
    var onDelete: () -> Void
    var onContext: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.displayName).font(.headline)
                    Text(model.repoID).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if chat.loadedInfo?.modelID == model.id {
                    Badge(text: L("使用中", "In use"), color: .green)
                }
            }
            details
            if let state = library.downloads[model.id], !model.isComplete {
                DownloadStatusView(state: state)
            }
            actions
        }
        .padding(.vertical, 4)
    }

    private var details: some View {
        HStack(spacing: 10) {
            Text(ByteFormat.string(model.totalBytes))
            if let context = model.maxContext {
                Label(L("最大 \(context.formatted()) トークン", "Up to \(context.formatted()) tokens"), systemImage: "text.alignleft")
            }
            if model.originalContext != nil {
                Badge(text: L("拡張済み", "Extended"), color: .purple)
            }
            Text(model.kindLabel)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var actions: some View {
        HStack {
            if model.isComplete {
                Button {
                    chat.loadModel(id: model.id, library: library, settings: settings)
                    onDone()
                } label: {
                    Label(L("使う", "Use"), systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(chat.loadedInfo?.modelID == model.id)
                Button(action: onContext) {
                    Label(L("コンテキスト長…", "Context…"), systemImage: "arrow.up.left.and.arrow.down.right")
                }
                .buttonStyle(.bordered)
                .disabled(library.resizingModelIDs.contains(model.id))
            } else {
                DownloadControls(modelID: model.id)
            }
            Spacer()
            Button(role: .destructive, action: onDelete) {
                Label(L("削除", "Delete"), systemImage: "trash")
            }
            .buttonStyle(.bordered)
        }
    }
}

/// ダウンロードの進み具合。
struct DownloadStatusView: View {
    let state: DownloadState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let progress = state.progress {
                ProgressView(value: progress.fractionCompleted)
                HStack {
                    Text("\(ByteFormat.string(progress.completedBytes)) / \(ByteFormat.string(progress.totalBytes))")
                    Spacer()
                    if state.phase == .downloading {
                        Text("\(ByteFormat.string(Int64(progress.bytesPerSecond)))/s")
                        if let remaining = progress.estimatedSecondsRemaining {
                            Text(Self.formatDuration(remaining))
                        }
                    }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }
            switch state.phase {
            case .downloading:
                if state.progress == nil {
                    ProgressView().controlSize(.small)
                }
            case .paused:
                Label(L("一時停止中（続きから再開できます）", "Paused (can resume)"), systemImage: "pause.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    static func formatDuration(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total >= 3600 { return L("残り約 \(total / 3600) 時間 \(total % 3600 / 60) 分", "~\(total / 3600)h \(total % 3600 / 60)m left") }
        if total >= 60 { return L("残り約 \(total / 60) 分", "~\(total / 60) min left") }
        return L("残り約 \(total) 秒", "~\(total) s left")
    }
}

/// 一時停止・再開ボタン。
struct DownloadControls: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    let modelID: String

    var body: some View {
        if library.downloads[modelID]?.phase == .downloading {
            Button {
                library.pauseDownload(id: modelID)
            } label: {
                Label(L("一時停止", "Pause"), systemImage: "pause.fill")
            }
            .buttonStyle(.bordered)
        } else {
            Button {
                library.resumeDownload(id: modelID, client: settings.makeHFClient())
            } label: {
                Label(L("再開", "Resume"), systemImage: "arrow.down.circle")
            }
            .buttonStyle(.borderedProminent)
        }
    }
}

/// 小さなラベル。
struct Badge: View {
    let text: String
    var color: Color = .accentColor

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}
