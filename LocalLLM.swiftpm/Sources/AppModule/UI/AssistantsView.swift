import SwiftData
import SwiftUI
import UniformTypeIdentifiers

/// マイアシスタントの一覧（シートで表示）。
struct AssistantsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context
    @Environment(KnowledgeStore.self) private var knowledge
    @Query(sort: \Assistant.createdAt) private var assistants: [Assistant]

    /// 「この アシスタントで新しいチャット」を押したとき
    var onStartChat: (Assistant) -> Void

    @State private var path: [UUID] = []
    @State private var deleting: Assistant?

    var body: some View {
        NavigationStack(path: $path) {
            List {
                Section {
                    if assistants.isEmpty {
                        Text(L("まだアシスタントがありません。下のひな形から作ってみましょう。",
                               "No assistants yet. Create one from a template below."))
                            .foregroundStyle(.secondary)
                    }
                    ForEach(assistants) { assistant in
                        NavigationLink(value: assistant.id) {
                            AssistantRow(assistant: assistant)
                        }
                        .swipeActions {
                            Button(role: .destructive) {
                                deleting = assistant
                            } label: {
                                Label(L("削除", "Delete"), systemImage: "trash")
                            }
                        }
                    }
                } header: {
                    Text(L("マイアシスタント", "My Assistants"))
                } footer: {
                    Text(L("モデル・システムプロンプト・生成設定・参考資料（自分のコードや文書）をまとめて、自分専用の AI を作れます。",
                           "Combine a model, system prompt, settings and your own code or documents into a personal AI."))
                }

                Section(L("ひな形から作る", "Create from a template")) {
                    ForEach(AssistantTemplate.all) { template in
                        Button {
                            create(from: template)
                        } label: {
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: template.symbolName)
                                    .frame(width: 28)
                                    .foregroundStyle(.tint)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(template.name).foregroundStyle(.primary)
                                    Text(template.summary)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle(L("アシスタント", "Assistants"))
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: UUID.self) { id in
                if let assistant = assistants.first(where: { $0.id == id }) {
                    AssistantEditorView(assistant: assistant, onStartChat: {
                        onStartChat(assistant)
                        dismiss()
                    })
                }
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close")) { dismiss() }
                }
            }
            .confirmationDialog(L("このアシスタントを削除しますか？", "Delete this assistant?"), isPresented: Binding(
                get: { deleting != nil },
                set: { if !$0 { deleting = nil } }
            ), titleVisibility: .visible) {
                Button(L("削除", "Delete"), role: .destructive) {
                    if let assistant = deleting {
                        knowledge.deleteAll(for: assistant.id)
                        context.delete(assistant)
                        try? context.save()
                    }
                    deleting = nil
                }
            } message: {
                Text(L("参考資料も削除されます。会話は残ります。", "Its reference files are deleted too. Chats are kept."))
            }
        }
    }

    private func create(from template: AssistantTemplate) {
        let assistant = template.makeAssistant()
        context.insert(assistant)
        try? context.save()
        path.append(assistant.id)
    }
}

private struct AssistantRow: View {
    @Environment(KnowledgeStore.self) private var knowledge
    let assistant: Assistant

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: assistant.symbolName)
                .font(.title3)
                .frame(width: 32)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(assistant.name.isEmpty ? L("名前なし", "Untitled") : assistant.name)
                Text(L("参考資料 \(knowledge.index(for: assistant.id).sources.count) ファイル",
                       "\(knowledge.index(for: assistant.id).sources.count) reference files"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// アシスタントの編集画面。
struct AssistantEditorView: View {
    @Environment(\.modelContext) private var context
    @Environment(ModelLibrary.self) private var library
    @Environment(KnowledgeStore.self) private var knowledge
    @Bindable var assistant: Assistant
    var onStartChat: () -> Void

    @State private var importing = false
    @State private var importMessage: String?
    @State private var testQuery = ""

    private static let symbols = [
        "person.crop.circle", "chevron.left.forwardslash.chevron.right", "hammer", "terminal", "swift",
        "character.bubble", "text.badge.checkmark", "book", "graduationcap", "lightbulb", "person.2.wave.2", "sparkles",
    ]

    var body: some View {
        Form {
            Section {
                TextField(L("名前", "Name"), text: $assistant.name)
                symbolPicker
                Button {
                    try? context.save()
                    onStartChat()
                } label: {
                    Label(L("このアシスタントで新しいチャット", "New chat with this assistant"), systemImage: "square.and.pencil")
                }
            }

            Section {
                Picker(L("使うモデル", "Model"), selection: $assistant.preferredModelID) {
                    Text(L("指定しない（読み込み中のモデル）", "Any (currently loaded)")).tag(String?.none)
                    ForEach(library.completedModels) { model in
                        Text(model.displayName).tag(String?.some(model.id))
                    }
                }
                if let hint = templateHint {
                    Text(hint).font(.caption).foregroundStyle(.secondary)
                }
            } header: {
                Text(L("モデル", "Model"))
            }

            Section {
                TextEditor(text: $assistant.systemPrompt)
                    .frame(minHeight: 140)
                    .font(.callout)
            } header: {
                Text(L("システムプロンプト", "System prompt"))
            }

            GenerationOverrideSection(assistant: assistant)

            KnowledgeSection(assistant: assistant, importing: $importing, importMessage: $importMessage, testQuery: $testQuery)
        }
        .navigationTitle(assistant.name.isEmpty ? L("アシスタント", "Assistant") : assistant.name)
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(isPresented: $importing,
                      allowedContentTypes: [.folder, .text, .sourceCode, .json, .xml, .html, .commaSeparatedText],
                      allowsMultipleSelection: true) { result in
            switch result {
            case .success(let urls):
                Task {
                    let report = await knowledge.importItems(urls, into: assistant.id)
                    importMessage = L("\(report.added.count) ファイルを取り込みました", "Imported \(report.added.count) files")
                        + (report.skipped.isEmpty ? "" : L("（\(report.skipped.count) 件はスキップ）", " (\(report.skipped.count) skipped)"))
                }
            case .failure(let error):
                importMessage = error.localizedDescription
            }
        }
        .onDisappear {
            assistant.updatedAt = Date()
            try? context.save()
        }
    }

    private var symbolPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(Self.symbols, id: \.self) { symbol in
                    Button {
                        assistant.symbolName = symbol
                    } label: {
                        Image(systemName: symbol)
                            .font(.title3)
                            .frame(width: 40, height: 40)
                            .background(symbol == assistant.symbolName ? Color.accentColor.opacity(0.2) : Color.clear,
                                        in: RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var templateHint: String? {
        guard let id = assistant.templateID,
              let hint = AssistantTemplate.all.first(where: { $0.id == id })?.recommendedModelHint else { return nil }
        return L("おすすめ: 名前に「\(hint)」を含むモデル（例: Qwen2.5 Coder）", "Recommended: a model with \"\(hint)\" in its name (e.g. Qwen2.5 Coder)")
    }
}

/// 生成設定をアシスタント専用にする部分。
private struct GenerationOverrideSection: View {
    @Bindable var assistant: Assistant

    var body: some View {
        Section {
            Toggle(L("専用の生成設定を使う", "Use custom generation settings"), isOn: $assistant.useCustomGeneration)
            if assistant.useCustomGeneration {
                let temperature = Binding<Double>(
                    get: { assistant.generation.temperature },
                    set: { value in
                        var generation = assistant.generation
                        generation.temperature = value
                        assistant.generation = generation
                    }
                )
                let maxTokens = Binding<Int>(
                    get: { assistant.generation.maxNewTokens },
                    set: { value in
                        var generation = assistant.generation
                        generation.maxNewTokens = value
                        assistant.generation = generation
                    }
                )
                HStack {
                    Text(L("温度", "Temperature"))
                    Slider(value: temperature, in: 0...2, step: 0.05)
                    Text(String(format: "%.2f", assistant.generation.temperature))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Stepper(value: maxTokens, in: 32...8192, step: 32) {
                    LabeledContent(L("最大生成トークン数", "Max new tokens"), value: "\(assistant.generation.maxNewTokens)")
                }
            }
            Picker(L("思考モード", "Thinking mode"), selection: Binding(
                get: { assistant.thinkingMode },
                set: { assistant.thinkingMode = $0 }
            )) {
                ForEach(ThinkingMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
        } header: {
            Text(L("生成", "Generation"))
        }
    }
}

/// 参考資料（自分のコードや文書）の取り込みと検索テスト。
private struct KnowledgeSection: View {
    @Environment(KnowledgeStore.self) private var knowledge
    @Bindable var assistant: Assistant
    @Binding var importing: Bool
    @Binding var importMessage: String?
    @Binding var testQuery: String

    var body: some View {
        let index = knowledge.index(for: assistant.id)
        Section {
            Toggle(L("参考資料を使う", "Use reference files"), isOn: $assistant.useKnowledge)
            Stepper(value: $assistant.knowledgeResultCount, in: 1...8) {
                LabeledContent(L("1 回に入れる数", "Snippets per question"), value: "\(assistant.knowledgeResultCount)")
            }
            Button {
                importing = true
            } label: {
                if knowledge.importingAssistantIDs.contains(assistant.id) {
                    HStack {
                        ProgressView()
                        Text(L("取り込み中…", "Importing…"))
                    }
                } else {
                    Label(L("ファイルやフォルダを追加", "Add files or folders"), systemImage: "plus.circle")
                }
            }
            .disabled(knowledge.importingAssistantIDs.contains(assistant.id))
            if let importMessage {
                Text(importMessage).font(.caption).foregroundStyle(.secondary)
            }
            ForEach(index.sources, id: \.self) { source in
                Text(source)
                    .font(.caption.monospaced())
                    .lineLimit(2)
                    .swipeActions {
                        Button(role: .destructive) {
                            knowledge.remove(source: source, from: assistant.id)
                        } label: {
                            Label(L("削除", "Delete"), systemImage: "trash")
                        }
                    }
            }
        } header: {
            Text(L("参考資料（\(index.sources.count) ファイル・\(index.chunkCount) 個の断片）",
                   "Reference files (\(index.sources.count) files · \(index.chunkCount) chunks)"))
        } footer: {
            Text(L("質問のたびに、関係のありそうな部分を検索してモデルに渡します。Swift Playgrounds のプロジェクトフォルダごと追加すると「自分のコードを知っているコーディング AI」になります。",
                   "Relevant parts are searched and passed to the model with every question. Add a whole Swift Playgrounds project folder to get a coding AI that knows your code."))
        }

        if index.chunkCount > 0 {
            Section(L("検索のテスト", "Test search")) {
                TextField(L("例: ダウンロードの再開", "e.g. resume download"), text: $testQuery)
                    .textInputAutocapitalization(.never)
                ForEach(testResults, id: \.chunk.id) { result in
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(result.chunk.source) L\(result.chunk.startLine)-\(result.chunk.endLine)")
                            .font(.caption.monospaced())
                        Text(String(format: "score %.2f", result.score))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var testResults: [KnowledgeSearchResult] {
        let query = testQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }
        return knowledge.search(query, in: assistant.id, limit: assistant.knowledgeResultCount)
    }
}
