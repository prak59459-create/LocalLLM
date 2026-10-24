import SwiftUI
import UniformTypeIdentifiers

/// コーディングワークスペース（CDWS）の入り口: プロジェクトの一覧と作成。
struct WorkspaceHomeView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(WorkspaceStore.self) private var store

    @State private var openSession: WorkspaceSession?
    @State private var creating = false
    @State private var importingFolder = false
    @State private var error: String?
    @State private var removing: WorkspaceProject?
    /// 作ったばかりのプロジェクト（作成画面が閉じてから開く）
    @State private var created: CreatedProject?

    private struct CreatedProject {
        var project: WorkspaceProject
        var roadmap: Roadmap
    }

    var body: some View {
        NavigationStack {
            content
                .background(CDWSTheme.canvas)
                .navigationTitle(L("コーディングワークスペース", "Coding Workspace"))
                .navigationBarTitleDisplayMode(.inline)
                .toolbarBackground(CDWSTheme.panel, for: .navigationBar)
                .toolbarBackground(.visible, for: .navigationBar)
                .toolbar { toolbar }
                .navigationDestination(item: $openSession) { session in
                    WorkspaceView(session: session, onClose: { close(session) })
                }
                .sheet(isPresented: $creating, onDismiss: openCreated) {
                    NewProjectSheet(onCreated: { project, roadmap in
                        created = CreatedProject(project: project, roadmap: roadmap)
                    })
                }
                .fileImporter(isPresented: $importingFolder, allowedContentTypes: [.folder]) { result in
                    importFolder(result)
                }
                .alert(L("開けませんでした", "Could not open"), isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
                    Button("OK") { error = nil }
                } message: {
                    Text(error ?? "")
                }
                .confirmationDialog(L("プロジェクトを一覧から外しますか？", "Remove the project from the list?"),
                                    isPresented: Binding(get: { removing != nil }, set: { if !$0 { removing = nil } }),
                                    titleVisibility: .visible) {
                    removeButtons
                }
        }
        // IDE らしい暗い見た目にそろえる
        .preferredColorScheme(.dark)
        .tint(CDWSTheme.violet)
    }

    @ViewBuilder
    private var content: some View {
        if store.projects.isEmpty {
            WorkspaceEmptyState(onCreate: { creating = true }, onImport: { importingFolder = true })
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    WorkspaceHero(count: store.projects.count, onCreate: { creating = true }, onImport: { importingFolder = true })
                    ProjectGrid(projects: store.sortedProjects, onOpen: { open($0, roadmap: nil) }, onRemove: { removing = $0 })
                    Text(L("「フォルダを開く」で、iCloud Drive などのフォルダ（Swift Playgrounds の .swiftpm も）をそのままプロジェクトとして編集できます。",
                           "Use \"Open Folder\" to edit any folder (iCloud Drive, Swift Playgrounds .swiftpm packages, …) as a project."))
                        .font(.caption)
                        .foregroundStyle(CDWSTheme.secondaryText)
                }
                .padding(20)
                .frame(maxWidth: 1100)
                .frame(maxWidth: .infinity)
            }
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .cancellationAction) {
            Button(L("閉じる", "Close")) { dismiss() }
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Button { importingFolder = true } label: {
                Label(L("フォルダを開く", "Open Folder"), systemImage: "folder.badge.plus")
            }
            Button { creating = true } label: {
                Label(L("新しいプロジェクト", "New Project"), systemImage: "plus")
            }
        }
    }

    @ViewBuilder
    private var removeButtons: some View {
        if let project = removing {
            if !project.isExternal {
                Button(L("ファイルごと削除", "Delete with files"), role: .destructive) {
                    store.remove(project, deleteFiles: true)
                    removing = nil
                }
            }
            Button(project.isExternal ? L("一覧から外す（フォルダは残す）", "Remove (keep the folder)") : L("一覧から外す（ファイルは残す）", "Remove (keep files)")) {
                store.remove(project, deleteFiles: false)
                removing = nil
            }
        }
    }

    private func open(_ project: WorkspaceProject, roadmap: Roadmap?) {
        do {
            let url = try store.open(project)
            openSession = WorkspaceSession(project: project, root: url, initialRoadmap: roadmap)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func openCreated() {
        guard let created else { return }
        self.created = nil
        open(created.project, roadmap: created.roadmap)
    }

    private func close(_ session: WorkspaceSession) {
        session.saveAll()
        session.stopAgent()
        store.close(session.project)
        openSession = nil
    }

    private func importFolder(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                let project = try store.addExisting(url)
                open(project, roadmap: nil)
            } catch {
                self.error = error.localizedDescription
            }
        case .failure(let failure):
            error = failure.localizedDescription
        }
    }
}

extension WorkspaceSession: Hashable {
    nonisolated static func == (lhs: WorkspaceSession, rhs: WorkspaceSession) -> Bool { lhs === rhs }
    nonisolated func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }
}

/// 一覧の上の大きな見出し
private struct WorkspaceHero: View {
    let count: Int
    var onCreate: () -> Void
    var onImport: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            CDWSIconTile(symbol: "chevron.left.forwardslash.chevron.right", size: 58)
            VStack(alignment: .leading, spacing: 4) {
                Text(L("iPad の中の AI と一緒に作る", "Build with the AI on your iPad"))
                    .font(.title2.bold())
                    .foregroundStyle(CDWSTheme.text)
                Text(L("\(count) 個のプロジェクト・エージェントが読んで、書いて、直します", "\(count) projects · the agent reads, writes and fixes"))
                    .font(.callout)
                    .foregroundStyle(CDWSTheme.secondaryText)
            }
            Spacer(minLength: 12)
            WorkspaceHeroButtons(onCreate: onCreate, onImport: onImport)
        }
        .padding(20)
        .background(CDWSTheme.hero, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).strokeBorder(CDWSTheme.strongStroke))
    }
}

private struct WorkspaceHeroButtons: View {
    var onCreate: () -> Void
    var onImport: () -> Void

    var body: some View {
        ViewThatFits {
            HStack(spacing: 10) { buttons }
            VStack(alignment: .trailing, spacing: 8) { buttons }
        }
    }

    @ViewBuilder
    private var buttons: some View {
        Button(action: onImport) {
            Label(L("フォルダを開く", "Open Folder"), systemImage: "folder.badge.plus")
        }
        .buttonStyle(.bordered)
        .tint(.white)
        Button(action: onCreate) {
            Label(L("新しいプロジェクト", "New Project"), systemImage: "plus")
                .fontWeight(.semibold)
        }
        .buttonStyle(.borderedProminent)
        .tint(CDWSTheme.violet)
    }
}

/// プロジェクトのカードを並べる
private struct ProjectGrid: View {
    let projects: [WorkspaceProject]
    var onOpen: (WorkspaceProject) -> Void
    var onRemove: (WorkspaceProject) -> Void

    private let columns = [GridItem(.adaptive(minimum: 250, maximum: 360), spacing: 14)]

    var body: some View {
        LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
            ForEach(projects) { project in
                Button { onOpen(project) } label: {
                    ProjectCard(project: project)
                }
                .buttonStyle(.plain)
                .contextMenu {
                    Button(role: .destructive) { onRemove(project) } label: {
                        Label(L("一覧から外す…", "Remove…"), systemImage: "trash")
                    }
                }
            }
        }
    }
}

private struct ProjectCard: View {
    let project: WorkspaceProject

    private var symbol: String {
        ProjectTemplates.template(id: project.templateID ?? "")?.symbolName ?? (project.isExternal ? "folder.fill" : "folder")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                CDWSIconTile(symbol: symbol, size: 40)
                Spacer()
                if project.isExternal {
                    CDWSChip(text: L("外のフォルダ", "External"), symbol: "link", color: CDWSTheme.cyan)
                }
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(project.name)
                    .font(.headline)
                    .foregroundStyle(CDWSTheme.text)
                    .lineLimit(1)
                Text(project.locationLabel)
                    .font(.caption)
                    .foregroundStyle(CDWSTheme.secondaryText)
                    .lineLimit(1)
            }
            HStack {
                if let date = project.lastOpened {
                    Label(date.formatted(.relative(presentation: .named)), systemImage: "clock")
                }
                Spacer()
                Image(systemName: "arrow.right.circle.fill").foregroundStyle(CDWSTheme.violet)
            }
            .font(.caption)
            .foregroundStyle(CDWSTheme.secondaryText)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .cdwsPanel(radius: 18)
        .contentShape(RoundedRectangle(cornerRadius: 18))
    }
}

private struct WorkspaceEmptyState: View {
    var onCreate: () -> Void
    var onImport: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            CDWSIconTile(symbol: "chevron.left.forwardslash.chevron.right", size: 76)
            Text(L("iPad の中の AI と一緒に作る", "Build together with the AI on your iPad"))
                .font(.title2.bold())
                .foregroundStyle(CDWSTheme.text)
            VStack(alignment: .leading, spacing: 10) {
                WorkspaceFeatureRow(symbol: "sparkles", text: L("エージェントがファイルを読んで、書いて、直す", "The agent reads, writes and fixes files"))
                WorkspaceFeatureRow(symbol: "folder", text: L("ファイルエクスプローラとコードエディタ", "File explorer and code editor"))
                WorkspaceFeatureRow(symbol: "point.3.connected.trianglepath.dotted", text: L("作った機能が丸で並ぶロードマップ", "A roadmap where features appear as circles"))
                WorkspaceFeatureRow(symbol: "play.rectangle", text: L("Web アプリのプレビューと JavaScript の実行", "Web previews and JavaScript runs"))
                WorkspaceFeatureRow(symbol: "arrow.uturn.backward", text: L("すべての変更は元に戻せる", "Every change can be undone"))
            }
            .padding(18)
            .cdwsPanel(radius: 18)
            WorkspaceHeroButtons(onCreate: onCreate, onImport: onImport)
        }
        .padding(32)
        .frame(maxWidth: 560)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct WorkspaceFeatureRow: View {
    let symbol: String
    let text: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol)
                .font(.callout.weight(.semibold))
                .foregroundStyle(CDWSTheme.cyan)
                .frame(width: 24)
            Text(text).font(.callout).foregroundStyle(CDWSTheme.text)
        }
    }
}

/// 新しいプロジェクトの作成画面。
private struct NewProjectSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(WorkspaceStore.self) private var store
    var onCreated: (WorkspaceProject, Roadmap) -> Void

    @State private var name = ""
    @State private var templateID = ProjectTemplates.web.id
    @State private var parentFolder: URL?
    @State private var pickingFolder = false
    @State private var error: String?

    private var template: ProjectTemplate { ProjectTemplates.template(id: templateID) ?? ProjectTemplates.empty }

    var body: some View {
        NavigationStack {
            Form {
                Section(L("名前", "Name")) {
                    TextField(L("例: 家計簿アプリ", "e.g. Budget App"), text: $name)
                }
                Section(L("ひな形", "Template")) {
                    ForEach(ProjectTemplates.all) { item in
                        TemplateRow(template: item, isSelected: item.id == templateID)
                            .contentShape(Rectangle())
                            .onTapGesture { templateID = item.id }
                    }
                }
                Section {
                    LabeledContent(L("保存先", "Location"),
                                   value: parentFolder.map { WorkspaceStore.displayPath($0) } ?? L("このアプリの中", "Inside this app"))
                    Button(L("フォルダを選ぶ…", "Choose Folder…")) { pickingFolder = true }
                    if parentFolder != nil {
                        Button(L("アプリの中に保存する", "Save inside the app")) { parentFolder = nil }
                    }
                } footer: {
                    Text(L("iCloud Drive の Playgrounds フォルダを選ぶと、iPad アプリのひな形を Swift Playgrounds でそのまま開けます。",
                           "Choose the Playgrounds folder in iCloud Drive to open the iPad app template in Swift Playgrounds."))
                }
                if let error {
                    Text(error).foregroundStyle(.orange)
                }
            }
            .navigationTitle(L("新しいプロジェクト", "New Project"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(L("キャンセル", "Cancel")) { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("作成", "Create"), action: create)
                        .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .fileImporter(isPresented: $pickingFolder, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { parentFolder = url }
            }
        }
    }

    private func create() {
        do {
            let (project, roadmap) = try store.create(name: name.trimmingCharacters(in: .whitespaces), template: template, in: parentFolder)
            onCreated(project, roadmap)
            dismiss()
        } catch {
            self.error = error.localizedDescription
        }
    }
}

private struct TemplateRow: View {
    let template: ProjectTemplate
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: template.symbolName)
                .font(.title3)
                .frame(width: 32)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(template.name)
                Text(template.summary).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if isSelected {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.tint)
            }
        }
    }
}
