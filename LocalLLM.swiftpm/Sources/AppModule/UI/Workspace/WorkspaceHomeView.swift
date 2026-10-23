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
                .navigationTitle(L("コーディングワークスペース", "Coding Workspace"))
                .navigationBarTitleDisplayMode(.inline)
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
    }

    @ViewBuilder
    private var content: some View {
        if store.projects.isEmpty {
            WorkspaceEmptyState(onCreate: { creating = true }, onImport: { importingFolder = true })
        } else {
            List {
                Section {
                    ForEach(store.sortedProjects) { project in
                        Button {
                            open(project, roadmap: nil)
                        } label: {
                            ProjectRow(project: project)
                        }
                        .swipeActions {
                            Button(role: .destructive) { removing = project } label: {
                                Label(L("外す", "Remove"), systemImage: "trash")
                            }
                        }
                        .contextMenu {
                            Button(role: .destructive) { removing = project } label: {
                                Label(L("一覧から外す…", "Remove…"), systemImage: "trash")
                            }
                        }
                    }
                } footer: {
                    Text(L("「フォルダを開く」で、iCloud Drive などのフォルダ（Swift Playgrounds の .swiftpm も）をそのままプロジェクトとして編集できます。",
                           "Use \"Open Folder\" to edit any folder (iCloud Drive, Swift Playgrounds .swiftpm packages, …) as a project."))
                }
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

private struct ProjectRow: View {
    let project: WorkspaceProject

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: ProjectTemplates.template(id: project.templateID ?? "")?.symbolName ?? "folder")
                .font(.title2)
                .foregroundStyle(.tint)
                .frame(width: 40)
            VStack(alignment: .leading, spacing: 3) {
                Text(project.name).font(.headline)
                Text(project.locationLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            if let date = project.lastOpened {
                Text(date.formatted(.relative(presentation: .named)))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

private struct WorkspaceEmptyState: View {
    var onCreate: () -> Void
    var onImport: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "chevron.left.forwardslash.chevron.right")
                .font(.system(size: 52))
                .foregroundStyle(.tint)
            Text(L("iPad の中の AI と一緒に作る", "Build together with the AI on your iPad"))
                .font(.title2.bold())
            VStack(alignment: .leading, spacing: 8) {
                Label(L("エージェントがファイルを読んで、書いて、直す", "The agent reads, writes and fixes files"), systemImage: "sparkles")
                Label(L("ファイルエクスプローラとコードエディタ", "File explorer and code editor"), systemImage: "folder")
                Label(L("作った機能が丸で並ぶロードマップ", "A roadmap where features appear as circles"), systemImage: "point.3.connected.trianglepath.dotted")
                Label(L("Web アプリのプレビューと JavaScript の実行", "Web previews and JavaScript runs"), systemImage: "play.rectangle")
                Label(L("すべての変更は元に戻せる", "Every change can be undone"), systemImage: "arrow.uturn.backward")
            }
            .font(.callout)
            .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button(action: onImport) {
                    Label(L("フォルダを開く", "Open Folder"), systemImage: "folder.badge.plus")
                }
                .buttonStyle(.bordered)
                Button(action: onCreate) {
                    Label(L("新しいプロジェクト", "New Project"), systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(32)
        .frame(maxWidth: 560)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
