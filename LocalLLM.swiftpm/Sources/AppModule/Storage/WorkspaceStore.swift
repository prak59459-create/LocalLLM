import Foundation
import Observation

/// コーディングワークスペースのプロジェクト 1 つ。
struct WorkspaceProject: Codable, Identifiable, Hashable, Sendable {
    enum Location: Codable, Hashable, Sendable {
        /// アプリの中（Documents/CDWS/フォルダ名）
        case app(folderName: String)
        /// 「ファイル」App で選んだフォルダ（iCloud Drive など）。ブックマークで覚えておく。
        case external(bookmark: Data, displayPath: String)
    }

    var id: UUID = UUID()
    var name: String
    var location: Location
    var templateID: String?
    var createdAt: Date = Date()
    var lastOpened: Date?

    var locationLabel: String {
        switch location {
        case .app: return L("このアプリの中", "Inside this app")
        case .external(_, let path): return path
        }
    }

    var isExternal: Bool {
        if case .external = location { return true }
        return false
    }
}

enum WorkspaceStoreError: Error, LocalizedError {
    case cannotAccess(String)
    case alreadyExists(String)

    var errorDescription: String? {
        switch self {
        case .cannotAccess(let name):
            return L("フォルダ「\(name)」を開けません。もう一度フォルダを選び直してください。",
                     "Cannot open the folder \"\(name)\". Please choose the folder again.")
        case .alreadyExists(let name):
            return L("「\(name)」という名前のフォルダは既にあります。", "A folder named \"\(name)\" already exists.")
        }
    }
}

/// プロジェクトの一覧（UserDefaults に保存）。
@MainActor
@Observable
final class WorkspaceStore {
    private(set) var projects: [WorkspaceProject] = []

    /// アプリの中のプロジェクトを置くフォルダ
    let appRoot: URL
    @ObservationIgnored private let defaults = UserDefaults.standard
    @ObservationIgnored private let key = "cdws.projects"
    /// startAccessingSecurityScopedResource したフォルダ（閉じるときに止める）
    @ObservationIgnored private var accessed: [UUID: URL] = [:]

    init() {
        let documents = (try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        appRoot = documents.appendingPathComponent("CDWS", isDirectory: true)
        try? FileManager.default.createDirectory(at: appRoot, withIntermediateDirectories: true)
        if let data = defaults.data(forKey: key), let saved = try? JSONDecoder().decode([WorkspaceProject].self, from: data) {
            projects = saved
        }
    }

    var sortedProjects: [WorkspaceProject] {
        projects.sorted { ($0.lastOpened ?? $0.createdAt) > ($1.lastOpened ?? $1.createdAt) }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(projects) { defaults.set(data, forKey: key) }
    }

    /// ひな形から新しいプロジェクトを作る。parent が nil ならアプリの中に作る。
    func create(name: String, template: ProjectTemplate, in parent: URL?) throws -> (WorkspaceProject, Roadmap) {
        let folderName = ProjectTemplates.folderName(for: name, template: template)
        let base = parent ?? appRoot
        let folder = base.appendingPathComponent(folderName, isDirectory: true)
        let scoped = parent?.startAccessingSecurityScopedResource() ?? false
        defer { if scoped { parent?.stopAccessingSecurityScopedResource() } }
        if FileManager.default.fileExists(atPath: folder.path) { throw WorkspaceStoreError.alreadyExists(folderName) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let roadmap = try ProjectTemplates.create(template, name: name, in: WorkspaceFileSystem(root: folder))
        let location: WorkspaceProject.Location
        if parent != nil {
            let bookmark = try folder.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
            location = .external(bookmark: bookmark, displayPath: Self.displayPath(folder))
        } else {
            location = .app(folderName: folderName)
        }
        let project = WorkspaceProject(name: name, location: location, templateID: template.id)
        projects.append(project)
        save()
        return (project, roadmap)
    }

    /// 「ファイル」App で選んだ既存のフォルダを追加する。
    func addExisting(_ url: URL) throws -> WorkspaceProject {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let bookmark = try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
        let name = url.deletingPathExtension().lastPathComponent
        if let existing = projects.first(where: {
            if case .external(_, let path) = $0.location { return path == Self.displayPath(url) }
            return false
        }) {
            return existing
        }
        let project = WorkspaceProject(name: name, location: .external(bookmark: bookmark, displayPath: Self.displayPath(url)))
        projects.append(project)
        save()
        return project
    }

    /// プロジェクトのフォルダを開く（外のフォルダならアクセスを始める）。
    func open(_ project: WorkspaceProject) throws -> URL {
        let url: URL
        switch project.location {
        case .app(let folderName):
            url = appRoot.appendingPathComponent(folderName, isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        case .external(let bookmark, _):
            var stale = false
            guard let resolved = try? URL(resolvingBookmarkData: bookmark, options: [], relativeTo: nil, bookmarkDataIsStale: &stale) else {
                throw WorkspaceStoreError.cannotAccess(project.name)
            }
            if resolved.startAccessingSecurityScopedResource() { accessed[project.id] = resolved }
            guard FileManager.default.fileExists(atPath: resolved.path) else { throw WorkspaceStoreError.cannotAccess(project.name) }
            if stale, let fresh = try? resolved.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil),
               let index = projects.firstIndex(where: { $0.id == project.id }) {
                projects[index].location = .external(bookmark: fresh, displayPath: Self.displayPath(resolved))
            }
            url = resolved
        }
        if let index = projects.firstIndex(where: { $0.id == project.id }) {
            projects[index].lastOpened = Date()
            save()
        }
        return url
    }

    func close(_ project: WorkspaceProject) {
        accessed[project.id]?.stopAccessingSecurityScopedResource()
        accessed[project.id] = nil
    }

    func rename(_ project: WorkspaceProject, to name: String) {
        guard let index = projects.firstIndex(where: { $0.id == project.id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { projects[index].name = trimmed }
        save()
    }

    /// 一覧から外す。deleteFiles ならアプリの中のフォルダも消す（外のフォルダは消さない）。
    func remove(_ project: WorkspaceProject, deleteFiles: Bool) {
        close(project)
        if deleteFiles, case .app(let folderName) = project.location {
            try? FileManager.default.removeItem(at: appRoot.appendingPathComponent(folderName))
        }
        projects.removeAll { $0.id == project.id }
        save()
    }

    static func displayPath(_ url: URL) -> String {
        let parts = url.standardizedFileURL.pathComponents
        return parts.suffix(3).joined(separator: "/")
    }
}
