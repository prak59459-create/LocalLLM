import Foundation
import Observation

/// アプリ自身の更新: Swift Playgrounds の .swiftpm を、GitHub や「ファイル」の新しいソースで上書きして、版を上げる。
@MainActor
@Observable
final class AppUpdateController {
    enum Source: String, CaseIterable, Identifiable {
        case github
        case file

        var id: String { rawValue }
        var label: String {
            switch self {
            case .github: return "GitHub"
            case .file: return L("ファイル", "File")
            }
        }
    }

    enum Phase: Equatable {
        case idle
        case working(String)
        case ready(UpdatePlanSummary)
        case done(String)
        case failed(String)
    }

    /// 画面に出す、更新の内容
    struct UpdatePlanSummary: Equatable {
        var added: [String]
        var changed: [String]
        var removed: [String]
        var unchanged: Int
        var from: String
        var to: String
    }

    var source: Source {
        didSet { defaults.set(source.rawValue, forKey: Keys.source) }
    }
    var repository: String {
        didSet { defaults.set(repository, forKey: Keys.repository) }
    }
    var branch: String {
        didSet { defaults.set(branch, forKey: Keys.branch) }
    }
    var token: String {
        didSet { Keychain.set(token, for: Keys.token) }
    }
    private(set) var targetName: String?
    private(set) var sourceFileName: String?
    private(set) var phase: Phase = .idle
    private(set) var hasBackup = false

    @ObservationIgnored private let defaults = UserDefaults.standard
    @ObservationIgnored private var plan: UpdatePlan?
    @ObservationIgnored private var stagedPackage: URL?
    @ObservationIgnored private var pickedSource: URL?

    /// アプリの版（実行中のもの）
    static var runningVersion: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }

    private var workRoot: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("AppUpdate", isDirectory: true)
    }

    private var backupFolder: URL {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("AppUpdateBackup", isDirectory: true)
    }

    init() {
        source = defaults.string(forKey: Keys.source).flatMap(Source.init(rawValue:)) ?? .github
        repository = defaults.string(forKey: Keys.repository) ?? "prak59459-create/LocalLLM"
        branch = defaults.string(forKey: Keys.branch) ?? "main"
        token = Keychain.get(Keys.token) ?? ""
        if let data = defaults.data(forKey: Keys.target), let url = resolve(data) {
            targetName = url.lastPathComponent
        }
        hasBackup = FileManager.default.fileExists(atPath: backupFolder.appendingPathComponent("plan.json").path)
    }

    // MARK: - 選ぶ

    /// 上書きするアプリ（.swiftpm）を選んだ
    func setTarget(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("Package.swift").path) else {
            phase = .failed(AppUpdateError.notAPackage(url.lastPathComponent).localizedDescription)
            return
        }
        if let bookmark = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            defaults.set(bookmark, forKey: Keys.target)
            targetName = url.lastPathComponent
            phase = .idle
        }
    }

    /// 新しいソース（.zip か .swiftpm のフォルダ）を選んだ
    func setSourceFile(_ url: URL) {
        pickedSource = url
        sourceFileName = url.lastPathComponent
        phase = .idle
    }

    private func resolve(_ bookmark: Data) -> URL? {
        var stale = false
        return try? URL(resolvingBookmarkData: bookmark, options: [], relativeTo: nil, bookmarkDataIsStale: &stale)
    }

    // MARK: - 確認・更新

    var canPrepare: Bool {
        guard targetName != nil else { return false }
        switch phase {
        case .working: return false
        default: return source == .github ? !repository.isEmpty : pickedSource != nil
        }
    }

    /// 新しいソースを取ってきて、何が変わるかを調べる
    func prepare() async {
        guard let data = defaults.data(forKey: Keys.target), let target = resolve(data) else {
            phase = .failed(L("更新するアプリを選んでください", "Choose the app to update"))
            return
        }
        do {
            let staging = workRoot.appendingPathComponent("staging", isDirectory: true)
            try? FileManager.default.removeItem(at: workRoot)
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)

            switch source {
            case .github:
                phase = .working(L("GitHub からダウンロードしています…", "Downloading from GitHub…"))
                guard let request = AppUpdater.githubArchiveRequest(repository: repository, ref: branch, token: token) else {
                    throw AppUpdateError.packageNotFound
                }
                let (downloaded, response) = try await URLSession.shared.download(for: request)
                if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                    throw AppUpdateError.download(http.statusCode)
                }
                let zip = workRoot.appendingPathComponent("source.zip")
                try FileManager.default.moveItem(at: downloaded, to: zip)
                phase = .working(L("展開しています…", "Extracting…"))
                try await Self.extract(zip, to: staging)
            case .file:
                guard let picked = pickedSource else { throw AppUpdateError.packageNotFound }
                phase = .working(L("ファイルを読み込んでいます…", "Reading the file…"))
                let scoped = picked.startAccessingSecurityScopedResource()
                defer { if scoped { picked.stopAccessingSecurityScopedResource() } }
                if picked.pathExtension.lowercased() == "zip" {
                    let zip = workRoot.appendingPathComponent("source.zip")
                    try FileManager.default.copyItem(at: picked, to: zip)
                    try await Self.extract(zip, to: staging)
                } else {
                    try FileManager.default.copyItem(at: picked, to: staging.appendingPathComponent(picked.lastPathComponent))
                }
            }

            phase = .working(L("変わるところを調べています…", "Comparing files…"))
            guard let package = AppUpdater.packageRoot(in: staging) else { throw AppUpdateError.packageNotFound }
            let scoped = target.startAccessingSecurityScopedResource()
            defer { if scoped { target.stopAccessingSecurityScopedResource() } }
            let plan = try AppUpdater.plan(source: package, target: target)
            self.plan = plan
            stagedPackage = package
            phase = .ready(UpdatePlanSummary(added: plan.added, changed: plan.changed, removed: plan.removed,
                                             unchanged: plan.unchangedCount,
                                             from: plan.previousVersion?.description ?? "?",
                                             to: plan.newVersion.description))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    nonisolated private static func extract(_ zip: URL, to folder: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            _ = try ZipArchive(url: zip).extract(to: folder, inflate: ZipArchive.systemInflater)
        }.value
    }

    /// 上書きして版を上げる（変える前のファイルは取っておく）
    func apply() async {
        guard let plan, let package = stagedPackage,
              let data = defaults.data(forKey: Keys.target), let target = resolve(data) else { return }
        phase = .working(L("上書きしています…", "Updating…"))
        let backup = backupFolder
        do {
            try await Task.detached(priority: .userInitiated) {
                let scoped = target.startAccessingSecurityScopedResource()
                defer { if scoped { target.stopAccessingSecurityScopedResource() } }
                try Self.coordinatedWrite(at: target) {
                    try AppUpdater.apply(plan, source: package, target: target, backup: backup)
                }
            }.value
            hasBackup = true
            self.plan = nil
            phase = .done(L("\(plan.newVersion.description) に更新しました。Swift Playgrounds でこのアプリのプロジェクトを開き直して ▶︎ を押すと、新しい版が動きます。",
                            "Updated to \(plan.newVersion.description). Reopen this app's project in Swift Playgrounds and press ▶︎ to run the new version."))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// 前回の更新を取り消す
    func restoreLastUpdate() async {
        guard let data = defaults.data(forKey: Keys.target), let target = resolve(data) else { return }
        phase = .working(L("元に戻しています…", "Restoring…"))
        let backup = backupFolder
        do {
            try await Task.detached(priority: .userInitiated) {
                let scoped = target.startAccessingSecurityScopedResource()
                defer { if scoped { target.stopAccessingSecurityScopedResource() } }
                try Self.coordinatedWrite(at: target) {
                    try AppUpdater.restore(from: backup, target: target)
                }
                try? FileManager.default.removeItem(at: backup)
            }.value
            hasBackup = false
            phase = .done(L("前回の更新を取り消しました。Swift Playgrounds で開き直してください。",
                            "The last update was undone. Reopen the project in Swift Playgrounds."))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// iCloud Drive などのファイルを安全に書き換えるため、NSFileCoordinator を通す
    nonisolated private static func coordinatedWrite(at url: URL, _ work: () throws -> Void) throws {
        var coordinationError: NSError?
        var workError: Error?
        NSFileCoordinator().coordinate(writingItemAt: url, options: [], error: &coordinationError) { _ in
            do { try work() } catch { workError = error }
        }
        if let workError { throw workError }
        if let coordinationError { throw coordinationError }
    }

    private enum Keys {
        static let source = "updater.source"
        static let repository = "updater.repository"
        static let branch = "updater.branch"
        static let token = "updater.githubToken"
        static let target = "updater.targetBookmark"
    }
}
