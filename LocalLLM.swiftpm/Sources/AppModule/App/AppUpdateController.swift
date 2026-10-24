import Foundation
import Observation

/// アプリ自身の更新。GitHub（または選んだ ZIP・.swiftpm）とこの iPad の .swiftpm を 1 ファイルずつ比べて、
/// 変わったファイルだけをダウンロードして書き換え、版を上げる。書き換える前のファイルは残すので取り消せる。
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
        case working(String, Double?)
        /// 確かめたら同じだった（result に内容がある）
        case upToDate
        /// 違うファイルがある（result と selection を見る）
        case ready
        case applied(UpdateRecord)
        case restored(String)
        case failed(String)
    }

    enum TargetStatus: Equatable {
        case none
        case ok(PackageSummary)
        /// 選んだフォルダの中に、パッケージがいくつかあった（選んでもらう）
        case choosing(folder: String, candidates: [String])
        case problem(String)
    }

    /// 差分を見る画面に出すもの
    struct FilePreview: Equatable {
        var before: String
        var after: String
        var note: String?
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
    var checkAutomatically: Bool {
        didSet { defaults.set(checkAutomatically, forKey: Keys.autoCheck) }
    }
    /// 更新に含めるファイル（相対パス）
    var selection: Set<String> = []

    private(set) var targetStatus: TargetStatus = .none
    private(set) var sourceFileName: String?
    private(set) var phase: Phase = .idle
    private(set) var result: UpdateCheckResult?
    private(set) var history: [UpdateRecord] = []
    private(set) var branches: [String] = []
    private(set) var rateLimit: GitHubRateLimit?
    /// 起動したときの確認で見つかった、新しいコミット
    private(set) var availableUpdate: GitHubCommit?
    private(set) var lastChecked: Date?
    private(set) var cacheSize: Int64 = 0

    @ObservationIgnored private let defaults = UserDefaults.standard
    @ObservationIgnored private var pickedSource: URL?
    @ObservationIgnored private var pendingFolderBookmark: Data?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var operationID = UUID()

    /// アプリの版（実行中のもの）
    nonisolated static var runningVersion: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }

    private let historyStore = UpdateHistory(folder: UpdateFileOps.workFolder.appendingPathComponent("History", isDirectory: true))
    private let objects = UpdateObjectStore(root: UpdateFileOps.cacheFolder.appendingPathComponent("objects", isDirectory: true))

    init() {
        source = defaults.string(forKey: Keys.source).flatMap(Source.init(rawValue:)) ?? .github
        repository = defaults.string(forKey: Keys.repository) ?? "prak59459-create/LocalLLM"
        branch = defaults.string(forKey: Keys.branch) ?? "main"
        token = Keychain.get(Keys.token) ?? ""
        checkAutomatically = defaults.object(forKey: Keys.autoCheck) as? Bool ?? true
        history = historyStore.load()
        if let data = defaults.data(forKey: Keys.availableCommit) {
            availableUpdate = try? JSONDecoder().decode(GitHubCommit.self, from: data)
        }
    }

    // MARK: - 状態

    var targetSummary: PackageSummary? {
        if case .ok(let summary) = targetStatus { return summary }
        return nil
    }

    var isWorking: Bool {
        if case .working = phase { return true }
        return false
    }

    var canCheck: Bool {
        guard targetSummary != nil, !isWorking else { return false }
        switch source {
        case .github: return GitHubRepository(repository) != nil
        case .file: return pickedSource != nil
        }
    }

    var selectedChanges: [UpdateFileChange] {
        result?.diff.changes.filter { selection.contains($0.path) } ?? []
    }

    /// 選んだファイルで更新したときの版
    var plannedVersion: PackageVersion? { result?.newVersion() }

    /// 今取り消せる更新
    var restorable: UpdateRecord? {
        guard let name = targetSummary?.folderName else { return nil }
        return UpdateHistory.restorable(in: history, target: name)
    }

    /// この iPad のパッケージが、どのコミットと同じだと分かっているか
    private var knownCommit: String? {
        get { defaults.string(forKey: Keys.knownCommit) }
        set { defaults.set(newValue, forKey: Keys.knownCommit) }
    }

    // MARK: - 更新するアプリを選ぶ

    /// 「ファイル」で選んだもの（.swiftpm そのもの・それが入っているフォルダ）
    func pickTarget(_ url: URL) {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let bookmark: Data
        do {
            bookmark = try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
        } catch {
            targetStatus = .problem(L("「\(url.lastPathComponent)」を覚えておけませんでした: \(error.localizedDescription)",
                                      "Could not remember \"\(url.lastPathComponent)\": \(error.localizedDescription)"))
            return
        }
        switch UpdateTargetResolver.resolve(url) {
        case .package:
            saveTarget(bookmark: bookmark, subpath: "")
        case .candidates(let found) where found.count == 1:
            saveTarget(bookmark: bookmark, subpath: found[0])
        case .candidates(let found):
            pendingFolderBookmark = bookmark
            targetStatus = .choosing(folder: url.lastPathComponent, candidates: found)
        case .insidePackage(let package):
            targetStatus = .problem(L("「\(url.lastPathComponent)」は \(package) の中にあります。\(package) そのもの（または、それが入っている Playgrounds フォルダ）を選んでください。",
                                      "\"\(url.lastPathComponent)\" is inside \(package). Choose \(package) itself (or the Playgrounds folder that contains it)."))
        case .notFound:
            targetStatus = .problem(L("「\(url.lastPathComponent)」には Swift Playgrounds のアプリ（Package.swift と Sources のある .swiftpm）が見つかりませんでした。",
                                      "No Swift Playgrounds app (.swiftpm with Package.swift and Sources) was found in \"\(url.lastPathComponent)\"."))
        }
    }

    /// フォルダの中にいくつかあったパッケージから 1 つ選んだ
    func chooseCandidate(_ path: String) {
        guard let bookmark = pendingFolderBookmark else { return }
        pendingFolderBookmark = nil
        saveTarget(bookmark: bookmark, subpath: path)
    }

    func forgetTarget() {
        defaults.removeObject(forKey: Keys.targetBookmark)
        defaults.removeObject(forKey: Keys.targetSubpath)
        targetStatus = .none
        result = nil
        phase = .idle
    }

    private func saveTarget(bookmark: Data, subpath: String) {
        defaults.set(bookmark, forKey: Keys.targetBookmark)
        defaults.set(subpath, forKey: Keys.targetSubpath)
        // 別のパッケージに変えたら、前の確認の結果は使わない
        result = nil
        selection = []
        phase = .idle
        refreshTarget()
    }

    /// 覚えているパッケージを読み直す
    func refreshTarget() {
        guard defaults.data(forKey: Keys.targetBookmark) != nil else {
            if case .choosing = targetStatus { return }
            targetStatus = .none
            return
        }
        do {
            let location = try resolveTarget()
            let accessing = location.scoped.startAccessingSecurityScopedResource()
            defer { if accessing { location.scoped.stopAccessingSecurityScopedResource() } }
            guard PackageScanner.isPackage(location.target) else {
                targetStatus = .problem(AppUpdateError.notAPackage(location.target.lastPathComponent).localizedDescription)
                return
            }
            targetStatus = .ok(PackageSummary.read(location.target))
        } catch {
            targetStatus = .problem(L("前に選んだアプリが見つかりません（移動・名前の変更・削除をしたかもしれません）。もう一度選んでください。",
                                      "The app chosen before cannot be found (it may have been moved, renamed or deleted). Choose it again."))
        }
    }

    private func resolveTarget() throws -> (target: URL, scoped: URL, stale: Bool) {
        guard let bookmark = defaults.data(forKey: Keys.targetBookmark) else { throw AppUpdateError.packageNotFound }
        var stale = false
        let scoped = try URL(resolvingBookmarkData: bookmark, options: [], relativeTo: nil, bookmarkDataIsStale: &stale)
        let subpath = defaults.string(forKey: Keys.targetSubpath) ?? ""
        let target = subpath.isEmpty ? scoped : scoped.appendingPathComponent(subpath, isDirectory: true)
        return (target, scoped, stale)
    }

    /// 選んだパッケージを読み書きできる間に work を実行する
    private func withTarget<T>(_ work: (URL) async throws -> T) async throws -> T {
        let location = try resolveTarget()
        let accessing = location.scoped.startAccessingSecurityScopedResource()
        defer { if accessing { location.scoped.stopAccessingSecurityScopedResource() } }
        if location.stale, let fresh = try? location.scoped.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            defaults.set(fresh, forKey: Keys.targetBookmark)
        }
        guard PackageScanner.isPackage(location.target) else { throw AppUpdateError.notAPackage(location.target.lastPathComponent) }
        return try await work(location.target)
    }

    /// 新しいソース（.zip か .swiftpm・フォルダ）を選んだ
    func pickSourceFile(_ url: URL) {
        pickedSource = url
        sourceFileName = url.lastPathComponent
        result = nil
        phase = .idle
    }

    // MARK: - 確認

    func check() {
        operation?.cancel()
        let id = UUID()
        operationID = id
        phase = .working(L("準備しています…", "Preparing…"), nil)
        operation = Task { await runCheck(id: id) }
    }

    func cancel() {
        operation?.cancel()
        operationID = UUID()
        phase = .idle
    }

    private func runCheck(id: UUID) async {
        result = nil
        selection = []
        do {
            let checked = try await withTarget { target -> UpdateCheckResult in
                setWorking(L("iCloud のファイルを確かめています…", "Checking iCloud files…"), id: id)
                let missing = await UpdateFileOps.downloadPlaceholders(in: target)
                if !missing.isEmpty { throw UpdateCheckError.notAvailableLocally(missing) }
                let base = UpdateHistory.base(in: history, target: target.lastPathComponent)
                switch source {
                case .github: return try await checkGitHub(target: target, base: base, id: id)
                case .file: return try await checkFile(target: target, base: base, id: id)
                }
            }
            guard id == operationID else { return }
            result = checked
            selection = checked.diff.defaultSelection
            lastChecked = Date()
            refreshTarget()
            if checked.diff.isUpToDate {
                if let sha = checked.commit?.sha { knownCommit = sha }
                setAvailableUpdate(nil)
                phase = .upToDate
            } else {
                phase = .ready
            }
            cacheSize = UpdateFileOps.folderSize(objects.root)
        } catch is CancellationError {
            if id == operationID { phase = .idle }
        } catch {
            guard id == operationID else { return }
            phase = (error as? URLError)?.code == .cancelled ? .idle : .failed(error.localizedDescription)
        }
        if source == .github { await refreshRateLimit() }
    }

    private func checkGitHub(target: URL, base: UpdateRecord?, id: UUID) async throws -> UpdateCheckResult {
        guard let repo = GitHubRepository(repository) else { throw GitHubAPIError.invalidRepository(repository) }
        let client = GitHubClient(repository: repo, token: token)
        let ref = branch.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "main" : branch.trimmingCharacters(in: .whitespacesAndNewlines)
        return try await UpdateEngine.checkGitHub(client: client, ref: ref, target: target, objects: objects, base: base) { stage in
            Task { @MainActor in self.setWorking(stage.message, fraction: stage.fraction, id: id) }
        }
    }

    private func checkFile(target: URL, base: UpdateRecord?, id: UUID) async throws -> UpdateCheckResult {
        guard let picked = pickedSource else { throw AppUpdateError.packageNotFound }
        let accessing = picked.startAccessingSecurityScopedResource()
        defer { if accessing { picked.stopAccessingSecurityScopedResource() } }
        var root = picked
        if picked.pathExtension.lowercased() == "zip" {
            setWorking(L("ZIP を展開しています…", "Extracting the ZIP…"), id: id)
            let staging = UpdateFileOps.cacheFolder.appendingPathComponent("staging", isDirectory: true)
            try? FileManager.default.removeItem(at: staging)
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            let zip = staging.appendingPathComponent("source.zip")
            try FileManager.default.copyItem(at: picked, to: zip)
            root = staging.appendingPathComponent("extracted", isDirectory: true)
            try await UpdateFileOps.extract(zip, to: root)
        }
        guard let package = UpdateTargetResolver.findPackage(in: root, preferredNames: [target.lastPathComponent, "LocalLLM.swiftpm"]) else {
            throw AppUpdateError.packageNotFound
        }
        guard package.standardizedFileURL.path != target.standardizedFileURL.path else {
            throw AppUpdateError.notAPackage(L("更新するアプリと同じもの", "the app being updated"))
        }
        setWorking(UpdateStage.comparing.message, id: id)
        let label = picked.lastPathComponent
        let objects = self.objects
        return try await Task.detached(priority: .userInitiated) {
            try UpdateEngine.checkFolder(package, label: label, target: target, objects: objects, base: base)
        }.value
    }

    private func setWorking(_ message: String, fraction: Double? = nil, id: UUID) {
        // 終わったあとに届いた途中経過では、結果の表示を上書きしない
        guard id == operationID, case .working = phase else { return }
        phase = .working(message, fraction)
    }

    // MARK: - 選ぶ

    func selectRecommended() {
        selection = result?.diff.defaultSelection ?? []
    }

    func selectAll() {
        selection = Set(result?.diff.changes.map(\.path) ?? [])
    }

    func selectNone() {
        selection = []
    }

    func setSelected(_ path: String, _ isOn: Bool) {
        if isOn { selection.insert(path) } else { selection.remove(path) }
    }

    // MARK: - 更新・取り消し

    func apply() {
        guard let result, !selectedChanges.isEmpty, !isWorking else { return }
        let request = result.request(selection: selection)
        let id = UUID()
        operationID = id
        phase = .working(L("ファイルを書き換えています…", "Writing files…"), nil)
        operation = Task { await runApply(request, id: id) }
    }

    private func runApply(_ request: UpdateRequest, id: UUID) async {
        let objects = self.objects
        let history = historyStore
        do {
            let record = try await withTarget { target -> UpdateRecord in
                try await Task.detached(priority: .userInitiated) {
                    try UpdateFileOps.coordinatedWrite(at: target) { root in
                        try AppUpdater.apply(request, target: root, objects: objects, history: history)
                    }
                }.value
            }
            reloadHistory()
            if let sha = record.commit?.sha { knownCommit = sha }
            setAvailableUpdate(nil)
            result = nil
            selection = []
            refreshTarget()
            phase = .applied(record)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func restoreLatest() {
        guard let record = restorable, !isWorking else { return }
        let id = UUID()
        operationID = id
        phase = .working(L("元に戻しています…", "Restoring…"), nil)
        operation = Task { await runRestore(record) }
    }

    private func runRestore(_ record: UpdateRecord) async {
        let history = historyStore
        do {
            try await withTarget { target in
                try await Task.detached(priority: .userInitiated) {
                    try UpdateFileOps.coordinatedWrite(at: target) { root in
                        try AppUpdater.restore(record, target: root, history: history)
                    }
                }.value
            }
            reloadHistory()
            knownCommit = UpdateHistory.base(in: self.history, target: record.targetName)?.commit?.sha
            result = nil
            selection = []
            refreshTarget()
            let previous = record.previousVersion?.description ?? "?"
            phase = .restored(L("\(record.newVersion.description) への更新を取り消して、\(previous) に戻しました。Swift Playgrounds でプロジェクトを開き直してください。",
                                "Undid the update to \(record.newVersion.description) and went back to \(previous). Reopen the project in Swift Playgrounds."))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func reloadHistory() {
        history = historyStore.load()
    }

    // MARK: - 差分を見る

    func preview(_ change: UpdateFileChange) async -> FilePreview {
        guard let result else { return FilePreview(before: "", after: "", note: nil) }
        let objects = self.objects
        do {
            return try await withTarget { target -> FilePreview in
                let before = change.localSHA == nil ? Data() : (try? Data(contentsOf: target.appendingPathComponent(change.path))) ?? Data()
                var after = Data()
                if change.isPackageManifest, let remote = result.remotePackageSwift {
                    let merged = PackageManifest.merge(local: result.localPackageSwift, remote: remote,
                                                       base: result.base?.remotePackageSwift, version: result.newVersion())
                    after = Data(merged.utf8)
                } else if let sha = change.remoteSHA, change.kind.writesRemoteFile {
                    after = objects.data(for: sha) ?? Data()
                }
                return Self.makePreview(before: before, after: after)
            }
        } catch {
            return FilePreview(before: "", after: "", note: error.localizedDescription)
        }
    }

    private nonisolated static func makePreview(before: Data, after: Data) -> FilePreview {
        let limit = 2_000_000
        if before.count > limit || after.count > limit {
            return FilePreview(before: "", after: "", note: L("大きいファイルなので、差分は表示しません", "The file is too large to show a diff"))
        }
        guard let old = String(data: before, encoding: .utf8), let new = String(data: after, encoding: .utf8) else {
            let size = ByteCountFormatter.string(fromByteCount: Int64(after.count), countStyle: .file)
            return FilePreview(before: "", after: "", note: L("テキストではないファイルです（\(size)）", "Not a text file (\(size))"))
        }
        return FilePreview(before: old, after: new, note: nil)
    }

    // MARK: - GitHub

    func loadBranches() async {
        guard let repo = GitHubRepository(repository) else { return }
        do {
            let names = try await GitHubClient(repository: repo, token: token).branches()
            branches = names.sorted { lhs, rhs in
                let a = lhs == "main" || lhs == "master", b = rhs == "main" || rhs == "master"
                return a != b ? a : lhs.localizedStandardCompare(rhs) == .orderedAscending
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func refreshRateLimit() async {
        guard let repo = GitHubRepository(repository) else { return }
        rateLimit = try? await GitHubClient(repository: repo, token: token).rateLimit()
    }

    /// 起動したときに、新しいコミットがあるかだけを軽く確かめる（3 時間に 1 回まで）
    func checkInBackgroundIfNeeded() async {
        guard checkAutomatically, source == .github, let repo = GitHubRepository(repository) else { return }
        if let last = defaults.object(forKey: Keys.lastBackgroundCheck) as? Date, Date().timeIntervalSince(last) < 3 * 3600 { return }
        defaults.set(Date(), forKey: Keys.lastBackgroundCheck)
        let client = GitHubClient(repository: repo, token: token)
        let ref = branch.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "main" : branch.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            let etag = defaults.string(forKey: Keys.etag)
            let fetched = try await client.latestCommit(ref: ref, etag: etag)
            var commit = fetched.commit
            if let commit, let data = try? JSONEncoder().encode(commit) {
                defaults.set(data, forKey: Keys.remoteCommit)
                defaults.set(fetched.etag, forKey: Keys.etag)
            } else if let data = defaults.data(forKey: Keys.remoteCommit) {
                commit = try? JSONDecoder().decode(GitHubCommit.self, from: data)
            }
            guard let commit else { return }
            if let known = knownCommit {
                setAvailableUpdate(known == commit.sha ? nil : commit)
                return
            }
            // まだ一度も確かめていない: GitHub の Package.swift の版と、動いている版を比べる
            let folder = targetSummary?.folderName ?? "LocalLLM.swiftpm"
            let (data, response) = try await URLSession.shared.data(for: GitHubAPI.fileRequest(repo, commit: commit.sha,
                                                                                               path: folder + "/Package.swift", token: token))
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let remote = PackageVersion.parse(packageSwift: String(decoding: data, as: UTF8.self)),
                  let running = PackageVersion.parse(description: Self.runningVersion) else { return }
            let newer = PackageVersion.compare(remote.display, running.display) == .orderedDescending
                || (PackageVersion.compare(remote.display, running.display) == .orderedSame && remote.bundle > running.bundle)
            setAvailableUpdate(newer ? commit : nil)
        } catch {
            // 起動時の確認は、失敗しても何もしない
        }
    }

    private func setAvailableUpdate(_ commit: GitHubCommit?) {
        availableUpdate = commit
        if let commit, let data = try? JSONEncoder().encode(commit) {
            defaults.set(data, forKey: Keys.availableCommit)
        } else {
            defaults.removeObject(forKey: Keys.availableCommit)
        }
    }

    // MARK: - 後片付け

    func refreshCacheSize() {
        cacheSize = UpdateFileOps.folderSize(objects.root)
    }

    func clearCache() {
        objects.removeAll()
        try? FileManager.default.removeItem(at: UpdateFileOps.cacheFolder.appendingPathComponent("staging"))
        result = nil
        selection = []
        if case .ready = phase { phase = .idle }
        refreshCacheSize()
    }

    private enum Keys {
        static let source = "updater.source"
        static let repository = "updater.repository"
        static let branch = "updater.branch"
        static let token = "updater.githubToken"
        static let autoCheck = "updater.autoCheck"
        static let targetBookmark = "updater.targetBookmark"
        static let targetSubpath = "updater.targetSubpath"
        static let knownCommit = "updater.knownCommit"
        static let availableCommit = "updater.availableCommit"
        static let remoteCommit = "updater.remoteCommit"
        static let etag = "updater.commitETag"
        static let lastBackgroundCheck = "updater.lastBackgroundCheck"
    }
}

/// 更新のファイル操作（メインスレッドの外で動かすので、クラスの外に置く）。
enum UpdateFileOps {
    /// 履歴とバックアップの置き場所
    static var workFolder: URL {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("AppUpdate", isDirectory: true)
    }

    /// ダウンロードしたファイルの置き場所（消えてもまた取ってくればよいもの）
    static var cacheFolder: URL {
        let base = (try? FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("AppUpdate", isDirectory: true)
    }

    static func extract(_ zip: URL, to folder: URL) async throws {
        let task = Task.detached(priority: .userInitiated) { () throws -> Void in
            _ = try ZipArchive(url: zip).extract(to: folder, inflate: ZipArchive.systemInflater)
        }
        try await task.value
    }

    /// iCloud Drive などのファイルを安全に書き換えるため、NSFileCoordinator を通す（Swift Playgrounds にも書き換えたことが伝わる）
    static func coordinatedWrite<T>(at url: URL, _ work: (URL) throws -> T) throws -> T {
        try withoutActuallyEscaping(work) { work in
            var coordinationError: NSError?
            var outcome: Result<T, Error>?
            // 書き込みの中では例外を投げない（結果を取っておいて、外で投げる）
            NSFileCoordinator().coordinate(writingItemAt: url, options: [], error: &coordinationError) { granted in
                outcome = Result { try work(granted) }
            }
            if let coordinationError { throw coordinationError }
            guard let outcome else { throw CocoaError(.fileWriteUnknown) }
            return try outcome.get()
        }
    }

    /// iCloud にあって、まだこの iPad に落ちていないファイルをダウンロードする。落ちてこなかったものを返す。
    static func downloadPlaceholders(in root: URL, timeout: TimeInterval = 60) async -> [String] {
        var waiting = PackageScanner.relativePaths(in: root).placeholders
        guard !waiting.isEmpty else { return [] }
        for path in waiting {
            try? FileManager.default.startDownloadingUbiquitousItem(at: root.appendingPathComponent(path))
        }
        let deadline = Date().addingTimeInterval(timeout)
        while !waiting.isEmpty, Date() < deadline {
            try? await Task.sleep(nanoseconds: 500_000_000)
            if Task.isCancelled { break }
            waiting = PackageScanner.relativePaths(in: root).placeholders
        }
        return waiting
    }

    static func folderSize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }
}
