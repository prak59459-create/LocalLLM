import Foundation

/// Package.swift に書かれたアプリの版（displayVersion と bundleVersion）。
struct PackageVersion: Codable, Equatable, Hashable, Sendable, CustomStringConvertible {
    var display: String
    var bundle: Int

    var description: String { "\(display) (\(bundle))" }

    /// Package.swift の文字列から読む
    static func parse(packageSwift text: String) -> PackageVersion? {
        guard let display = PackageManifest.stringValue(of: "displayVersion", in: text) else { return nil }
        let bundle = PackageManifest.stringValue(of: "bundleVersion", in: text).flatMap { Int($0) } ?? 1
        return PackageVersion(display: display, bundle: bundle)
    }

    /// "0.3 (3)" のような表示から読む
    static func parse(description text: String) -> PackageVersion? {
        let parts = text.split(separator: " ")
        guard let display = parts.first.map(String.init), !display.isEmpty else { return nil }
        let bundle = parts.dropFirst().first.map { $0.filter(\.isNumber) }.flatMap { Int($0) } ?? 1
        return PackageVersion(display: display, bundle: bundle)
    }

    /// "1.2.10" と "1.2.9" のような版の比べ方（数字ごとに比べる）
    static func compare(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let a = lhs.split(separator: ".").map { Int($0.filter(\.isNumber)) ?? 0 }
        let b = rhs.split(separator: ".").map { Int($0.filter(\.isNumber)) ?? 0 }
        for index in 0..<max(a.count, b.count) {
            let x = index < a.count ? a[index] : 0
            let y = index < b.count ? b[index] : 0
            if x != y { return x < y ? .orderedAscending : .orderedDescending }
        }
        return .orderedSame
    }

    /// 最後の数字を 1 つ上げる（"0.2" → "0.2.1"、"0.2.1" → "0.2.2"）
    static func bumpPatch(_ version: String) -> String {
        var parts = version.split(separator: ".").map(String.init)
        if parts.count < 3 {
            while parts.count < 2 { parts.append("0") }
            parts.append("1")
        } else if let last = Int(parts[parts.count - 1].filter(\.isNumber)) {
            parts[parts.count - 1] = String(last + 1)
        } else {
            parts.append("1")
        }
        return parts.joined(separator: ".")
    }

    /// 更新したあとの版。bundleVersion は毎回 1 つ上げる。
    /// 新しいソースの displayVersion の方が大きければそれを使い、そうでなければ今の版の最後の数字を上げる。
    static func next(current: PackageVersion?, incoming: PackageVersion?) -> PackageVersion {
        let bundle = max(current?.bundle ?? 0, incoming?.bundle ?? 0) + 1
        let display: String
        switch (current, incoming) {
        case (let current?, let incoming?):
            display = compare(incoming.display, current.display) == .orderedDescending ? incoming.display : bumpPatch(current.display)
        case (let current?, nil):
            display = bumpPatch(current.display)
        case (nil, let incoming?):
            display = incoming.display
        case (nil, nil):
            display = "1.0"
        }
        return PackageVersion(display: display, bundle: bundle)
    }

    /// Package.swift の版を書き換えた文字列
    func apply(to text: String) -> String {
        var result = PackageManifest.replacing("displayVersion", withExpression: "\"\(display)\"", in: text)
        result = PackageManifest.replacing("bundleVersion", withExpression: "\"\(bundle)\"", in: result)
        return result
    }
}

enum AppUpdateError: Error, LocalizedError, Equatable {
    case packageNotFound
    case notAPackage(String)
    case download(Int)
    case unsafePath(String)
    case localFilesChanged([String])
    case missingDownload(String)
    case verificationFailed([String])
    case nothingToRestore

    var errorDescription: String? {
        switch self {
        case .packageNotFound:
            return L("新しいソースの中に、アプリのパッケージ（Package.swift と Sources/ のあるフォルダ）が見つかりません。",
                     "No app package (a folder with Package.swift and Sources/) was found in the new source.")
        case .notAPackage(let name):
            return L("「\(name)」は Swift Playgrounds のアプリ（Package.swift と Sources のある .swiftpm）ではないようです。",
                     "\"\(name)\" does not look like a Swift Playgrounds app (.swiftpm with Package.swift and Sources).")
        case .download(let status):
            return L("ダウンロードに失敗しました（HTTP \(status)）。リポジトリ名・ブランチ・トークンを確かめてください。",
                     "Download failed (HTTP \(status)). Check the repository, branch and token.")
        case .unsafePath(let path):
            return L("パッケージの外を指すファイル名なので更新できません: \(path)", "Refusing a path outside the package: \(path)")
        case .localFilesChanged(let paths):
            return L("確認したあとに、この iPad のファイルが変わりました（\(paths.prefix(3).joined(separator: ", "))）。もう一度「更新を確認」してください。",
                     "Files on this iPad changed after the check (\(paths.prefix(3).joined(separator: ", "))). Check for updates again.")
        case .missingDownload(let path):
            return L("\(path) のダウンロードが見つかりません。もう一度「更新を確認」してください。",
                     "The download for \(path) is missing. Check for updates again.")
        case .verificationFailed(let paths):
            return L("書いたファイルが GitHub の内容と一致しなかったので、元に戻しました（\(paths.prefix(3).joined(separator: ", "))）。",
                     "Written files did not match GitHub, so the update was rolled back (\(paths.prefix(3).joined(separator: ", "))).")
        case .nothingToRestore:
            return L("取り消せる更新がありません。", "There is no update to undo.")
        }
    }
}

/// 取ってきたファイルの置き場所（git の識別子ごとに 1 つ。同じ中身は 2 回ダウンロードしない）。
struct UpdateObjectStore: Sendable {
    var root: URL

    func url(for sha: String) -> URL {
        root.appendingPathComponent(String(sha.prefix(2)), isDirectory: true).appendingPathComponent(sha)
    }

    /// 置いてあって、中身が識別子と合っているか
    func contains(_ sha: String) -> Bool {
        guard let data = try? Data(contentsOf: url(for: sha), options: .mappedIfSafe) else { return false }
        return GitBlob.sha(of: data) == sha
    }

    func data(for sha: String) -> Data? {
        guard let data = try? Data(contentsOf: url(for: sha)), GitBlob.sha(of: data) == sha else { return nil }
        return data
    }

    /// 中身を確かめてから置く
    func store(_ data: Data, sha: String) throws {
        guard GitBlob.sha(of: data) == sha else { throw GitHubAPIError.integrity(sha) }
        let destination = url(for: sha)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: destination, options: .atomic)
    }

    func removeAll() {
        try? FileManager.default.removeItem(at: root)
    }
}

/// 更新 1 回の記録（取り消すのにも使う）。
struct UpdateRecord: Codable, Identifiable, Hashable, Sendable {
    var id: UUID
    var date: Date
    /// "GitHub owner/name (main)" やファイル名
    var sourceLabel: String
    var repository: String?
    var branch: String?
    var commit: GitHubCommit?
    var targetName: String
    var previousVersion: PackageVersion?
    var newVersion: PackageVersion
    /// 書いたファイル（元から無かったものは created にも入る）
    var written: [String]
    var created: [String]
    var deleted: [String]
    /// 更新したあとのパッケージの中身が、新しいソースのどの内容と同じか（次の更新で「この iPad でだけ変えたファイル」を見分ける）
    var manifest: [String: String]
    /// 前回書いた GitHub の Package.swift（この iPad の設定を残すのに使う）
    var remotePackageSwift: String?
    var restored: Bool = false

    var changedFileCount: Int { Set(written + deleted).subtracting(["Package.swift"]).count }
}

/// 更新の履歴（新しい順）。それぞれの更新で書き換える前のファイルを残しておき、新しい方から順に取り消せる。
struct UpdateHistory: Sendable {
    var folder: URL
    var limit = 10

    private var historyURL: URL { folder.appendingPathComponent("history.json") }
    func backupFolder(for id: UUID) -> URL { folder.appendingPathComponent("backups/\(id.uuidString)", isDirectory: true) }

    func load() -> [UpdateRecord] {
        guard let data = try? Data(contentsOf: historyURL) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([UpdateRecord].self, from: data)) ?? []
    }

    func save(_ records: [UpdateRecord]) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(records).write(to: historyURL, options: .atomic)
    }

    /// 記録を先頭に足し、古いもの（とそのバックアップ）を消す
    func add(_ record: UpdateRecord) throws {
        var records = load().filter { $0.id != record.id }
        records.insert(record, at: 0)
        for old in records.dropFirst(limit) {
            try? FileManager.default.removeItem(at: backupFolder(for: old.id))
        }
        try save(Array(records.prefix(limit)))
    }

    func markRestored(_ id: UUID) throws {
        var records = load()
        guard let index = records.firstIndex(where: { $0.id == id }) else { return }
        records[index].restored = true
        try save(records)
        try? FileManager.default.removeItem(at: backupFolder(for: id))
    }

    /// 今取り消せる更新（取り消していないもののうち一番新しいもの）
    static func restorable(in records: [UpdateRecord], target: String) -> UpdateRecord? {
        records.first { !$0.restored && $0.targetName == target }
    }

    /// 今のパッケージが、どの新しいソースの内容から来ているか（一番新しい、取り消していない更新）
    static func base(in records: [UpdateRecord], target: String) -> UpdateRecord? {
        restorable(in: records, target: target)
    }
}

/// 更新の中身（書くファイルと消すファイル）。
struct UpdateRequest: Sendable {
    /// 書くファイル: 相対パス → 中身の識別子（UpdateObjectStore に置いてあるもの）
    var writes: [String: String]
    var deletes: [String]
    /// 確認したときの、この iPad のファイルの識別子（nil = 無かった）。書く前に変わっていないか確かめる。
    var expectedLocal: [String: String?]
    /// 新しいソースの Package.swift（分かっていれば。次の更新でこの iPad の設定を残すのに使う）
    var remotePackageSwift: String?
    /// Package.swift を新しいソースのものにする（false なら今の Package.swift の版だけ書き換える）
    var takeRemotePackageSwift: Bool
    /// 前回の更新で書いた新しいソースの Package.swift
    var basePackageSwift: String?
    var newVersion: PackageVersion
    /// 更新したあとのパッケージの中身（記録用）
    var manifest: [String: String]
    var sourceLabel: String
    var repository: String?
    var branch: String?
    var commit: GitHubCommit?
}

/// 更新を書き込む・取り消す。途中で失敗したら、書き換える前に戻す。
enum AppUpdater {
    /// 更新を書き込む。書き換える前のファイルは history のバックアップに残す。
    static func apply(_ request: UpdateRequest, target: URL, objects: UpdateObjectStore, history: UpdateHistory,
                      id: UUID = UUID(), date: Date = Date()) throws -> UpdateRecord {
        let fileManager = FileManager.default
        let packageURL = target.appendingPathComponent("Package.swift")
        guard PackageScanner.isPackage(target), fileManager.fileExists(atPath: packageURL.path) else {
            throw AppUpdateError.notAPackage(target.lastPathComponent)
        }
        let allPaths = Array(request.writes.keys) + request.deletes
        if let bad = allPaths.first(where: { !PackageScanner.isSafeRelativePath($0) || PackageScanner.isIgnored($0) }) {
            throw AppUpdateError.unsafePath(bad)
        }
        // 確認したあとに、この iPad で変わったファイルがあれば止める
        let changedSinceCheck = request.expectedLocal.compactMap { path, expected -> String? in
            PackageScanner.sha(of: path, in: target) == expected ? nil : path
        }
        if !changedSinceCheck.isEmpty { throw AppUpdateError.localFilesChanged(changedSinceCheck.sorted()) }
        // 書くファイルが全部そろっているか
        var contents: [String: Data] = [:]
        for (path, sha) in request.writes where path != "Package.swift" {
            guard let data = objects.data(for: sha) else { throw AppUpdateError.missingDownload(path) }
            contents[path] = data
        }

        let localPackageSwift = try String(contentsOf: packageURL, encoding: .utf8)
        let newPackageSwift: String
        if request.takeRemotePackageSwift, let remote = request.remotePackageSwift {
            newPackageSwift = PackageManifest.merge(local: localPackageSwift, remote: remote, base: request.basePackageSwift,
                                                    version: request.newVersion)
        } else {
            newPackageSwift = request.newVersion.apply(to: localPackageSwift)
        }

        // バックアップ
        let backup = history.backupFolder(for: id)
        try? fileManager.removeItem(at: backup)
        let backupFiles = backup.appendingPathComponent("files", isDirectory: true)
        try fileManager.createDirectory(at: backupFiles, withIntermediateDirectories: true)
        let touched = Set(allPaths + ["Package.swift"])
        var created: [String] = []
        for path in touched.sorted() {
            let original = target.appendingPathComponent(path)
            guard fileManager.fileExists(atPath: original.path) else {
                if request.writes[path] != nil { created.append(path) }
                continue
            }
            let copy = backupFiles.appendingPathComponent(path)
            try fileManager.createDirectory(at: copy.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fileManager.copyItem(at: original, to: copy)
        }
        let record = UpdateRecord(id: id, date: date, sourceLabel: request.sourceLabel, repository: request.repository,
                                  branch: request.branch, commit: request.commit, targetName: target.lastPathComponent,
                                  previousVersion: PackageVersion.parse(packageSwift: localPackageSwift),
                                  newVersion: request.newVersion,
                                  written: Array(Set(request.writes.keys).union(["Package.swift"])).sorted(),
                                  created: created, deleted: request.deletes.sorted(), manifest: request.manifest,
                                  remotePackageSwift: request.remotePackageSwift ?? request.basePackageSwift)

        // 書き込む（失敗したら元に戻す）
        do {
            for (path, data) in contents.sorted(by: { $0.key < $1.key }) {
                let destination = target.appendingPathComponent(path)
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: destination, options: .atomic)
            }
            try Data(newPackageSwift.utf8).write(to: packageURL, options: .atomic)
            for path in request.deletes {
                let url = target.appendingPathComponent(path)
                if fileManager.fileExists(atPath: url.path) { try fileManager.removeItem(at: url) }
            }
            removeEmptyFolders(in: target.appendingPathComponent("Sources"))

            // 書いたものが GitHub の中身と同じか確かめる
            let mismatched = contents.keys.filter { path in
                PackageScanner.sha(of: path, in: target) != request.writes[path]
            } + request.deletes.filter { fileManager.fileExists(atPath: target.appendingPathComponent($0).path) }
            if !mismatched.isEmpty { throw AppUpdateError.verificationFailed(mismatched.sorted()) }
        } catch {
            rollBack(record, target: target, backupFiles: backupFiles)
            try? fileManager.removeItem(at: backup)
            throw error
        }
        try history.add(record)
        return record
    }

    /// 更新を取り消す（バックアップから戻して、新しく足したファイルを消す）
    static func restore(_ record: UpdateRecord, target: URL, history: UpdateHistory) throws {
        let backupFiles = history.backupFolder(for: record.id).appendingPathComponent("files", isDirectory: true)
        guard FileManager.default.fileExists(atPath: backupFiles.path) else { throw AppUpdateError.nothingToRestore }
        try restoreFiles(record, target: target, backupFiles: backupFiles)
        try history.markRestored(record.id)
    }

    private static func rollBack(_ record: UpdateRecord, target: URL, backupFiles: URL) {
        try? restoreFiles(record, target: target, backupFiles: backupFiles)
    }

    private static func restoreFiles(_ record: UpdateRecord, target: URL, backupFiles: URL) throws {
        let fileManager = FileManager.default
        for path in record.created where PackageScanner.isSafeRelativePath(path) {
            try? fileManager.removeItem(at: target.appendingPathComponent(path))
        }
        for path in Set(record.written + record.deleted).sorted() where PackageScanner.isSafeRelativePath(path) {
            let copy = backupFiles.appendingPathComponent(path)
            guard fileManager.fileExists(atPath: copy.path) else { continue }
            let destination = target.appendingPathComponent(path)
            try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(contentsOf: copy).write(to: destination, options: .atomic)
        }
        removeEmptyFolders(in: target.appendingPathComponent("Sources"))
    }

    static func removeEmptyFolders(in folder: URL) {
        let fileManager = FileManager.default
        guard let items = try? fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        for item in items where (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
            removeEmptyFolders(in: item)
            let rest = (try? fileManager.contentsOfDirectory(atPath: item.path)) ?? ["?"]
            if rest.allSatisfy({ $0 == ".DS_Store" }) {
                try? fileManager.removeItem(at: item)
            }
        }
    }
}
