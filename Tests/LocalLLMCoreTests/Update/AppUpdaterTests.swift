import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

/// テスト用のフォルダとファイル
private enum Scratch {
    static func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("upd-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func write(_ text: String, _ path: String, in root: URL) throws {
        let url = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    static func read(_ path: String, in root: URL) -> String? {
        try? String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }

    static func manifest(version: String, bundle: Int, team: String = "", extra: String = "") -> String {
        """
        let package = Package(
            name: "LocalLLM",
            products: [
                .iOSApplication(
                    name: "LocalLLM",
                    bundleIdentifier: "com.example.LocalLLM",
                    teamIdentifier: "\(team)",
                    displayVersion: "\(version)",
                    bundleVersion: "\(bundle)",\(extra)
                    appCategory: .productivity
                )
            ]
        )
        """
    }
}

final class PackageScannerTests: XCTestCase {
    func testIndexIgnoresPlaygroundsSettingsAndFindsICloudPlaceholders() throws {
        let root = try Scratch.folder().appendingPathComponent("LocalLLM.swiftpm")
        try Scratch.write("pkg", "Package.swift", in: root)
        try Scratch.write("hello\n", "Sources/AppModule/A.swift", in: root)
        try Scratch.write("x", ".swiftpm/playgrounds/settings.plist", in: root)
        try Scratch.write("x", "Sources/AppModule/.DS_Store", in: root)
        try Scratch.write("", "Sources/AppModule/.B.swift.icloud", in: root)
        let index = PackageScanner.index(of: root)
        XCTAssertEqual(Set(index.files.keys), ["Package.swift", "Sources/AppModule/A.swift"])
        XCTAssertEqual(index.files["Sources/AppModule/A.swift"]?.sha, "ce013625030ba8dba906f756967f9e9ca394464a")
        XCTAssertEqual(index.placeholders, ["Sources/AppModule/B.swift"])
        XCTAssertThrowsError(try UpdateEngine.scanLocal(root)) {
            XCTAssertEqual($0 as? UpdateCheckError, .notAvailableLocally(["Sources/AppModule/B.swift"]))
        }
        XCTAssertEqual(PackageScanner.iCloudPlaceholderName(".A.swift.icloud"), "A.swift")
        XCTAssertNil(PackageScanner.iCloudPlaceholderName(".icloud"))
        XCTAssertNil(PackageScanner.iCloudPlaceholderName("A.swift"))
    }

    func testSafePaths() {
        XCTAssertTrue(PackageScanner.isSafeRelativePath("Sources/AppModule/A.swift"))
        for bad in ["", "/etc/passwd", "../x", "Sources/../../x", "Sources//A.swift", "Sources/./A.swift", "a\\b"] {
            XCTAssertFalse(PackageScanner.isSafeRelativePath(bad), bad)
        }
        XCTAssertTrue(PackageScanner.isIgnored(".swiftpm/x"))
        XCTAssertTrue(PackageScanner.isIgnored("Sources/.build/x"))
        XCTAssertFalse(PackageScanner.isIgnored("Sources/A.swift"))
    }

    func testTargetResolution() throws {
        let playgrounds = try Scratch.folder().appendingPathComponent("Playgrounds")
        let app = playgrounds.appendingPathComponent("LocalLLM.swiftpm")
        try Scratch.write("pkg", "Package.swift", in: app)
        try Scratch.write("a", "Sources/AppModule/App/LocalLLMApp.swift", in: app)
        let other = playgrounds.appendingPathComponent("Another.swiftpm")
        try Scratch.write("pkg", "Package.swift", in: other)
        try Scratch.write("a", "Sources/Main.swift", in: other)
        try FileManager.default.createDirectory(at: playgrounds.appendingPathComponent("Empty"), withIntermediateDirectories: true)

        XCTAssertEqual(UpdateTargetResolver.resolve(app), .package)
        XCTAssertEqual(UpdateTargetResolver.resolve(playgrounds), .candidates(["LocalLLM.swiftpm", "Another.swiftpm"]))
        XCTAssertEqual(UpdateTargetResolver.resolve(app.appendingPathComponent("Package.swift")), .insidePackage("LocalLLM.swiftpm"))
        XCTAssertEqual(UpdateTargetResolver.resolve(app.appendingPathComponent("Sources/AppModule")), .insidePackage("LocalLLM.swiftpm"))
        XCTAssertEqual(UpdateTargetResolver.resolve(playgrounds.appendingPathComponent("Empty")), .notFound)

        let summary = PackageSummary.read(app)
        XCTAssertTrue(summary.looksLikeThisApp)
        XCTAssertEqual(summary.swiftFileCount, 1)
        XCTAssertFalse(PackageSummary.read(other).looksLikeThisApp)
    }

    func testFindsThePackageInsideAGitHubZip() throws {
        let extracted = try Scratch.folder()
        let repo = extracted.appendingPathComponent("LocalLLM-main")
        try Scratch.write("dev package", "Package.swift", in: repo)
        try Scratch.write("x", "Tests/LocalLLMCoreTests/A.swift", in: repo)
        try Scratch.write("pkg", "LocalLLM.swiftpm/Package.swift", in: repo)
        try Scratch.write("a", "LocalLLM.swiftpm/Sources/AppModule/A.swift", in: repo)
        try Scratch.write("pkg", "Examples/Other.swiftpm/Package.swift", in: repo)
        try Scratch.write("a", "Examples/Other.swiftpm/Sources/A.swift", in: repo)
        XCTAssertEqual(UpdateTargetResolver.findPackage(in: extracted, preferredNames: ["LocalLLM.swiftpm"])?.lastPathComponent, "LocalLLM.swiftpm")
        XCTAssertEqual(UpdateTargetResolver.findPackage(in: extracted, preferredNames: ["Other.swiftpm"])?.lastPathComponent, "Other.swiftpm")
        let package = repo.appendingPathComponent("LocalLLM.swiftpm")
        XCTAssertEqual(UpdateTargetResolver.findPackage(in: package, preferredNames: [])?.path, package.path)
        XCTAssertNil(UpdateTargetResolver.findPackage(in: repo.appendingPathComponent("Tests"), preferredNames: []))
    }

    func testLocationDescription() {
        let iCloud = URL(fileURLWithPath: "/private/var/mobile/Library/Mobile Documents/iCloud~com~apple~Playgrounds/Documents/LocalLLM.swiftpm")
        XCTAssertEqual(PackageSummary.locationDescription(iCloud), "iCloud Drive › Playgrounds › LocalLLM.swiftpm")
        let drive = URL(fileURLWithPath: "/private/var/mobile/Library/Mobile Documents/com~apple~CloudDocs/Dev/LocalLLM.swiftpm")
        XCTAssertEqual(PackageSummary.locationDescription(drive), "iCloud Drive › Dev › LocalLLM.swiftpm")
    }
}

final class AppUpdaterTests: XCTestCase {
    /// この iPad のパッケージ（target）と、新しいソース（source）
    private func makePackages() throws -> (target: URL, source: URL, work: URL) {
        let work = try Scratch.folder()
        let target = work.appendingPathComponent("Playgrounds/LocalLLM.swiftpm")
        try Scratch.write(Scratch.manifest(version: "0.3", bundle: 3, team: "TEAM123"), "Package.swift", in: target)
        try Scratch.write("same", "Sources/AppModule/Same.swift", in: target)
        try Scratch.write("old", "Sources/AppModule/Changed.swift", in: target)
        try Scratch.write("gone", "Sources/AppModule/Old/Removed.swift", in: target)
        try Scratch.write("settings", ".swiftpm/playgrounds/settings.plist", in: target)
        try Scratch.write("keep", "Notes.txt", in: target)

        let source = work.appendingPathComponent("download/LocalLLM-main/LocalLLM.swiftpm")
        try Scratch.write(Scratch.manifest(version: "0.3", bundle: 3), "Package.swift", in: source)
        try Scratch.write("same", "Sources/AppModule/Same.swift", in: source)
        try Scratch.write("new", "Sources/AppModule/Changed.swift", in: source)
        try Scratch.write("added", "Sources/AppModule/Workspace/Added.swift", in: source)
        try Scratch.write("ignored", ".git/HEAD", in: source)
        return (target, source, work)
    }

    func testCheckApplyRecheckAndRestore() throws {
        let (target, source, work) = try makePackages()
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))

        let result = try UpdateEngine.checkFolder(source, label: "test.zip", target: target, objects: objects, base: nil)
        XCTAssertEqual(result.diff.changes(of: .added).map(\.path), ["Sources/AppModule/Workspace/Added.swift"])
        XCTAssertEqual(result.diff.changes(of: .modified).map(\.path), ["Sources/AppModule/Changed.swift"])
        XCTAssertEqual(result.diff.changes(of: .removed).map(\.path), ["Sources/AppModule/Old/Removed.swift"])
        // Package.swift はチーム（この iPad の設定）しか違わないので変更に入らない
        XCTAssertTrue(result.diff.manifestDiffersOnlyInSettings)
        XCTAssertEqual(result.diff.unchangedCount, 2)
        XCTAssertEqual(result.newVersion(), PackageVersion(display: "0.3.1", bundle: 4))

        let record = try AppUpdater.apply(result.request(selection: result.diff.defaultSelection), target: target,
                                          objects: objects, history: history)
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "new")
        XCTAssertEqual(Scratch.read("Sources/AppModule/Workspace/Added.swift", in: target), "added")
        XCTAssertNil(Scratch.read("Sources/AppModule/Old/Removed.swift", in: target))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.appendingPathComponent("Sources/AppModule/Old").path))
        XCTAssertEqual(Scratch.read(".swiftpm/playgrounds/settings.plist", in: target), "settings")
        XCTAssertEqual(Scratch.read("Notes.txt", in: target), "keep")
        XCTAssertNil(Scratch.read(".git/HEAD", in: target))
        let manifest = try XCTUnwrap(Scratch.read("Package.swift", in: target))
        XCTAssertEqual(PackageVersion.parse(packageSwift: manifest), PackageVersion(display: "0.3.1", bundle: 4))
        XCTAssertTrue(manifest.contains("teamIdentifier: \"TEAM123\""))
        XCTAssertEqual(record.created, ["Sources/AppModule/Workspace/Added.swift"])
        XCTAssertEqual(record.changedFileCount, 3)
        XCTAssertEqual(history.load().map(\.id), [record.id])

        // もう一度確かめると、最新になっている（Package.swift の版の違いは無視する）
        let again = try UpdateEngine.checkFolder(source, label: "test.zip", target: target, objects: objects, base: record)
        XCTAssertTrue(again.diff.changes.isEmpty, "\(again.diff.changes)")
        XCTAssertTrue(again.diff.isUpToDate)

        // 取り消すと元どおり
        try AppUpdater.restore(record, target: target, history: history)
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "old")
        XCTAssertEqual(Scratch.read("Sources/AppModule/Old/Removed.swift", in: target), "gone")
        XCTAssertNil(Scratch.read("Sources/AppModule/Workspace/Added.swift", in: target))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.appendingPathComponent("Sources/AppModule/Workspace").path))
        XCTAssertEqual(PackageVersion.parse(packageSwift: Scratch.read("Package.swift", in: target) ?? ""), PackageVersion(display: "0.3", bundle: 3))
        XCTAssertEqual(history.load().first?.restored, true)
        XCTAssertNil(UpdateHistory.restorable(in: history.load(), target: target.lastPathComponent))
    }

    func testOnlyChosenFilesAreWritten() throws {
        let (target, source, work) = try makePackages()
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))
        let result = try UpdateEngine.checkFolder(source, label: "test", target: target, objects: objects, base: nil)
        let before = try FileManager.default.attributesOfItem(atPath: target.appendingPathComponent("Sources/AppModule/Same.swift").path)

        let record = try AppUpdater.apply(result.request(selection: ["Sources/AppModule/Changed.swift"]), target: target,
                                          objects: objects, history: history)
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "new")
        XCTAssertNil(Scratch.read("Sources/AppModule/Workspace/Added.swift", in: target))
        XCTAssertEqual(Scratch.read("Sources/AppModule/Old/Removed.swift", in: target), "gone")
        // 同じだったファイルには触らない
        let after = try FileManager.default.attributesOfItem(atPath: target.appendingPathComponent("Sources/AppModule/Same.swift").path)
        XCTAssertEqual(before[.modificationDate] as? Date, after[.modificationDate] as? Date)
        XCTAssertEqual(record.written, ["Package.swift", "Sources/AppModule/Changed.swift"])

        // 見送ったものは、次の確認でもまた出る
        let next = try UpdateEngine.checkFolder(source, label: "test", target: target, objects: objects, base: record)
        XCTAssertEqual(next.diff.changes.map(\.path), ["Sources/AppModule/Old/Removed.swift", "Sources/AppModule/Workspace/Added.swift"])
        XCTAssertEqual(next.diff.changes.map(\.kind), [.removed, .added])
    }

    func testPackageSwiftChangesAreMergedKeepingTeam() throws {
        let (target, source, work) = try makePackages()
        try Scratch.write(Scratch.manifest(version: "0.4", bundle: 4, extra: "\n                capabilities: [.camera(purposeString: \"x\")],"),
                          "Package.swift", in: source)
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))
        let result = try UpdateEngine.checkFolder(source, label: "test", target: target, objects: objects, base: nil)
        XCTAssertEqual(result.diff.changes.first?.path, "Package.swift")
        XCTAssertEqual(result.newVersion(), PackageVersion(display: "0.4", bundle: 5))
        _ = try AppUpdater.apply(result.request(selection: result.diff.defaultSelection), target: target, objects: objects, history: history)
        let manifest = try XCTUnwrap(Scratch.read("Package.swift", in: target))
        XCTAssertTrue(manifest.contains("capabilities: [.camera(purposeString: \"x\")]"))
        XCTAssertTrue(manifest.contains("teamIdentifier: \"TEAM123\""))
        XCTAssertEqual(PackageVersion.parse(packageSwift: manifest), PackageVersion(display: "0.4", bundle: 5))
    }

    func testRefusesWhenLocalFilesChangedAfterTheCheck() throws {
        let (target, source, work) = try makePackages()
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))
        let result = try UpdateEngine.checkFolder(source, label: "test", target: target, objects: objects, base: nil)
        try Scratch.write("edited meanwhile", "Sources/AppModule/Changed.swift", in: target)
        XCTAssertThrowsError(try AppUpdater.apply(result.request(selection: result.diff.defaultSelection), target: target,
                                                  objects: objects, history: history)) {
            XCTAssertEqual($0 as? AppUpdateError, .localFilesChanged(["Sources/AppModule/Changed.swift"]))
        }
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "edited meanwhile")
        XCTAssertNil(Scratch.read("Sources/AppModule/Workspace/Added.swift", in: target))
        XCTAssertTrue(history.load().isEmpty)
    }

    func testMissingDownloadStopsBeforeWriting() throws {
        let (target, source, work) = try makePackages()
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))
        let result = try UpdateEngine.checkFolder(source, label: "test", target: target, objects: objects, base: nil)
        // 置いてあるファイルが壊れていたら使わない
        let sha = try XCTUnwrap(result.diff.changes(of: .added).first?.remoteSHA)
        try Data("corrupted".utf8).write(to: objects.url(for: sha))
        XCTAssertThrowsError(try AppUpdater.apply(result.request(selection: result.diff.defaultSelection), target: target,
                                                  objects: objects, history: history)) {
            XCTAssertEqual($0 as? AppUpdateError, .missingDownload("Sources/AppModule/Workspace/Added.swift"))
        }
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "old")
    }

    func testFailedWriteRollsBack() throws {
        let (target, source, work) = try makePackages()
        // 新しいソースでは Blocker.swift がファイルだが、この iPad では同じ名前のフォルダになっていて書けない
        try Scratch.write("blocker", "Sources/AppModule/Zzz/Blocker.swift", in: source)
        try Scratch.write("x", "Sources/AppModule/Zzz/Blocker.swift/inner.txt", in: target)
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))
        let result = try UpdateEngine.checkFolder(source, label: "test", target: target, objects: objects, base: nil)
        var request = result.request(selection: result.diff.defaultSelection)
        request.expectedLocal["Sources/AppModule/Zzz/Blocker.swift"] = .some(nil)
        XCTAssertThrowsError(try AppUpdater.apply(request, target: target, objects: objects, history: history))
        // 先に書いたファイルも元に戻っている
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "old")
        XCTAssertNil(Scratch.read("Sources/AppModule/Workspace/Added.swift", in: target))
        XCTAssertEqual(Scratch.read("Sources/AppModule/Old/Removed.swift", in: target), "gone")
        XCTAssertEqual(PackageVersion.parse(packageSwift: Scratch.read("Package.swift", in: target) ?? ""), PackageVersion(display: "0.3", bundle: 3))
        XCTAssertTrue(history.load().isEmpty)
    }

    func testHistoryRestoresNewestFirst() throws {
        let (target, source, work) = try makePackages()
        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"), limit: 3)
        let first = try AppUpdater.apply(try UpdateEngine.checkFolder(source, label: "1", target: target, objects: objects, base: nil)
            .request(selection: ["Sources/AppModule/Changed.swift"]), target: target, objects: objects, history: history)
        try Scratch.write("newer", "Sources/AppModule/Changed.swift", in: source)
        let secondCheck = try UpdateEngine.checkFolder(source, label: "2", target: target, objects: objects,
                                                       base: UpdateHistory.base(in: history.load(), target: target.lastPathComponent))
        XCTAssertEqual(secondCheck.diff.changes.first { $0.path == "Sources/AppModule/Changed.swift" }?.editedHere, false)
        let second = try AppUpdater.apply(secondCheck.request(selection: secondCheck.diff.defaultSelection), target: target,
                                          objects: objects, history: history)
        XCTAssertEqual(PackageVersion.parse(packageSwift: Scratch.read("Package.swift", in: target) ?? ""), PackageVersion(display: "0.3.2", bundle: 5))
        XCTAssertEqual(UpdateHistory.restorable(in: history.load(), target: target.lastPathComponent)?.id, second.id)

        try AppUpdater.restore(second, target: target, history: history)
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "new")
        XCTAssertEqual(UpdateHistory.restorable(in: history.load(), target: target.lastPathComponent)?.id, first.id)
        try AppUpdater.restore(first, target: target, history: history)
        XCTAssertEqual(Scratch.read("Sources/AppModule/Changed.swift", in: target), "old")
        XCTAssertEqual(PackageVersion.parse(packageSwift: Scratch.read("Package.swift", in: target) ?? ""), PackageVersion(display: "0.3", bundle: 3))
        XCTAssertThrowsError(try AppUpdater.restore(first, target: target, history: history))
    }

    func testRejectsTargetsThatAreNotPackages() throws {
        let work = try Scratch.folder()
        let source = work.appendingPathComponent("src")
        try Scratch.write("x", "Package.swift", in: source)
        try Scratch.write("x", "Sources/A.swift", in: source)
        XCTAssertThrowsError(try UpdateEngine.checkFolder(source, label: "x", target: work.appendingPathComponent("nope"),
                                                          objects: UpdateObjectStore(root: work.appendingPathComponent("o")), base: nil))
    }
}

/// 本物の GitHub と比べて、変わったファイルだけを取ってきて更新する（LOCALLLM_LIVE_TESTS=1 のときだけ）
final class GitHubUpdateLiveTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["LOCALLLM_LIVE_TESTS"] == "1" else {
            throw XCTSkip("set LOCALLLM_LIVE_TESTS=1 to run tests against github.com")
        }
    }

    /// このリポジトリの LocalLLM.swiftpm
    private var checkout: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("LocalLLM.swiftpm")
    }

    private func liveOrSkip<T>(_ work: () async throws -> T) async throws -> T {
        do {
            return try await work()
        } catch let error as GitHubAPIError {
            if case .rateLimited = error { throw XCTSkip("GitHub rate limit: \(error.localizedDescription)") }
            throw error
        } catch let error as URLError {
            throw XCTSkip("network unavailable: \(error)")
        }
    }

    func testDifferentialUpdateFromGitHubMain() async throws {
        let repository = try XCTUnwrap(GitHubRepository("prak59459-create/LocalLLM"))
        let client = GitHubClient(repository: repository, token: ProcessInfo.processInfo.environment["GITHUB_TOKEN"])
        let work = try Scratch.folder()
        let target = work.appendingPathComponent("LocalLLM.swiftpm")
        try FileManager.default.copyItem(at: checkout, to: target)
        // わざと古くする: 1 つ書き換え、1 つ消し、1 つ足す
        try Scratch.write("// edited on the iPad\n", "Sources/AppModule/App/LocalLLMApp.swift", in: target)
        try FileManager.default.removeItem(at: target.appendingPathComponent("Sources/AppModule/Core/Update/SHA1.swift"))
        try Scratch.write("// not on GitHub\n", "Sources/AppModule/Core/Update/Leftover.swift", in: target)

        let objects = UpdateObjectStore(root: work.appendingPathComponent("objects"))
        let history = UpdateHistory(folder: work.appendingPathComponent("history"))
        let stages = StageRecorder()
        let result = try await liveOrSkip {
            try await UpdateEngine.checkGitHub(client: client, ref: "main", target: target, objects: objects, base: nil) { stages.add($0) }
        }
        XCTAssertEqual(result.remote.prefix, "LocalLLM.swiftpm/")
        XCTAssertEqual(result.commit?.sha.count, 40)
        let paths = Set(result.diff.changes.map(\.path))
        XCTAssertTrue(paths.contains("Sources/AppModule/App/LocalLLMApp.swift"))
        XCTAssertTrue(paths.contains("Sources/AppModule/Core/Update/Leftover.swift"))
        XCTAssertGreaterThan(result.diff.unchangedCount, 20, "most files should already match GitHub")
        XCTAssertTrue(stages.values.contains(.listing))
        // 変わったファイルは全部、識別子を確かめてダウンロード済み
        for item in result.downloadItems { XCTAssertTrue(objects.contains(item.sha), item.path) }

        let record = try AppUpdater.apply(result.request(selection: Set(result.diff.changes.map(\.path))), target: target,
                                          objects: objects, history: history)
        XCTAssertEqual(record.commit?.sha, result.commit?.sha)
        XCTAssertNil(Scratch.read("Sources/AppModule/Core/Update/Leftover.swift", in: target))

        // 更新したあとは、GitHub と全部同じ（版の違いだけ）
        let again = try await liveOrSkip {
            try await UpdateEngine.checkGitHub(client: client, ref: record.commit?.sha ?? "main", target: target, objects: objects,
                                               base: record) { _ in }
        }
        XCTAssertTrue(again.diff.changes.isEmpty, "\(again.diff.changes.map(\.path))")
        let local = PackageScanner.index(of: target)
        for (path, file) in again.remote.files where path != "Package.swift" {
            XCTAssertEqual(local.files[path]?.sha, file.sha, path)
        }
    }
}

private final class StageRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [UpdateStage] = []

    func add(_ stage: UpdateStage) {
        lock.lock()
        stored.append(stage)
        lock.unlock()
    }

    var values: [UpdateStage] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}
