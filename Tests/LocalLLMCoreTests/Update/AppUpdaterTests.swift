import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

final class PackageVersionTests: XCTestCase {
    let package = """
    .iOSApplication(
        name: "LocalLLM",
        displayVersion: "0.2",
        bundleVersion: "2",
        appIcon: .placeholder(icon: .chatMessage),
    """

    func testParseAndApply() {
        let version = PackageVersion.parse(packageSwift: package)
        XCTAssertEqual(version, PackageVersion(display: "0.2", bundle: 2))
        let updated = PackageVersion(display: "0.2.1", bundle: 3).apply(to: package)
        XCTAssertTrue(updated.contains("displayVersion: \"0.2.1\""))
        XCTAssertTrue(updated.contains("bundleVersion: \"3\""))
        XCTAssertTrue(updated.contains("appIcon: .placeholder(icon: .chatMessage)"))
    }

    func testVersionGoesUpEveryTime() {
        XCTAssertEqual(PackageVersion.bumpPatch("0.2"), "0.2.1")
        XCTAssertEqual(PackageVersion.bumpPatch("0.2.9"), "0.2.10")
        XCTAssertEqual(PackageVersion.bumpPatch("1"), "1.0.1")
        XCTAssertEqual(PackageVersion.compare("0.2.10", "0.2.9"), .orderedDescending)
        XCTAssertEqual(PackageVersion.compare("1.0", "1.0.0"), .orderedSame)

        // 同じ版のソースで上書き → 最後の数字と bundleVersion が上がる
        let same = PackageVersion.next(current: .init(display: "0.2", bundle: 2), incoming: .init(display: "0.2", bundle: 2))
        XCTAssertEqual(same, PackageVersion(display: "0.2.1", bundle: 3))
        // 新しい版のソース → その版を使う
        let newer = PackageVersion.next(current: .init(display: "0.2.3", bundle: 7), incoming: .init(display: "0.3", bundle: 3))
        XCTAssertEqual(newer, PackageVersion(display: "0.3", bundle: 8))
        // 古いソースでも版は下がらない
        let older = PackageVersion.next(current: .init(display: "0.3", bundle: 9), incoming: .init(display: "0.2", bundle: 2))
        XCTAssertEqual(older, PackageVersion(display: "0.3.1", bundle: 10))
    }
}

final class AppUpdaterTests: XCTestCase {
    private func makeFolder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("upd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ text: String, _ path: String, in root: URL) throws {
        let url = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    private func read(_ path: String, in root: URL) -> String? {
        try? String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }

    func testPlanApplyAndRestore() throws {
        let target = try makeFolder()
        try write("displayVersion: \"0.2\",\nbundleVersion: \"5\",", "Package.swift", in: target)
        try write("same", "Sources/AppModule/Same.swift", in: target)
        try write("old", "Sources/AppModule/Changed.swift", in: target)
        try write("gone", "Sources/AppModule/Old/Removed.swift", in: target)
        try write("settings", ".swiftpm/playgrounds/settings.plist", in: target)
        try write("keep", "Notes.txt", in: target)

        let download = try makeFolder()
        let source = download.appendingPathComponent("LocalLLM-main/LocalLLM.swiftpm")
        try write("displayVersion: \"0.2\",\nbundleVersion: \"2\",", "Package.swift", in: source)
        try write("same", "Sources/AppModule/Same.swift", in: source)
        try write("new", "Sources/AppModule/Changed.swift", in: source)
        try write("added", "Sources/AppModule/Workspace/Added.swift", in: source)
        try write("ignored", ".git/HEAD", in: source)
        try write("dev package", "Package.swift", in: download.appendingPathComponent("LocalLLM-main"))

        XCTAssertEqual(AppUpdater.packageRoot(in: download)?.lastPathComponent, "LocalLLM.swiftpm")

        let plan = try AppUpdater.plan(source: source, target: target)
        XCTAssertEqual(plan.added, ["Sources/AppModule/Workspace/Added.swift"])
        XCTAssertEqual(plan.changed, ["Package.swift", "Sources/AppModule/Changed.swift"])
        XCTAssertEqual(plan.removed, ["Sources/AppModule/Old/Removed.swift"])
        XCTAssertEqual(plan.unchangedCount, 1)
        XCTAssertEqual(plan.newVersion, PackageVersion(display: "0.2.1", bundle: 6))

        let backup = try makeFolder().appendingPathComponent("backup")
        try AppUpdater.apply(plan, source: source, target: target, backup: backup)
        XCTAssertEqual(read("Sources/AppModule/Changed.swift", in: target), "new")
        XCTAssertEqual(read("Sources/AppModule/Workspace/Added.swift", in: target), "added")
        XCTAssertNil(read("Sources/AppModule/Old/Removed.swift", in: target))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.appendingPathComponent("Sources/AppModule/Old").path))
        XCTAssertEqual(read(".swiftpm/playgrounds/settings.plist", in: target), "settings")
        XCTAssertEqual(read("Notes.txt", in: target), "keep")
        XCTAssertNil(read(".git/HEAD", in: target))
        XCTAssertEqual(PackageVersion.parse(packageSwift: read("Package.swift", in: target) ?? ""), PackageVersion(display: "0.2.1", bundle: 6))

        // 2 回目は版がまた上がる
        let second = try AppUpdater.plan(source: source, target: target)
        XCTAssertEqual(second.newVersion, PackageVersion(display: "0.2.2", bundle: 7))
        XCTAssertEqual(second.changed, ["Package.swift"])

        try AppUpdater.restore(from: backup, target: target)
        XCTAssertEqual(read("Sources/AppModule/Changed.swift", in: target), "old")
        XCTAssertEqual(read("Sources/AppModule/Old/Removed.swift", in: target), "gone")
        XCTAssertNil(read("Sources/AppModule/Workspace/Added.swift", in: target))
        XCTAssertEqual(PackageVersion.parse(packageSwift: read("Package.swift", in: target) ?? ""), PackageVersion(display: "0.2", bundle: 5))
    }

    func testRejectsTargetsThatAreNotPackages() throws {
        let target = try makeFolder()
        let source = try makeFolder()
        try write("x", "Package.swift", in: source)
        XCTAssertThrowsError(try AppUpdater.plan(source: source, target: target))
    }

    func testGitHubRequests() {
        let open = AppUpdater.githubArchiveRequest(repository: "https://github.com/owner/repo.git", ref: "", token: nil)
        XCTAssertEqual(open?.url?.absoluteString, "https://codeload.github.com/owner/repo/zip/refs/heads/main")
        let privateRepo = AppUpdater.githubArchiveRequest(repository: "owner/repo", ref: "feature/x", token: "t")
        XCTAssertEqual(privateRepo?.url?.absoluteString, "https://api.github.com/repos/owner/repo/zipball/feature/x")
        XCTAssertEqual(privateRepo?.value(forHTTPHeaderField: "Authorization"), "Bearer t")
        XCTAssertNil(AppUpdater.githubArchiveRequest(repository: "nope", ref: "main", token: nil))
    }
}
