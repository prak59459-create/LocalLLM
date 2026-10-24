import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import LocalLLMCore

final class SHA1Tests: XCTestCase {
    func testKnownVectors() {
        XCTAssertEqual(SHA1.hex(Data()), "da39a3ee5e6b4b0d3255bfef95601890afd80709")
        XCTAssertEqual(SHA1.hex(Data("abc".utf8)), "a9993e364706816aba3e25717850c26c9cd0d89d")
        XCTAssertEqual(SHA1.hex(Data("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".utf8)),
                       "84983e441c3bd26ebaae4aa1f95129e5e54670f1")
        XCTAssertEqual(SHA1.hex(Data("The quick brown fox jumps over the lazy dog".utf8)),
                       "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12")
        // 64 バイトちょうど（ブロックの境目）
        XCTAssertEqual(SHA1.hex(Data(repeating: UInt8(ascii: "a"), count: 64)), "0098ba824b5c16427bd7a1122a5a442a25ec644d")
        XCTAssertEqual(SHA1.hex(Data(repeating: UInt8(ascii: "a"), count: 1_000_000)), "34aa973cd4c4daa4f61eeb2bdbad27316534016f")
    }

    func testGitBlobMatchesGitHashObject() {
        XCTAssertEqual(GitBlob.sha(of: Data()), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
        XCTAssertEqual(GitBlob.sha(of: Data("hello\n".utf8)), "ce013625030ba8dba906f756967f9e9ca394464a")
        XCTAssertEqual(GitBlob.sha(of: Data("こんにちは\n".utf8)), SHA1.hex(Data("blob 16\u{0}こんにちは\n".utf8)))
    }
}

final class GitHubSourceTests: XCTestCase {
    func testRepositoryParsing() {
        for text in ["owner/repo", " owner/repo ", "https://github.com/owner/repo", "https://github.com/owner/repo.git",
                     "github.com/owner/repo/tree/main", "git@github.com:owner/repo.git", "https://www.github.com/owner/repo/"] {
            XCTAssertEqual(GitHubRepository(text)?.description, "owner/repo", text)
        }
        XCTAssertNil(GitHubRepository("nope"))
        XCTAssertNil(GitHubRepository("owner/re po"))
        XCTAssertNil(GitHubRepository("owner/repo?x"))
    }

    func testCommitAndTreeParsing() throws {
        let commitJSON = """
        {"sha":"d2185e811dd90480af722a089a4fcf2cb8786052","commit":{"message":"Merge pull request #4\\n\\nBump",
         "author":{"name":"prak","date":"2026-10-23T10:00:00Z"},"committer":{"name":"GitHub","date":"2026-10-23T10:05:00Z"},
         "tree":{"sha":"db82116d2a5aeaf341a4a7c94315a84eb343ad93"}}}
        """
        let commit = try GitHubCommit.parse(Data(commitJSON.utf8))
        XCTAssertEqual(commit.shortSHA, "d2185e8")
        XCTAssertEqual(commit.title, "Merge pull request #4")
        XCTAssertEqual(commit.author, "prak")
        XCTAssertEqual(commit.treeSHA, "db82116d2a5aeaf341a4a7c94315a84eb343ad93")
        XCTAssertEqual(commit.date, ISO8601DateFormatter().date(from: "2026-10-23T10:05:00Z"))

        let treeJSON = """
        {"sha":"x","truncated":false,"tree":[
          {"path":"README.md","mode":"100644","type":"blob","sha":"a1","size":10},
          {"path":"App.swiftpm","mode":"040000","type":"tree","sha":"t1"},
          {"path":"App.swiftpm/Package.swift","mode":"100644","type":"blob","sha":"p1","size":20},
          {"path":"App.swiftpm/Sources/A.swift","mode":"100644","type":"blob","sha":"s1","size":30},
          {"path":"App.swiftpm/Sources/Link.swift","mode":"120000","type":"blob","sha":"l1","size":8},
          {"path":"App.swiftpm/Vendor","mode":"160000","type":"commit","sha":"c1"},
          {"path":"App.swiftpm/.swiftpm/settings.plist","mode":"100644","type":"blob","sha":"x1","size":1}
        ]}
        """
        let tree = try GitHubTree.parse(Data(treeJSON.utf8))
        XCTAssertFalse(tree.truncated)
        let package = try XCTUnwrap(RemotePackage.locate(in: tree, preferredNames: ["Other.swiftpm"]))
        XCTAssertEqual(package.prefix, "App.swiftpm/")
        XCTAssertEqual(package.name, "App.swiftpm")
        XCTAssertEqual(Set(package.files.keys), ["Package.swift", "Sources/A.swift"])
        XCTAssertEqual(package.files["Sources/A.swift"], PackageFile(sha: "s1", size: 30))
    }

    func testLocatePrefersTheTargetName() throws {
        func blob(_ path: String) -> GitHubTreeEntry { GitHubTreeEntry(path: path, sha: path, size: 1, isFile: true) }
        let tree = GitHubTree(entries: [
            blob("Package.swift"), blob("Tests/X.swift"),
            blob("A.swiftpm/Package.swift"), blob("A.swiftpm/Sources/A.swift"),
            blob("apps/LocalLLM.swiftpm/Package.swift"), blob("apps/LocalLLM.swiftpm/Sources/B.swift"),
            blob(".build/checkouts/Z.swiftpm/Package.swift"), blob(".build/checkouts/Z.swiftpm/Sources/Z.swift"),
        ], truncated: false)
        XCTAssertEqual(RemotePackage.locate(in: tree, preferredNames: ["LocalLLM.swiftpm"])?.prefix, "apps/LocalLLM.swiftpm/")
        XCTAssertEqual(RemotePackage.locate(in: tree, preferredNames: ["Nope"])?.prefix, "A.swiftpm/")
        XCTAssertNil(RemotePackage.locate(in: GitHubTree(entries: [blob("Package.swift")], truncated: false), preferredNames: []))
        // リポジトリそのものがパッケージ
        let rootPackage = GitHubTree(entries: [blob("Package.swift"), blob("Sources/App/Main.swift")], truncated: false)
        XCTAssertEqual(RemotePackage.locate(in: rootPackage, preferredNames: [])?.prefix, "")
    }

    func testRequestsAndErrors() throws {
        let repo = try XCTUnwrap(GitHubRepository("owner/repo"))
        XCTAssertEqual(GitHubAPI.commitURL(repo, ref: "feature/x").absoluteString, "https://api.github.com/repos/owner/repo/commits/feature/x")
        XCTAssertEqual(GitHubAPI.treeURL(repo, sha: "abc").absoluteString, "https://api.github.com/repos/owner/repo/git/trees/abc?recursive=1")
        let raw = GitHubAPI.fileRequest(repo, commit: "c0ffee", path: "App.swiftpm/Sources/My File.swift", token: nil)
        XCTAssertEqual(raw.url?.absoluteString, "https://raw.githubusercontent.com/owner/repo/c0ffee/App.swiftpm/Sources/My%20File.swift")
        XCTAssertNil(raw.value(forHTTPHeaderField: "Authorization"))
        let api = GitHubAPI.fileRequest(repo, commit: "c0ffee", path: "A/日本.swift", token: "t")
        XCTAssertEqual(api.url?.absoluteString,
                       "https://api.github.com/repos/owner/repo/contents/A/%E6%97%A5%E6%9C%AC.swift?ref=c0ffee")
        XCTAssertEqual(api.value(forHTTPHeaderField: "Accept"), "application/vnd.github.raw")
        XCTAssertEqual(api.value(forHTTPHeaderField: "Authorization"), "Bearer t")

        XCTAssertNoThrow(try GitHubAPI.check(status: 200, headers: [:], body: Data()))
        XCTAssertThrowsError(try GitHubAPI.check(status: 404, headers: [:], body: Data())) { XCTAssertEqual($0 as? GitHubAPIError, .notFound) }
        XCTAssertThrowsError(try GitHubAPI.check(status: 401, headers: [:], body: Data())) { XCTAssertEqual($0 as? GitHubAPIError, .unauthorized) }
        let limited = Data(#"{"message":"API rate limit exceeded for 1.2.3.4."}"#.utf8)
        XCTAssertThrowsError(try GitHubAPI.check(status: 403, headers: ["X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1800000000"],
                                                 body: limited)) {
            XCTAssertEqual($0 as? GitHubAPIError, .rateLimited(Date(timeIntervalSince1970: 1_800_000_000)))
        }
        XCTAssertThrowsError(try GitHubAPI.check(status: 500, headers: [:], body: Data(#"{"message":"boom"}"#.utf8))) {
            XCTAssertEqual($0 as? GitHubAPIError, .http(500, "boom"))
        }
        XCTAssertEqual(GitHubAPI.parseBranches(Data(#"[{"name":"main"},{"name":"dev"}]"#.utf8)), ["main", "dev"])
    }
}

final class PackageManifestTests: XCTestCase {
    let local = """
    // コメント: displayVersion: "9.9"
    let package = Package(
        name: "LocalLLM",
        products: [
            .iOSApplication(
                name: "LocalLLM",
                targets: ["AppModule"],
                bundleIdentifier: "jp.example.mine",
                teamIdentifier: "ABCDE12345",
                displayVersion: "0.3.2",
                bundleVersion: "5",
                appIcon: .asset("AppIcon"),
                accentColor: .presetColor(.indigo),
                appCategory: .productivity
            )
        ]
    )
    """

    let remote = """
    // コメント: displayVersion: "9.9"
    let package = Package(
        name: "LocalLLM",
        products: [
            .iOSApplication(
                name: "LocalLLM",
                targets: ["AppModule"],
                bundleIdentifier: "com.example.LocalLLM",
                teamIdentifier: "",
                displayVersion: "0.3",
                bundleVersion: "3",
                appIcon: .placeholder(icon: .chatMessage),
                accentColor: .presetColor(.indigo),
                appCategory: .productivity
            )
        ]
    )
    """

    func testValues() {
        XCTAssertEqual(PackageManifest.stringValue(of: "displayVersion", in: local), "0.3.2")
        XCTAssertEqual(PackageManifest.stringValue(of: "name", in: local), "LocalLLM")
        XCTAssertEqual(PackageManifest.stringValue(of: "teamIdentifier", in: remote), "")
        XCTAssertEqual(PackageManifest.expressionValue(of: "appIcon", in: remote), ".placeholder(icon: .chatMessage)")
        XCTAssertEqual(PackageManifest.expressionValue(of: "appCategory", in: remote), ".productivity")
        XCTAssertEqual(PackageManifest.expressionValue(of: "targets", in: remote), "[\"AppModule\"]")
        XCTAssertNil(PackageManifest.valueRange(of: "missing", in: remote))
        XCTAssertEqual(PackageVersion.parse(packageSwift: local), PackageVersion(display: "0.3.2", bundle: 5))
        // コメントの中の値は書き換えない
        let bumped = PackageVersion(display: "0.4", bundle: 6).apply(to: local)
        XCTAssertTrue(bumped.contains("// コメント: displayVersion: \"9.9\""))
        XCTAssertTrue(bumped.contains("displayVersion: \"0.4\","))
        XCTAssertTrue(bumped.contains("bundleVersion: \"6\","))
    }

    func testEquivalenceIgnoresVersionAndThisIPadsSettings() {
        XCTAssertTrue(PackageManifest.isEquivalent(local, remote))
        let withCapability = remote.replacingOccurrences(of: "appCategory: .productivity",
                                                         with: "capabilities: [.camera(purposeString: \"x\")],\n            appCategory: .productivity")
        XCTAssertFalse(PackageManifest.isEquivalent(local, withCapability))
        XCTAssertTrue(PackageManifest.isEquivalent(remote, remote.replacingOccurrences(of: "\n", with: "\r\n")))
    }

    func testMergeKeepsThisIPadsSettings() {
        let withCapability = remote.replacingOccurrences(of: "appCategory: .productivity",
                                                         with: "capabilities: [.camera(purposeString: \"x\")],\n            appCategory: .productivity")
        let merged = PackageManifest.merge(local: local, remote: withCapability, base: nil, version: PackageVersion(display: "0.3.3", bundle: 6))
        XCTAssertTrue(merged.contains("bundleIdentifier: \"jp.example.mine\""))
        XCTAssertTrue(merged.contains("teamIdentifier: \"ABCDE12345\""))
        XCTAssertTrue(merged.contains("appIcon: .asset(\"AppIcon\")"))
        XCTAssertTrue(merged.contains("capabilities: [.camera(purposeString: \"x\")]"))
        XCTAssertEqual(PackageVersion.parse(packageSwift: merged), PackageVersion(display: "0.3.3", bundle: 6))

        // この iPad でアイコンを変えていなければ（前回の GitHub の値と同じなら）、GitHub の新しいアイコンにする
        let untouched = local.replacingOccurrences(of: ".asset(\"AppIcon\")", with: ".placeholder(icon: .chatMessage)")
        let newIcon = remote.replacingOccurrences(of: ".placeholder(icon: .chatMessage)", with: ".asset(\"NewIcon\")")
        let merged2 = PackageManifest.merge(local: untouched, remote: newIcon, base: remote, version: PackageVersion(display: "0.4", bundle: 7))
        XCTAssertTrue(merged2.contains("appIcon: .asset(\"NewIcon\")"))
        XCTAssertTrue(merged2.contains("teamIdentifier: \"ABCDE12345\""))
    }

    func testVersionRules() {
        XCTAssertEqual(PackageVersion.bumpPatch("0.2"), "0.2.1")
        XCTAssertEqual(PackageVersion.bumpPatch("0.2.9"), "0.2.10")
        XCTAssertEqual(PackageVersion.bumpPatch("1"), "1.0.1")
        XCTAssertEqual(PackageVersion.compare("0.2.10", "0.2.9"), .orderedDescending)
        XCTAssertEqual(PackageVersion.compare("1.0", "1.0.0"), .orderedSame)
        XCTAssertEqual(PackageVersion.next(current: .init(display: "0.2", bundle: 2), incoming: .init(display: "0.2", bundle: 2)),
                       PackageVersion(display: "0.2.1", bundle: 3))
        XCTAssertEqual(PackageVersion.next(current: .init(display: "0.2.3", bundle: 7), incoming: .init(display: "0.3", bundle: 3)),
                       PackageVersion(display: "0.3", bundle: 8))
        XCTAssertEqual(PackageVersion.next(current: .init(display: "0.3", bundle: 9), incoming: .init(display: "0.2", bundle: 2)),
                       PackageVersion(display: "0.3.1", bundle: 10))
        XCTAssertEqual(PackageVersion.parse(description: "0.3.1 (4)"), PackageVersion(display: "0.3.1", bundle: 4))
    }
}

final class UpdateDiffTests: XCTestCase {
    private func files(_ pairs: [String: String]) -> [String: PackageFile] {
        pairs.mapValues { PackageFile(sha: $0, size: 1) }
    }

    func testDiffWithoutBase() {
        let local = files(["Package.swift": "p0", "Sources/Same.swift": "s", "Sources/Changed.swift": "c0",
                           "Sources/Old.swift": "o", "Notes.txt": "n"])
        let remote = files(["Package.swift": "p1", "Sources/Same.swift": "s", "Sources/Changed.swift": "c1",
                            "Sources/New.swift": "n1"])
        let diff = UpdateDiffer.diff(local: local, remote: remote, base: nil)
        XCTAssertEqual(diff.changes.map(\.path), ["Package.swift", "Sources/Changed.swift", "Sources/New.swift", "Sources/Old.swift"])
        XCTAssertEqual(diff.changes(of: .added).map(\.path), ["Sources/New.swift"])
        XCTAssertEqual(diff.changes(of: .modified).map(\.path), ["Package.swift", "Sources/Changed.swift"])
        XCTAssertEqual(diff.changes(of: .removed).map(\.path), ["Sources/Old.swift"])
        XCTAssertEqual(diff.unchangedCount, 1)
        XCTAssertFalse(diff.isUpToDate)
        XCTAssertEqual(diff.defaultSelection.count, 4)

        // Package.swift の違いが版や設定だけなら、同じとみなす
        let sameLocal = files(["Package.swift": "p0", "Sources/Same.swift": "s", "Notes.txt": "n"])
        let sameRemote = files(["Package.swift": "p1", "Sources/Same.swift": "s"])
        let same = UpdateDiffer.diff(local: sameLocal, remote: sameRemote, base: nil, manifestEquivalent: true)
        XCTAssertTrue(same.changes.isEmpty)
        XCTAssertTrue(same.isUpToDate)
        XCTAssertTrue(same.manifestDiffersOnlyInSettings)
    }

    func testThreeWayWithBase() {
        let base = ["Package.swift": "p", "Sources/A.swift": "a0", "Sources/B.swift": "b0", "Sources/C.swift": "c0",
                    "Sources/D.swift": "d0", "Sources/E.swift": "e0", "Sources/F.swift": "f0"]
        let local = files(["Package.swift": "p", "Sources/A.swift": "a0",   // GitHub だけが変えた
                           "Sources/B.swift": "b-here",                     // この iPad だけが変えた
                           "Sources/C.swift": "c-here",                     // 両方が変えた
                           // D はこの iPad で消した
                           "Sources/E.swift": "e0",                         // GitHub で消えた
                           "Sources/F.swift": "f-here",                     // GitHub で消えたが、ここでは書き換えた
                           "Sources/Mine.swift": "m"])                      // この iPad で足した
        let remote = files(["Package.swift": "p", "Sources/A.swift": "a1", "Sources/B.swift": "b0", "Sources/C.swift": "c1",
                            "Sources/D.swift": "d0", "Sources/G.swift": "g1"])
        let diff = UpdateDiffer.diff(local: local, remote: remote, base: base)
        func change(_ path: String) -> UpdateFileChange? { diff.changes.first { $0.path == path } }
        XCTAssertEqual(change("Sources/A.swift")?.kind, .modified)
        XCTAssertEqual(change("Sources/A.swift")?.editedHere, false)
        XCTAssertEqual(change("Sources/B.swift")?.kind, .localEdit)
        XCTAssertEqual(change("Sources/C.swift")?.kind, .modified)
        XCTAssertEqual(change("Sources/C.swift")?.editedHere, true)
        XCTAssertEqual(change("Sources/D.swift")?.kind, .localDeleted)
        XCTAssertEqual(change("Sources/E.swift")?.kind, .removed)
        XCTAssertEqual(change("Sources/E.swift")?.editedHere, false)
        XCTAssertEqual(change("Sources/F.swift")?.kind, .removed)
        XCTAssertEqual(change("Sources/F.swift")?.editedHere, true)
        XCTAssertEqual(change("Sources/G.swift")?.kind, .added)
        XCTAssertNil(change("Sources/Mine.swift"))
        XCTAssertEqual(diff.keptLocalFiles, ["Sources/Mine.swift"])
        XCTAssertEqual(diff.defaultSelection, ["Sources/A.swift", "Sources/C.swift", "Sources/E.swift", "Sources/F.swift", "Sources/G.swift"])

        // 一部だけ更新したあとの記録: 見送った GitHub の変更は前回の内容のまま残る
        let manifest = UpdateDiffer.manifest(after: diff, remote: remote, base: base,
                                             applied: ["Sources/A.swift", "Sources/G.swift", "Sources/E.swift"])
        XCTAssertEqual(manifest["Sources/A.swift"], "a1")
        XCTAssertEqual(manifest["Sources/G.swift"], "g1")
        XCTAssertNil(manifest["Sources/E.swift"])
        XCTAssertEqual(manifest["Sources/C.swift"], "c0")     // 見送った
        XCTAssertEqual(manifest["Sources/F.swift"], "f0")     // 消すのを見送った
        XCTAssertEqual(manifest["Sources/B.swift"], "b0")
        XCTAssertEqual(manifest["Sources/D.swift"], "d0")
        XCTAssertNil(manifest["Sources/Mine.swift"])

        // 見送った変更は、次の確認でもまた「変更」として出る
        var afterLocal = local
        afterLocal["Sources/A.swift"] = PackageFile(sha: "a1", size: 1)
        afterLocal["Sources/G.swift"] = PackageFile(sha: "g1", size: 1)
        afterLocal["Sources/E.swift"] = nil
        let next = UpdateDiffer.diff(local: afterLocal, remote: remote, base: manifest)
        XCTAssertEqual(next.changes.first { $0.path == "Sources/C.swift" }?.kind, .modified)
        XCTAssertNil(next.changes.first { $0.path == "Sources/A.swift" })
        XCTAssertEqual(next.changes.first { $0.path == "Sources/F.swift" }?.kind, .removed)
    }
}
