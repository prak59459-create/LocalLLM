import Foundation
import XCTest
@testable import LocalLLMCore

private func makeWorkspace() throws -> WorkspaceFileSystem {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("cdws-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return WorkspaceFileSystem(root: root)
}

final class WorkspaceFileSystemTests: XCTestCase {
    func testPathsStayInsideTheProject() throws {
        let fs = try makeWorkspace()
        XCTAssertEqual(try fs.normalize("./src/../src/app.js"), "src/app.js")
        XCTAssertEqual(try fs.normalize("/src/app.js"), "src/app.js")
        XCTAssertEqual(try fs.normalize(fs.root.path + "/a/b"), "a/b")
        XCTAssertEqual(try fs.normalize(""), "")
        XCTAssertThrowsError(try fs.normalize("../secret"))
        XCTAssertThrowsError(try fs.normalize("a/../../b"))
        XCTAssertThrowsError(try fs.write("../x.txt", "no"))
    }

    func testWriteReadEditAndList() throws {
        let fs = try makeWorkspace()
        let created = try fs.write("src/app.js", "let a = 1;\nlet b = 2;\n")
        XCTAssertEqual(created.kind, .created)
        XCTAssertNil(created.before)
        XCTAssertEqual(try fs.read("src/app.js"), "let a = 1;\nlet b = 2;\n")

        let edited = try fs.edit("src/app.js", old: "let b = 2;", new: "let b = 3;")
        XCTAssertEqual(edited.kind, .modified)
        XCTAssertEqual(edited.stats.added, 1)
        XCTAssertEqual(edited.stats.removed, 1)
        XCTAssertEqual(try fs.read("src/app.js"), "let a = 1;\nlet b = 3;\n")

        XCTAssertThrowsError(try fs.edit("src/app.js", old: "missing", new: "x")) { error in
            XCTAssertEqual(error as? WorkspaceError, .editNotFound("src/app.js"))
        }
        try fs.write("dup.txt", "x\nx\n")
        XCTAssertThrowsError(try fs.edit("dup.txt", old: "x", new: "y")) { error in
            XCTAssertEqual(error as? WorkspaceError, .editNotUnique("dup.txt", 2))
        }
        try fs.edit("dup.txt", old: "x", new: "y", replaceAll: true)
        XCTAssertEqual(try fs.read("dup.txt"), "y\ny\n")

        let names = try fs.children().map(\.name)
        XCTAssertEqual(names, ["src", "dup.txt"])  // フォルダが先
        XCTAssertTrue(fs.tree().contains("src/\n  app.js"))
    }

    func testFuzzyEditIgnoresTrailingWhitespaceAndCRLF() throws {
        let fs = try makeWorkspace()
        try fs.write("a.txt", "one  \r\ntwo\r\nthree\n")
        try fs.edit("a.txt", old: "one\ntwo", new: "1\n2")
        XCTAssertEqual(try fs.read("a.txt"), "1\n2\nthree\n")
    }

    func testReadNumberedPaging() throws {
        let fs = try makeWorkspace()
        try fs.write("long.txt", (1...50).map { "line \($0)" }.joined(separator: "\n"))
        let page = try fs.readNumbered("long.txt", offset: 10, limit: 5)
        XCTAssertTrue(page.text.hasPrefix("10| line 10\n"))
        XCTAssertEqual(page.shownUntil, 14)
        XCTAssertEqual(page.totalLines, 50)
    }

    func testSearchFindMoveDeleteAndBinaryFiles() throws {
        let fs = try makeWorkspace()
        try fs.write("a/one.swift", "struct One {}\nlet value = 1\n")
        try fs.write("b/two.swift", "struct Two {}\n")
        try fs.write(".git/config", "struct Hidden {}\n")
        let matches = try fs.search("struct")
        XCTAssertEqual(matches.map(\.path), ["a/one.swift", "b/two.swift"])
        XCTAssertEqual(try fs.search("VALUE").first?.line, 2)
        XCTAssertEqual(try fs.search("^let", isRegex: true).count, 1)
        XCTAssertEqual(try fs.find("*.swift"), ["a/one.swift", "b/two.swift"])
        XCTAssertEqual(try fs.find("b/*"), ["b/two.swift"])

        let moved = try fs.move("b/two.swift", to: "c/2.swift")
        XCTAssertEqual(moved.movedFrom, "b/two.swift")
        XCTAssertTrue(fs.exists("c/2.swift"))
        XCTAssertFalse(fs.exists("b/two.swift"))

        let deleted = try fs.delete("a")
        XCTAssertEqual(deleted.map(\.path), ["a/one.swift"])
        XCTAssertEqual(deleted.first?.before, "struct One {}\nlet value = 1\n")
        XCTAssertFalse(fs.exists("a"))

        try Data([0x89, 0x50, 0x4E, 0x47, 0x00, 0x01]).write(to: fs.root.appendingPathComponent("image.png"))
        XCTAssertThrowsError(try fs.read("image.png")) { error in
            XCTAssertEqual(error as? WorkspaceError, .notText("image.png"))
        }
    }

    func testGlob() {
        XCTAssertTrue(Glob.matches("*.js", "app.js"))
        XCTAssertFalse(Glob.matches("*.js", "app.jsx"))
        XCTAssertTrue(Glob.matches("a?c", "abc"))
        XCTAssertTrue(Glob.matches("**/*.css", "web/styles/site.css"))
        XCTAssertTrue(Glob.matches("README*", "readme.md"))
    }
}

final class ChangeHistoryTests: XCTestCase {
    func testUndoRestoresEveryKindOfChange() throws {
        let fs = try makeWorkspace()
        var history = ChangeHistory()
        let batch = UUID()

        var created = try fs.write("new.txt", "hello")
        created.batchID = batch
        history.record(created)
        try fs.write("keep.txt", "v1")
        var modified = try fs.write("keep.txt", "v2")
        modified.batchID = batch
        history.record(modified)
        var moved = try fs.move("keep.txt", to: "moved.txt")
        moved.batchID = batch
        history.record(moved)

        try history.undoBatch(batch, in: fs)
        XCTAssertTrue(history.changes.isEmpty)
        XCTAssertFalse(fs.exists("new.txt"))
        XCTAssertFalse(fs.exists("moved.txt"))
        XCTAssertEqual(try fs.read("keep.txt"), "v1")

        let deleted = try fs.delete("keep.txt")
        history.record(deleted)
        try history.undo(deleted[0].id, in: fs)
        XCTAssertEqual(try fs.read("keep.txt"), "v1")
    }
}

final class TextDiffTests: XCTestCase {
    func testDiffMarksInsertedAndRemovedLines() {
        let old = "a\nb\nc\nd\n"
        let new = "a\nB\nc\nd\ne\n"
        let lines = TextDiff.diff(old: old, new: new)
        XCTAssertEqual(lines.map(\.kind), [.context, .removed, .added, .context, .context, .added])
        XCTAssertEqual(lines[2].newNumber, 2)
        XCTAssertEqual(lines[1].oldNumber, 2)
        XCTAssertEqual(TextDiff.stats(old: old, new: new).added, 2)
        XCTAssertEqual(TextDiff.stats(old: old, new: new).removed, 1)
    }

    func testDiffReconstructsBothSides() {
        // ランダムな変更でも、差分から元と新しい中身をそのまま組み立て直せること
        var generator = SystemRandomNumberGenerator()
        for _ in 0..<40 {
            let old = (0..<Int.random(in: 0...30, using: &generator)).map { _ in String(Int.random(in: 0...5, using: &generator)) }
            var new = old
            for _ in 0..<Int.random(in: 0...8, using: &generator) {
                if !new.isEmpty && Bool.random(using: &generator) {
                    new.remove(at: Int.random(in: 0..<new.count, using: &generator))
                } else {
                    new.insert(String(Int.random(in: 0...9, using: &generator)), at: Int.random(in: 0...new.count, using: &generator))
                }
            }
            let lines = TextDiff.diff(oldLines: old, newLines: new)
            XCTAssertEqual(lines.filter { $0.kind != .added }.map(\.text), old)
            XCTAssertEqual(lines.filter { $0.kind != .removed }.map(\.text), new)
        }
    }

    func testHunksAndUnifiedOutput() {
        let old = (1...20).map { "line \($0)" }.joined(separator: "\n")
        let new = old.replacingOccurrences(of: "line 3\n", with: "line three\n")
            .replacingOccurrences(of: "line 18\n", with: "line 18\nextra\n")
        let hunks = TextDiff.hunks(old: old, new: new, context: 2)
        XCTAssertEqual(hunks.count, 2)
        XCTAssertEqual(hunks[0].header, "@@ -1,5 +1,5 @@")
        XCTAssertEqual(hunks[1].oldStart, 17)
        let unified = TextDiff.unified(old: old, new: new, path: "f.txt")
        XCTAssertTrue(unified.hasPrefix("--- a/f.txt\n+++ b/f.txt\n@@"))
        XCTAssertTrue(unified.contains("-line 3\n+line three\n"))
        XCTAssertTrue(unified.contains("+extra\n"))
        XCTAssertEqual(TextDiff.unified(old: old, new: old, path: "f.txt"), "")
    }

    func testLargeDiffIsFast() {
        let old = (0..<3000).map { "let value\($0) = \($0)" }
        var new = old
        for index in stride(from: 0, to: 3000, by: 97) { new[index] = "// changed \(index)" }
        let start = Date()
        let lines = TextDiff.diff(oldLines: old, newLines: new)
        XCTAssertLessThan(Date().timeIntervalSince(start), 2.0)
        XCTAssertEqual(lines.filter { $0.kind == .added }.count, 31)
    }
}

final class RoadmapTests: XCTestCase {
    func testAddBranchUpdateAndSummary() {
        var roadmap = Roadmap()
        let goal = roadmap.add(title: "ToDo app", kind: .goal, status: .inProgress, at: (0, 0))
        let list = roadmap.add(title: "List screen", parent: goal.id)
        let save = roadmap.add(title: "Save to disk", parent: list.id)
        XCTAssertEqual([goal.number, list.number, save.number], [1, 2, 3])
        XCTAssertEqual(roadmap.children(of: goal.id).map(\.title), ["List screen"])
        XCTAssertEqual(roadmap.parents(of: save.id).map(\.title), ["List screen"])
        XCTAssertEqual(roadmap.node(reference: "#3")?.id, save.id)
        XCTAssertEqual(roadmap.node(reference: "2")?.id, list.id)
        XCTAssertEqual(roadmap.node(reference: "save to disk")?.id, save.id)

        // 子は親の近くに、ほかと重ならないように置かれる
        for node in roadmap.nodes {
            for other in roadmap.nodes where other.id != node.id {
                XCTAssertGreaterThan(hypot(node.x - other.x, node.y - other.y), Roadmap.spacing * 0.7)
            }
        }
        XCTAssertLessThan(hypot(list.x - goal.x, list.y - goal.y), Roadmap.spacing * 1.5)

        roadmap.update(list.id) { $0.status = .done }
        let summary = roadmap.summary()
        XCTAssertTrue(summary.contains("#2 [done] List screen ← #1"))
        XCTAssertEqual(roadmap.progress.done, 1)

        roadmap.connect(goal.id, save.id)
        roadmap.connect(goal.id, save.id)
        XCTAssertEqual(roadmap.edges.count, 3)
        roadmap.remove(list.id)
        XCTAssertEqual(roadmap.edges.count, 1)
    }

    func testAutoLayoutPutsChildrenToTheRight() {
        var roadmap = Roadmap()
        let root = roadmap.add(title: "root", at: (500, 500))
        let a = roadmap.add(title: "a", parent: root.id)
        let b = roadmap.add(title: "b", parent: root.id)
        let c = roadmap.add(title: "c", parent: a.id)
        roadmap.autoLayout()
        let positions = Dictionary(uniqueKeysWithValues: roadmap.nodes.map { ($0.id, $0) })
        XCTAssertEqual(positions[root.id]?.x, 0)
        XCTAssertGreaterThan(positions[a.id]!.x, positions[root.id]!.x)
        XCTAssertGreaterThan(positions[c.id]!.x, positions[a.id]!.x)
        XCTAssertNotEqual(positions[a.id]!.y, positions[b.id]!.y)
    }

    func testStatusParsing() {
        XCTAssertEqual(RoadmapNode.Status.parse("in_progress"), .inProgress)
        XCTAssertEqual(RoadmapNode.Status.parse("Done"), .done)
        XCTAssertEqual(RoadmapNode.Status.parse("完了"), .done)
        XCTAssertNil(RoadmapNode.Status.parse("???"))
    }
}

final class WorkspaceToolTests: XCTestCase {
    private func call(_ name: String, _ arguments: [(String, JSONValue)]) -> ToolCall {
        ToolCall(name: name, arguments: .object(arguments))
    }

    func testToolsReadWriteEditAndRoadmap() throws {
        let fs = try makeWorkspace()
        let executor = WorkspaceToolExecutor(fileSystem: fs)
        var roadmap = Roadmap()
        roadmap.add(title: "App", kind: .goal, at: (0, 0))

        var result = executor.execute(call("write_file", [("path", .string("index.html")), ("content", .string("<p>hi</p>\n"))]), roadmap: &roadmap)
        XCTAssertFalse(result.isError, result.text)
        XCTAssertEqual(result.changes.first?.kind, .created)

        // 引数の名前の揺れも読む
        result = executor.execute(call("edit_file", [("file_path", .string("index.html")), ("old", .string("hi")), ("new", .string("hello"))]), roadmap: &roadmap)
        XCTAssertFalse(result.isError, result.text)
        XCTAssertEqual(try fs.read("index.html"), "<p>hello</p>\n")

        result = executor.execute(call("read_file", [("path", .string("index.html"))]), roadmap: &roadmap)
        XCTAssertEqual(result.text, "1| <p>hello</p>\n2| \n")

        result = executor.execute(call("list_files", []), roadmap: &roadmap)
        XCTAssertEqual(result.text, "index.html")

        result = executor.execute(call("search_files", [("pattern", .string("hello"))]), roadmap: &roadmap)
        XCTAssertEqual(result.text, "index.html:1: <p>hello</p>")

        result = executor.execute(call("roadmap_add", [("title", .string("Dark mode")), ("parent", .string("#1"))]), roadmap: &roadmap)
        XCTAssertTrue(result.roadmapChanged)
        XCTAssertEqual(roadmap.nodes.count, 2)
        XCTAssertEqual(roadmap.edges.count, 1)

        result = executor.execute(call("roadmap_update", [("id", .int(2)), ("status", .string("done"))]), roadmap: &roadmap)
        XCTAssertFalse(result.isError, result.text)
        XCTAssertEqual(roadmap.node(reference: "#2")?.status, .done)

        result = executor.execute(call("read_file", [("path", .string("../etc/passwd"))]), roadmap: &roadmap)
        XCTAssertTrue(result.isError)
        result = executor.execute(call("nope", []), roadmap: &roadmap)
        XCTAssertTrue(result.isError)
    }

    func testPreviewShowsTheChangeBeforeWriting() throws {
        let fs = try makeWorkspace()
        try fs.write("a.txt", "one\ntwo\n")
        let executor = WorkspaceToolExecutor(fileSystem: fs)
        let preview = executor.preview(call("edit_file", [("path", .string("a.txt")), ("old_string", .string("two")), ("new_string", .string("2"))]))
        XCTAssertEqual(preview?.before, "one\ntwo\n")
        XCTAssertEqual(preview?.after, "one\n2\n")
        XCTAssertEqual(try fs.read("a.txt"), "one\ntwo\n")
        let newFile = executor.preview(call("write_file", [("path", .string("b.txt")), ("content", .string("x"))]))
        XCTAssertNil(newFile?.before)
        XCTAssertFalse(fs.exists("b.txt"))
    }

    func testProfilesAndPrompt() {
        let minimal = WorkspaceToolbox.definitions(profile: .minimal, javaScript: false).map(\.name)
        XCTAssertEqual(minimal, ["read_file", "write_file", "edit_file"])
        // 標準は 5 つだけ（コンテキストを食わないように）
        let standard = WorkspaceToolbox.definitions(profile: .standard, javaScript: false).map(\.name)
        XCTAssertEqual(standard, ["read_file", "write_file", "edit_file", "search_files", "roadmap"])
        let full = WorkspaceToolbox.definitions(profile: .full, javaScript: true).map(\.name)
        XCTAssertTrue(full.contains("run_javascript"))
        XCTAssertEqual(Set(full).count, full.count)
        XCTAssertLessThanOrEqual(full.count, 8)
        let prompt = CodingAgentPrompt.system(projectName: "Demo", tree: "index.html", roadmap: "#1 [idea] X",
                                              notes: "Use tabs", openFile: "index.html", javaScript: true)
        XCTAssertTrue(prompt.contains("Demo"))
        XCTAssertTrue(prompt.contains("#1 [idea] X"))
        XCTAssertTrue(prompt.contains("Use tabs"))
    }
}

final class ProjectTemplateTests: XCTestCase {
    func testTemplatesCreateFilesAndRoadmap() throws {
        for template in ProjectTemplates.all {
            let fs = try makeWorkspace()
            let roadmap = try ProjectTemplates.create(template, name: "My Cool App", in: fs)
            XCTAssertEqual(roadmap.nodes.first?.kind, .goal)
            XCTAssertEqual(roadmap.nodes.count, template.roadmap.count + 1)
            for path in template.files.keys {
                let text = try fs.read(path)
                XCTAssertFalse(text.contains("{{"), "\(template.id) \(path)")
            }
            if let main = template.mainFile { XCTAssertTrue(fs.exists(main), template.id) }
        }
        XCTAssertEqual(ProjectTemplates.identifier(from: "my cool-app 2"), "MyCoolApp2")
        XCTAssertEqual(ProjectTemplates.identifier(from: "日本語"), "MyApp")
        XCTAssertEqual(ProjectTemplates.identifier(from: "2048"), "My2048")
        XCTAssertEqual(ProjectTemplates.folderName(for: "a/b", template: ProjectTemplates.swiftApp), "a_b.swiftpm")
        let app = ProjectTemplates.swiftApp.render(ProjectTemplates.swiftApp.files["Sources/AppModule/App.swift"]!, name: "Cool App")
        XCTAssertTrue(app.contains("struct CoolAppApp: App"))
    }
}

final class SyntaxHighlighterTests: XCTestCase {
    private func kinds(_ text: String, _ language: CodeLanguage) -> [(String, HighlightToken.Kind)] {
        let units = Array(text.utf16)
        return SyntaxHighlighter.tokens(in: text, language: language).map { token in
            (String(decoding: units[token.location..<(token.location + token.length)], as: UTF16.self), token.kind)
        }
    }

    func testSwift() {
        let tokens = kinds("let name = \"a\\\"b\" // hi\nstruct View: Thing { @State var x = 42 }", .swift)
        let map = Dictionary(tokens.map { ($0.0, $0.1) }, uniquingKeysWith: { first, _ in first })
        XCTAssertEqual(map["let"], .keyword)
        XCTAssertEqual(map["\"a\\\"b\""], .string)
        XCTAssertEqual(map["// hi"], .comment)
        XCTAssertEqual(map["View"], .type)
        XCTAssertEqual(map["@State"], .attribute)
        XCTAssertEqual(map["42"], .number)
        XCTAssertNil(map["name"])
    }

    func testJavaScriptHTMLAndJSON() {
        let js = Dictionary(kinds("const s = `x${1}`; /* c */ function f() {}", .javascript).map { ($0.0, $0.1) }, uniquingKeysWith: { a, _ in a })
        XCTAssertEqual(js["const"], .keyword)
        XCTAssertEqual(js["`x${1}`"], .string)
        XCTAssertEqual(js["/* c */"], .comment)

        let html = kinds("<!-- c --><a href=\"x\">t</a>", .html)
        XCTAssertEqual(html.map(\.0), ["<!-- c -->", "<a", "href", "\"x\"", ">", "</a", ">"])

        let json = Dictionary(kinds("{\"key\": \"value\", \"n\": 1.5, \"b\": true}", .json).map { ($0.0, $0.1) }, uniquingKeysWith: { a, _ in a })
        XCTAssertEqual(json["\"key\""], .attribute)
        XCTAssertEqual(json["\"value\""], .string)
        XCTAssertEqual(json["1.5"], .number)
        XCTAssertEqual(json["true"], .keyword)
    }

    func testMarkdownAndLanguageDetection() {
        let md = kinds("# Title\n- item with `code`\n```\nlet x\n```\n", .markdown)
        XCTAssertEqual(md.first?.1, .heading)
        XCTAssertTrue(md.contains { $0.0 == "`code`" && $0.1 == .string })
        XCTAssertTrue(md.contains { $0.0 == "let x" && $0.1 == .string })
        XCTAssertEqual(CodeLanguage.from(path: "a/b/App.swift"), .swift)
        XCTAssertEqual(CodeLanguage.from(path: "index.HTML"), .html)
        XCTAssertEqual(CodeLanguage.from(path: "Makefile"), .plain)
    }

    func testLargeFileIsFast() {
        let line = "let value = compute(\"text\", 123) // comment\n"
        let text = String(repeating: line, count: 3000)
        let start = Date()
        let tokens = SyntaxHighlighter.tokens(in: text, language: .swift)
        XCTAssertLessThan(Date().timeIntervalSince(start), 1.5)
        XCTAssertEqual(tokens.count, 3000 * 4)
    }
}

final class SlimToolTests: XCTestCase {
    private func call(_ name: String, _ arguments: [(String, JSONValue)]) -> ToolCall {
        ToolCall(name: name, arguments: .object(arguments))
    }

    func testSearchListsFilesWhenEmptyAndFindsNames() throws {
        let fs = try makeWorkspace()
        _ = try fs.write("index.html", "<h1>Hello</h1>")
        _ = try fs.write("src/app.js", "console.log('hello')\nlet total = 1")
        let executor = WorkspaceToolExecutor(fileSystem: fs)
        var roadmap = Roadmap()
        let list = executor.execute(call("search_files", []), roadmap: &roadmap)
        XCTAssertTrue(list.text.contains("index.html"))
        XCTAssertTrue(list.text.contains("app.js"))
        let found = executor.execute(call("search_files", [("pattern", .string("total"))]), roadmap: &roadmap)
        XCTAssertTrue(found.text.contains("src/app.js:2:"), found.text)
        let byName = executor.execute(call("search_files", [("pattern", .string("index"))]), roadmap: &roadmap)
        XCTAssertTrue(byName.text.contains("index.html"), byName.text)
    }

    func testCombinedRoadmapTool() throws {
        let fs = try makeWorkspace()
        let executor = WorkspaceToolExecutor(fileSystem: fs)
        var roadmap = Roadmap()
        let added = executor.execute(call("roadmap", [("title", .string("Dark mode"))]), roadmap: &roadmap)
        XCTAssertFalse(added.isError)
        XCTAssertTrue(added.roadmapChanged)
        let reference = try XCTUnwrap(roadmap.nodes.first?.reference)
        let updated = executor.execute(call("roadmap", [("id", .string(reference)), ("status", .string("in_progress"))]), roadmap: &roadmap)
        XCTAssertEqual(updated.text, "OK \(reference) in_progress")
        XCTAssertEqual(roadmap.nodes.first?.status, .inProgress)
        let listed = executor.execute(call("roadmap", []), roadmap: &roadmap)
        XCTAssertTrue(listed.text.contains("Dark mode"))
    }

    func testOutputsAreShort() throws {
        let fs = try makeWorkspace()
        let executor = WorkspaceToolExecutor(fileSystem: fs)
        var roadmap = Roadmap()
        let long = (1...1000).map { "let line\($0) = \($0)" }.joined(separator: "\n")
        let written = executor.execute(call("write_file", [("path", .string("big.js")), ("content", .string(long))]), roadmap: &roadmap)
        XCTAssertEqual(written.text, "OK big.js (1000 " + L("行", "lines") + ")")
        let read = executor.execute(call("read_file", [("path", .string("big.js"))]), roadmap: &roadmap)
        XCTAssertLessThan(read.text.count, 5200)
        XCTAssertTrue(read.text.contains("offset"))
    }
}

