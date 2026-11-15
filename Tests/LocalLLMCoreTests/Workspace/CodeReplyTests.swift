import Foundation
import XCTest
@testable import LocalLLMCore

final class CodeReplyTests: XCTestCase {
    private let fence = "```"

    func testFileNameOnTheLineBeforeTheBlock() {
        let reply = """
        index.html
        \(fence)html
        <!DOCTYPE html>
        <html><body><button>1</button></body></html>
        \(fence)
        style.css
        \(fence)css
        button { width: 60px; }
        \(fence)
        電卓のボタンを作りました。
        """
        let result = CodeReply.parse(reply, knownFiles: ["index.html"], openFile: nil)
        XCTAssertEqual(result.files.map(\.path), ["index.html", "style.css"])
        XCTAssertEqual(result.files[1].content, "button { width: 60px; }\n")
        XCTAssertTrue(result.commands.isEmpty)
    }

    func testNameInInfoStringOrFirstLineComment() {
        let reply = """
        \(fence)javascript script.js
        console.log(1)
        \(fence)
        \(fence)css
        /* app/main.css */
        body {}
        \(fence)
        \(fence)html title="page.html"
        <p>x</p>
        \(fence)
        """
        let result = CodeReply.parse(reply, knownFiles: [], openFile: nil)
        XCTAssertEqual(result.files.map(\.path), ["script.js", "app/main.css", "page.html"])
    }

    func testUnnamedBlockGoesToTheMatchingFile() {
        let reply = "\(fence)html\n<!DOCTYPE html>\n<p>calc</p>\n\(fence)"
        XCTAssertEqual(CodeReply.parse(reply, knownFiles: ["index.html", "a.css"], openFile: "a.css").files.map(\.path), ["index.html"])
        XCTAssertEqual(CodeReply.parse(reply, knownFiles: [], openFile: nil).files.map(\.path), ["index.html"])
        XCTAssertEqual(CodeReply.parse(reply, knownFiles: ["page.html"], openFile: "page.html").files.map(\.path), ["page.html"])
    }

    func testShellBlocksAreIgnoredAndElidedFilesAreSkipped() {
        let reply = """
        \(fence)bash
        open index.html
        \(fence)
        index.html
        \(fence)html
        <html>
        <!-- ... existing code ... -->
        </html>
        \(fence)
        """
        let result = CodeReply.parse(reply, knownFiles: ["index.html"], openFile: nil)
        XCTAssertTrue(result.files.isEmpty)
        XCTAssertEqual(result.skipped.map(\.path), ["index.html"])
    }

    func testLastBlockForTheSameFileWins() {
        let reply = "a.js\n\(fence)js\n1\n\(fence)\na.js\n\(fence)js\n2\n\(fence)"
        XCTAssertEqual(CodeReply.parse(reply, knownFiles: [], openFile: nil).files, [CodeReply.File(path: "a.js", content: "2\n")])
    }

    func testOneLineCommands() {
        XCTAssertEqual(CodeReply.command(in: "READ index.html"), CodeReply.Command(kind: .read, arguments: ["index.html"]))
        XCTAssertEqual(CodeReply.command(in: "/read `src/app.js`"), CodeReply.Command(kind: .read, arguments: ["src/app.js"]))
        XCTAssertEqual(CodeReply.command(in: "- LIST"), CodeReply.Command(kind: .list, arguments: []))
        XCTAssertEqual(CodeReply.command(in: "FIND addEventListener"), CodeReply.Command(kind: .find, arguments: ["addEventListener"]))
        XCTAssertEqual(CodeReply.command(in: "RENAME a.js -> b.js"), CodeReply.Command(kind: .move, arguments: ["a.js", "b.js"]))
        XCTAssertEqual(CodeReply.command(in: "DELETE: old.css"), CodeReply.Command(kind: .delete, arguments: ["old.css"]))
        XCTAssertEqual(CodeReply.command(in: "RUN script.js"), CodeReply.Command(kind: .run, arguments: ["script.js"]))
        // 文章はコマンドにしない
        XCTAssertNil(CodeReply.command(in: "Read the file index.html first."))
        XCTAssertNil(CodeReply.command(in: "READ the whole thing"))
        XCTAssertNil(CodeReply.command(in: "RUN index.html"))
        XCTAssertNil(CodeReply.command(in: "List of changes:"))
    }

    func testCommandsBecomeToolCallsAndAreFoundOutsideBlocks() {
        let reply = """
        READ script.js
        \(fence)bash
        READ inside.js
        \(fence)
        """
        let result = CodeReply.parse(reply, knownFiles: ["script.js"], openFile: nil)
        XCTAssertEqual(result.commands.map(\.line), ["READ script.js"])
        XCTAssertTrue(result.commands[0].needsReply)
        let call = result.commands[0].toolCall
        XCTAssertEqual(call?.name, WorkspaceToolbox.Name.readFile)
        XCTAssertEqual(call?.argument("path"), "script.js")
        XCTAssertEqual(CodeReply.Command(kind: .move, arguments: ["a", "b"]).toolCall?.argument("to"), "b")
        XCTAssertFalse(CodeReply.Command(kind: .delete, arguments: ["a.js"]).needsReply)
    }

    func testCondensedReplyDropsCode() {
        let reply = "index.html\n\(fence)html\n<p>long</p>\n\(fence)\nREAD script.js"
        XCTAssertEqual(CodeReply.condensed(reply, written: ["index.html"]), "index.html\nREAD script.js\n(wrote index.html)")
    }

    func testPromptHasCurrentFilesAndCommands() {
        let context = SimpleCodingPrompt.contextFiles(all: ["index.html", "notes.txt", "big.js"], openFile: nil,
                                                      read: { $0 == "big.js" ? String(repeating: "x", count: 5000) : "<p>\($0)</p>" },
                                                      budget: 1000)
        // 中心のファイルが先、長いファイルは途中まで、入らないものは名前だけ
        XCTAssertEqual(context.files.map(\.path), ["index.html", "big.js"])
        XCTAssertEqual(context.files.map(\.truncated), [false, true])
        XCTAssertEqual(context.others, ["notes.txt"])
        let system = SimpleCodingPrompt.system(projectName: "Calc", files: context.files, others: context.others,
                                               earlier: ["make a button"], notes: nil)
        XCTAssertTrue(system.contains("<p>index.html</p>"))
        XCTAssertTrue(system.contains("READ path"))
        XCTAssertTrue(system.contains("Other files: notes.txt"))
        XCTAssertTrue(system.contains("- make a button"))
    }

    func testRequestGetsEnglishKeywords() {
        XCTAssertEqual(SimpleCodingPrompt.englishKeywords("電卓風のボタンを作ってください"), ["calculator", "-style", "button", "make"])
        // 長い言葉を優先（「ストップウォッチ」の中の「ストップ」などに引っぱられない）
        XCTAssertEqual(SimpleCodingPrompt.englishKeywords("黄色い文字色にして"), ["yellow", "text color"])
        XCTAssertEqual(SimpleCodingPrompt.englishKeywords("make it blue"), [])
        let message = SimpleCodingPrompt.userMessage("背景を青に")
        XCTAssertTrue(message.hasPrefix("Request: 背景を青に\n(English: background, blue)"))
        XCTAssertEqual(SimpleCodingPrompt.userMessage("make it blue"), "Request: make it blue\n\nWrite the complete new files for this request.")
    }

    func testNewPageReplacesTheShownPage() {
        let reply = CodeReply.Result(files: [CodeReply.File(path: "calculator.html", content: "<p>calc</p>\n"),
                                             CodeReply.File(path: "style.css", content: "p{}\n")], skipped: [])
        let kept = CodeReply.keepMainPage(reply, knownFiles: ["index.html"], request: "電卓を作って")
        XCTAssertEqual(kept.files.map(\.path), ["index.html", "style.css"])
        // 名前を指定されたとき・index.html が無いときは、そのまま
        XCTAssertEqual(CodeReply.keepMainPage(reply, knownFiles: ["index.html"], request: "calculator.html を追加").files.map(\.path),
                       ["calculator.html", "style.css"])
        XCTAssertEqual(CodeReply.keepMainPage(reply, knownFiles: [], request: "x").files.map(\.path), ["calculator.html", "style.css"])
    }

    func testEchoedPromptIsNotWrittenAndNamesMatchExistingFiles() {
        let reply = """
        Index.html
        \(fence)html
        <p>new</p>
        \(fence)
        Done.

        Current files:
        script.js
        \(fence)javascript
        old()
        \(fence)
        """
        let result = CodeReply.parse(reply, knownFiles: ["index.html", "script.js"], openFile: nil)
        XCTAssertEqual(result.files, [CodeReply.File(path: "index.html", content: "<p>new</p>\n")])
    }

    func testFilesNamedInTheRequestComeFirst() {
        let context = SimpleCodingPrompt.contextFiles(all: ["index.html", "src/script.js"], openFile: "index.html",
                                                      read: { _ in String(repeating: "y", count: 900) }, budget: 1000,
                                                      request: "script.js のエラーを直して")
        XCTAssertEqual(context.files.map(\.path), ["src/script.js"])
        XCTAssertEqual(context.others, ["index.html"])
    }
}
