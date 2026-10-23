import XCTest
@testable import LocalLLMCore

final class MarkdownParserTests: XCTestCase {
    func testParagraphsAndHeadings() {
        let blocks = MarkdownParser.parse("# 見出し\n本文1行目\n本文2行目\n\n次の段落")
        XCTAssertEqual(blocks, [
            .heading(level: 1, text: "見出し"),
            .paragraph("本文1行目\n本文2行目"),
            .paragraph("次の段落"),
        ])
    }

    func testCodeBlock() {
        let blocks = MarkdownParser.parse("説明\n```swift\nlet x = 1\n\nprint(x)\n```\n後")
        XCTAssertEqual(blocks, [
            .paragraph("説明"),
            .code(language: "swift", code: "let x = 1\n\nprint(x)", isClosed: true),
            .paragraph("後"),
        ])
    }

    func testUnclosedCodeBlockWhileStreaming() {
        let blocks = MarkdownParser.parse("```\nfor i in 0..<3 {")
        XCTAssertEqual(blocks, [.code(language: nil, code: "for i in 0..<3 {", isClosed: false)])
    }

    func testListsQuotesRulesTables() {
        let text = "- りんご\n  - 赤い\n1. 一つ目\n2) 二つ目\n> 引用\n> 続き\n---\n| a | b |\n|---|---|"
        XCTAssertEqual(MarkdownParser.parse(text), [
            .listItem(marker: "•", text: "りんご", indent: 0),
            .listItem(marker: "•", text: "赤い", indent: 1),
            .listItem(marker: "1.", text: "一つ目", indent: 0),
            .listItem(marker: "2.", text: "二つ目", indent: 0),
            .quote("引用\n続き"),
            .rule,
            .table(["| a | b |", "|---|---|"]),
        ])
    }

    func testHashWithoutSpaceIsNotHeading() {
        XCTAssertEqual(MarkdownParser.parse("#hashtag"), [.paragraph("#hashtag")])
    }

    func testEmpty() {
        XCTAssertEqual(MarkdownParser.parse(""), [])
    }
}
