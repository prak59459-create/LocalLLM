import XCTest
@testable import LocalLLMCore

final class SpeechTextTests: XCTestCase {
    func testMarkdownIsCleanedForSpeech() {
        let markdown = """
        <think>ユーザーは挨拶している。</think>
        # こんにちは
        **太字**と`コード`、[リンク](https://example.com)です。
        - 項目その一
        - 項目その二

        ```swift
        print("hello")
        ```
        詳しくは https://example.com/docs を見てください。
        | 列1 | 列2 |
        |-----|-----|
        """
        let spoken = SpeechText.spoken(from: markdown)
        XCTAssertFalse(spoken.contains("挨拶している"))
        XCTAssertFalse(spoken.contains("print"))
        XCTAssertFalse(spoken.contains("*"))
        XCTAssertFalse(spoken.contains("#"))
        XCTAssertFalse(spoken.contains("https://"))
        XCTAssertTrue(spoken.contains("こんにちは"))
        XCTAssertTrue(spoken.contains("太字とコード、リンクです。"))
        XCTAssertTrue(spoken.contains("項目その一"))
        XCTAssertTrue(spoken.contains("（コードは省略します）"))
        XCTAssertTrue(spoken.contains("（リンク）"))
        XCTAssertTrue(spoken.contains("列1"))
    }

    func testEnglishAndUnclosedBlocks() {
        XCTAssertEqual(SpeechText.spoken(from: "**Hello** _world_.\n```\ncode"), "Hello _world_.\n(code omitted)")
        XCTAssertEqual(SpeechText.spoken(from: "<think>still thinking"), "")
        XCTAssertEqual(SpeechText.spoken(from: "plain text"), "plain text")
    }

    func testLanguageDetection() {
        XCTAssertEqual(SpeechText.language(of: "今日はいい天気ですね。"), "ja-JP")
        XCTAssertEqual(SpeechText.language(of: "The weather is nice today."), "en-US")
        XCTAssertEqual(SpeechText.language(of: "Swift の配列について説明します"), "ja-JP")
        XCTAssertEqual(SpeechText.language(of: "12345"), "en-US")
    }
}
