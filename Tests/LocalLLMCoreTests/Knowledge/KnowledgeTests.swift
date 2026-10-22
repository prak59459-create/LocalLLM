import Foundation
import XCTest
@testable import LocalLLMCore

final class KnowledgeTokenizerTests: XCTestCase {
    func testIdentifiersAreSplit() {
        XCTAssertEqual(KnowledgeTokenizer.terms("URLSessionDataTask"), ["url", "session", "data", "task", "urlsessiondatatask"])
        XCTAssertEqual(KnowledgeTokenizer.terms("fooBar snake_case kebab-case"),
                       ["foo", "bar", "foobar", "snake", "case", "snake_case", "kebab", "case", "kebab-case"])
        XCTAssertEqual(KnowledgeTokenizer.terms("http2Server v1"), ["http2", "server", "http2server", "v1"])
    }

    func testStopwordsShortWordsAndNumbers() {
        XCTAssertEqual(KnowledgeTokenizer.terms("The cat is on a mat, x = 1"), ["cat", "mat", "1"])
        XCTAssertEqual(KnowledgeTokenizer.terms("-- __ !!"), [])
        XCTAssertEqual(KnowledgeTokenizer.terms(""), [])
    }

    func testJapaneseBigrams() {
        XCTAssertEqual(KnowledgeTokenizer.terms("吾輩は猫である"), ["吾輩", "輩は", "は猫", "猫で", "であ", "ある"])
        XCTAssertEqual(KnowledgeTokenizer.terms("猫 と 犬"), ["猫", "と", "犬"])
        // カタカナの長音、英語との混在、絵文字は区切り
        XCTAssertEqual(KnowledgeTokenizer.terms("コーヒーをSwiftで😀書く"),
                       ["コー", "ーヒ", "ヒー", "ーを", "swift", "で", "書く"])
    }

    func testFullWidthIsNormalized() {
        XCTAssertEqual(KnowledgeTokenizer.terms("ＡＢＣ１２３ ｶﾀｶﾅ"), ["abc123", "カタ", "タカ", "カナ"])
        XCTAssertEqual(KnowledgeTokenizer.terms("Café naïve"), ["café", "naïve"])
    }
}

final class KnowledgeChunkerTests: XCTestCase {
    func testEmptyAndSmallText() {
        XCTAssertTrue(KnowledgeChunker.chunks(source: "a.txt", text: "").isEmpty)
        XCTAssertTrue(KnowledgeChunker.chunks(source: "a.txt", text: "  \n\n\t\n").isEmpty)
        let chunks = KnowledgeChunker.chunks(source: "a.txt", text: "one\r\ntwo\nthree\n")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].text, "one\ntwo\nthree")
        XCTAssertEqual(chunks[0].startLine, 1)
        XCTAssertEqual(chunks[0].endLine, 3)
        XCTAssertEqual(chunks[0].source, "a.txt")
    }

    func testLongCodeIsSplitAtDeclarations() {
        var lines: [String] = []
        for index in 0..<40 {
            lines.append("func function\(index)() {")
            for body in 0..<6 { lines.append("    let value\(body) = compute(\(index), \(body))") }
            lines.append("}")
            lines.append("")
        }
        let text = lines.joined(separator: "\n")
        let chunks = KnowledgeChunker.chunks(source: "Code.swift", text: text, maxCharacters: 600, overlapLines: 2)
        XCTAssertGreaterThan(chunks.count, 5)
        var covered = Set<Int>()
        for (index, chunk) in chunks.enumerated() {
            XCTAssertEqual(chunk.id, index)
            XCTAssertLessThanOrEqual(chunk.text.count, 600)
            XCTAssertLessThanOrEqual(chunk.startLine, chunk.endLine)
            // 行番号と本文が合っている
            XCTAssertEqual(chunk.text.components(separatedBy: "\n"), Array(lines[(chunk.startLine - 1)..<chunk.endLine]))
            covered.formUnion(chunk.startLine...chunk.endLine)
            if index > 0 {
                XCTAssertGreaterThan(chunk.startLine, chunks[index - 1].startLine)
                // 重なりは 2 行まで
                XCTAssertGreaterThanOrEqual(chunk.startLine, chunks[index - 1].endLine - 1)
                // 区切りやすい行（空行か宣言）から始まる
                let first = lines[chunk.startLine - 1]
                XCTAssertTrue(first.isEmpty || first.hasPrefix("func ") || first == "}", first)
            }
        }
        let lastNonEmpty = lines.lastIndex { !$0.isEmpty }! + 1
        XCTAssertTrue(Set(1...lastNonEmpty).isSubset(of: covered))
    }

    func testVeryLongLineIsSplit() {
        let line = String(repeating: "あいうえお", count: 100)
        let chunks = KnowledgeChunker.chunks(source: "long.txt", text: line, maxCharacters: 120, overlapLines: 0)
        XCTAssertEqual(chunks.map(\.text).joined(), line)
        XCTAssertTrue(chunks.allSatisfy { $0.startLine == 1 && $0.endLine == 1 && $0.text.count <= 120 })
    }
}

final class KnowledgeIndexTests: XCTestCase {
    func makeIndex() -> KnowledgeIndex {
        var index = KnowledgeIndex()
        index.add(source: "Network.swift", text: """
        import Foundation

        func fetchUser(id: Int) async throws -> User {
            let url = URL(string: "https://example.com/users/\\(id)")!
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(User.self, from: data)
        }
        """)
        index.add(source: "メモ.md", text: """
        # 買い物リスト
        牛乳と卵とパンを買う。

        # 猫の世話
        猫に毎朝ごはんをあげる。猫の爪を切る。
        """)
        index.add(source: "Math.swift", text: "func add(_ a: Int, _ b: Int) -> Int { a + b }\n")
        return index
    }

    func testFindsRelevantChunks() throws {
        let index = makeIndex()
        XCTAssertEqual(index.sources, ["Network.swift", "メモ.md", "Math.swift"])
        let network = index.search("How do I fetch a user with URLSession?")
        XCTAssertEqual(network.first?.chunk.source, "Network.swift")
        XCTAssertTrue(network.allSatisfy { $0.score > 0 })
        XCTAssertEqual(network.map(\.score), network.map(\.score).sorted(by: >))
        let cat = index.search("猫の爪")
        XCTAssertEqual(cat.first?.chunk.source, "メモ.md")
        XCTAssertTrue(try XCTUnwrap(cat.first).chunk.text.contains("猫の爪"))
        XCTAssertTrue(index.search("zebra quantum").isEmpty)
        XCTAssertTrue(index.search("the a of").isEmpty)
        XCTAssertEqual(index.search("func", limit: 1).count, 1)
        XCTAssertTrue(index.search("func", limit: 0).isEmpty)
    }

    func testReplaceAndRemoveSource() {
        var index = makeIndex()
        let before = index.chunkCount
        index.add(source: "Math.swift", text: "func multiply(_ a: Int, _ b: Int) -> Int { a * b }\n")
        XCTAssertEqual(index.chunkCount, before)
        XCTAssertTrue(index.search("multiply").first?.chunk.source == "Math.swift")
        XCTAssertTrue(index.search("add").isEmpty)
        index.remove(source: "Network.swift")
        XCTAssertEqual(index.sources, ["メモ.md", "Math.swift"])
        XCTAssertTrue(index.search("URLSession").isEmpty)
        // ID は重ならない
        let ids = index.chunks.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    func testCodableRoundTrip() throws {
        let index = makeIndex()
        let decoded = try JSONDecoder().decode(KnowledgeIndex.self, from: JSONEncoder().encode(index))
        XCTAssertEqual(decoded.search("猫 ごはん"), index.search("猫 ごはん"))
        XCTAssertEqual(decoded.chunkCount, index.chunkCount)
        var updated = decoded
        updated.add(source: "New.txt", text: "brand new content about giraffes")
        XCTAssertEqual(updated.search("giraffes").first?.chunk.source, "New.txt")
        XCTAssertEqual(Set(updated.chunks.map(\.id)).count, updated.chunkCount)
    }

    func testPromptBlock() {
        let index = makeIndex()
        XCTAssertEqual(KnowledgePromptBuilder.contextBlock(results: [], maxCharacters: 1000), "")
        let results = index.search("fetch user URLSession 猫")
        let block = KnowledgePromptBuilder.contextBlock(results: results, maxCharacters: 4000)
        XCTAssertTrue(block.contains("[1] Network.swift (L1-7)"))
        XCTAssertTrue(block.contains("```\nimport Foundation"))
        XCTAssertFalse(block.hasSuffix("\n"))
        // 入りきらない分は切る
        let short = KnowledgePromptBuilder.contextBlock(results: results, maxCharacters: 200)
        XCTAssertLessThanOrEqual(short.count, 200)
        XCTAssertTrue(short.contains("…"))
        XCTAssertFalse(short.contains("[2]"))
        XCTAssertEqual(KnowledgePromptBuilder.contextBlock(results: results, maxCharacters: 30), "")
    }

    func testFenceIsLongerThanBackticksInText() {
        var index = KnowledgeIndex()
        index.add(source: "README.md", text: "Example:\n```swift\nprint(\"fence\")\n```\n")
        let block = KnowledgePromptBuilder.contextBlock(results: index.search("fence example"), maxCharacters: 1000)
        XCTAssertTrue(block.contains("````\nExample:"))
        XCTAssertTrue(block.contains("```\n````"))
    }
}
