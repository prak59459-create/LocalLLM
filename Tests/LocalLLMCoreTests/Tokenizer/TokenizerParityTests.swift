import XCTest
@testable import LocalLLMCore

/// Python の `tokenizers` と結果が完全に一致するかを確かめる。
/// 事前に ./Tools/fetch_fixtures.sh と Tools/fixtures/tokenizer_expectations.py を実行しておく（無ければスキップ）。
final class TokenizerParityTests: XCTestCase {
    struct Case: Decodable {
        var text: String
        var ids: [Int]
        var idsWithSpecial: [Int]
        var decoded: String
        var decodedSkipSpecial: String
    }

    struct RandomDecode: Decodable {
        var ids: [Int]
        var decoded: String
        var decodedSkipSpecial: String
    }

    struct SingleDecode: Decodable {
        var id: Int
        var decoded: String
    }

    struct Expectation: Decodable {
        var name: String
        var vocabSize: Int
        var cases: [Case]
        var randomDecodes: [RandomDecode]
        var singleTokenDecodes: [SingleDecode]
    }

    static var fixturesRoot: URL { TestFixtures.root }

    func loadExpectations() throws -> [(Expectation, URL)] {
        let directory = Self.fixturesRoot.appendingPathComponent("expectations/tokenizer")
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil),
              !files.isEmpty else {
            throw XCTSkip("tokenizer expectations not generated")
        }
        return try files.filter { $0.pathExtension == "json" }.sorted { $0.path < $1.path }.map { file in
            let expectation = try JSONDecoder().decode(Expectation.self, from: Data(contentsOf: file))
            let tokenizerDirectory = Self.fixturesRoot.appendingPathComponent("tokenizers").appendingPathComponent(expectation.name)
            return (expectation, tokenizerDirectory)
        }
    }

    func testEncodeDecodeParity() throws {
        var failures = 0
        for (expectation, directory) in try loadExpectations() {
            let tokenizer = try HFTokenizer.load(directory: directory)
            XCTAssertEqual(tokenizer.vocabSize, expectation.vocabSize, expectation.name)
            for item in expectation.cases {
                let ids = tokenizer.encode(item.text, addSpecialTokens: false)
                if ids != item.ids {
                    failures += 1
                    XCTFail("[\(expectation.name)] encode mismatch for \(item.text.prefix(60).debugDescription)\n got: \(ids.prefix(40))\nwant: \(item.ids.prefix(40))")
                }
                let special = tokenizer.encode(item.text, addSpecialTokens: true)
                if special != item.idsWithSpecial {
                    failures += 1
                    XCTFail("[\(expectation.name)] encode(special) mismatch for \(item.text.prefix(60).debugDescription)\n got: \(special.prefix(40))\nwant: \(item.idsWithSpecial.prefix(40))")
                }
                let decoded = tokenizer.decode(item.ids, skipSpecialTokens: false)
                if decoded != item.decoded || !decoded.unicodeScalars.elementsEqual(item.decoded.unicodeScalars) {
                    failures += 1
                    XCTFail("[\(expectation.name)] decode mismatch\n got: \(decoded.prefix(80).debugDescription)\nwant: \(item.decoded.prefix(80).debugDescription)")
                }
                let skipped = tokenizer.decode(item.idsWithSpecial, skipSpecialTokens: true)
                if !skipped.unicodeScalars.elementsEqual(item.decodedSkipSpecial.unicodeScalars) {
                    failures += 1
                    XCTFail("[\(expectation.name)] decode(skip) mismatch\n got: \(skipped.prefix(80).debugDescription)\nwant: \(item.decodedSkipSpecial.prefix(80).debugDescription)")
                }
                if failures > 30 { return }
            }
            for item in expectation.randomDecodes {
                let decoded = tokenizer.decode(item.ids, skipSpecialTokens: false)
                XCTAssertTrue(decoded.unicodeScalars.elementsEqual(item.decoded.unicodeScalars),
                              "[\(expectation.name)] random decode \(item.ids): got \(decoded.debugDescription) want \(item.decoded.debugDescription)")
                let skipped = tokenizer.decode(item.ids, skipSpecialTokens: true)
                XCTAssertTrue(skipped.unicodeScalars.elementsEqual(item.decodedSkipSpecial.unicodeScalars),
                              "[\(expectation.name)] random decode(skip) \(item.ids)")
            }
            for item in expectation.singleTokenDecodes {
                let decoded = tokenizer.decode([item.id], skipSpecialTokens: false)
                XCTAssertTrue(decoded.unicodeScalars.elementsEqual(item.decoded.unicodeScalars),
                              "[\(expectation.name)] single decode \(item.id): got \(decoded.debugDescription) want \(item.decoded.debugDescription)")
            }
        }
    }
}
