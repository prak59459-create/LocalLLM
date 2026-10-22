import XCTest
@testable import LocalLLMCore

/// jinja2 / transformers の apply_chat_template と結果が一致するかを確かめる。
/// 事前に Tools/fixtures/chat_template_expectations.py を実行しておく（無ければスキップ）。
final class ChatTemplateParityTests: XCTestCase {
    struct JinjaCases: Decodable {
        struct Item: Decodable {
            var template: String
            var output: String?
            var error: String?
        }
        var cases: [Item]
    }

    struct ChatExpectation: Decodable {
        struct Message: Decodable {
            var role: String
            var content: String
        }
        struct Result: Decodable {
            var conversation: String
            var addGenerationPrompt: Bool
            var enableThinking: Bool?
            var output: String?
            var error: String?
        }
        var name: String
        var template: String
        var bosToken: String?
        var eosToken: String?
        var conversations: [String: [Message]]
        var results: [Result]
    }

    private var directory: URL {
        TestFixtures.root.appendingPathComponent("expectations/chat")
    }

    func testJinjaCases() throws {
        let url = directory.appendingPathComponent("jinja_cases.json")
        guard let data = try? Data(contentsOf: url) else { throw XCTSkip("jinja cases not generated") }
        let cases = try JSONDecoder().decode(JinjaCases.self, from: data)
        let context: [String: JinjaValue] = [
            "messages": .array([
                .object(JinjaObject([("role", .string("user")), ("content", .string(" hi "))])),
                .object(JinjaObject([("role", .string("assistant")), ("content", .string("yo"))])),
            ]),
            "tools": .none,
        ]
        var failures = 0
        for item in cases.cases {
            do {
                let output = try JinjaTemplate(item.template).render(context)
                if let expected = item.output {
                    if !output.unicodeScalars.elementsEqual(expected.unicodeScalars) {
                        failures += 1
                        XCTFail("template \(item.template.debugDescription)\n got: \(output.debugDescription)\nwant: \(expected.debugDescription)")
                    }
                } else {
                    failures += 1
                    XCTFail("template \(item.template.debugDescription) should fail (\(item.error ?? "")) but rendered \(output.debugDescription)")
                }
            } catch {
                if item.output != nil {
                    failures += 1
                    XCTFail("template \(item.template.debugDescription) threw \(error)")
                }
            }
        }
        print("jinja cases: \(cases.cases.count - failures)/\(cases.cases.count) passed")
    }

    func testChatTemplates() throws {
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            throw XCTSkip("chat expectations not generated")
        }
        let expectationFiles = files.filter { $0.pathExtension == "json" && $0.lastPathComponent != "jinja_cases.json" }
        if expectationFiles.isEmpty { throw XCTSkip("chat expectations not generated") }
        for file in expectationFiles.sorted(by: { $0.path < $1.path }) {
            let expectation = try JSONDecoder().decode(ChatExpectation.self, from: Data(contentsOf: file))
            let renderer = ChatTemplateRenderer(template: expectation.template, bosToken: expectation.bosToken,
                                                eosToken: expectation.eosToken)
            XCTAssertTrue(renderer.usesJinja, "[\(expectation.name)] template failed to parse")
            for result in expectation.results {
                let messages = (expectation.conversations[result.conversation] ?? []).map {
                    ChatMessage(role: ChatMessage.Role(rawValue: $0.role) ?? .user, content: $0.content)
                }
                do {
                    let output = try renderer.render(messages: messages, addGenerationPrompt: result.addGenerationPrompt,
                                                     enableThinking: result.enableThinking,
                                                     extraContext: ["date_string": .string("26 Jul 2024")])
                    if let expected = result.output {
                        XCTAssertTrue(output.unicodeScalars.elementsEqual(expected.unicodeScalars),
                                      "[\(expectation.name)] \(result.conversation) gen=\(result.addGenerationPrompt) think=\(String(describing: result.enableThinking))\n got: \(output.debugDescription)\nwant: \(expected.debugDescription)")
                    } else {
                        XCTFail("[\(expectation.name)] \(result.conversation) should raise: \(result.error ?? "")")
                    }
                } catch {
                    XCTAssertNil(result.output, "[\(expectation.name)] \(result.conversation) threw \(error)")
                }
            }
        }
    }
}
