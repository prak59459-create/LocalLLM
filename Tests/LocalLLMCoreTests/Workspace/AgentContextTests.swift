import Foundation
import XCTest
@testable import LocalLLMCore

final class AgentContextTests: XCTestCase {
    private func tool(_ text: String) -> ChatMessage { ChatMessage(role: .tool, content: text, toolName: "read_file", toolCallID: "x") }

    func testKeepsOnlyRecentTurns() {
        let messages = [
            ChatMessage(role: .user, content: "1"), ChatMessage(role: .assistant, content: "a"),
            ChatMessage(role: .user, content: "2"), ChatMessage(role: .assistant, content: "b"),
            ChatMessage(role: .user, content: "3"),
        ]
        XCTAssertEqual(AgentContext.recentTurns(messages, keepTurns: 2).map(\.content), ["2", "b", "3"])
        XCTAssertEqual(AgentContext.recentTurns(messages, keepTurns: 9).count, 5)
        XCTAssertEqual(AgentContext.recentTurns([ChatMessage(role: .assistant, content: "x")], keepTurns: 1).count, 1)
    }

    func testOldToolResultsAndWrittenContentAreShortened() {
        let big = String(repeating: "line of code\n", count: 500)
        let write = ToolCall(name: "write_file", arguments: .object([("path", .string("a.js")), ("content", .string(big))]))
        let messages = [
            ChatMessage(role: .user, content: "old request"),
            ChatMessage(role: .assistant, content: String(repeating: "explanation ", count: 200)),
            ChatMessage(role: .user, content: "new request"),
            ChatMessage(role: .assistant, content: "", toolCalls: [write]),
            tool(big), tool(big), tool(big),
        ]
        let compacted = AgentContext.compact(messages)
        // 前の依頼の説明は短くなる
        XCTAssertLessThan(compacted[1].content.count, 500)
        // 書いた中身（引数）は省く
        let content = compacted[3].toolCalls?.first?.argument("content") ?? ""
        XCTAssertLessThan(content.count, 300)
        XCTAssertEqual(compacted[3].toolCalls?.first?.argument("path"), "a.js")
        // 最新の 2 つの結果は長めに、古い結果は短く
        XCTAssertLessThan(compacted[4].content.count, 260)
        XCTAssertGreaterThan(compacted[6].content.count, 2000)
        XCTAssertLessThan(compacted[6].content.count, 2600)
        // 全体として大きく減る
        let before = messages.reduce(0) { $0 + $1.content.count }
        let after = compacted.reduce(0) { $0 + $1.content.count }
        XCTAssertLessThan(after, before / 2)
        // 短いものはそのまま
        XCTAssertEqual(AgentContext.compact([ChatMessage(role: .user, content: "hi"), tool("small")]).last?.content, "small")
    }

    func testTightBudgetIsSmaller() {
        let big = String(repeating: "x", count: 5000)
        let messages = [ChatMessage(role: .user, content: "go"), tool(big)]
        let normal = AgentContext.compact(messages)
        let tight = AgentContext.compact(messages, budget: .tight)
        XCTAssertLessThan(tight[1].content.count, normal[1].content.count)
    }
}

final class MemoryBudgetTests: XCTestCase {
    private let mb: Int64 = 1 << 20

    func testThresholds() {
        let roomy = MemoryBudget(available: 2000 * mb, appLimit: 3000 * mb)
        XCTAssertFalse(roomy.isLow)
        XCTAssertFalse(roomy.isCritical)
        let low = MemoryBudget(available: 300 * mb, appLimit: 3000 * mb)
        XCTAssertTrue(low.isLow)
        XCTAssertFalse(low.isCritical)
        let critical = MemoryBudget(available: 100 * mb, appLimit: 3000 * mb)
        XCTAssertTrue(critical.isCritical)
    }

    func testPrefillChunkShrinksWithMemory() {
        // Qwen の語彙（約 15 万）・float16 の logits
        let roomy = MemoryBudget(available: 2000 * mb, appLimit: 3000 * mb)
        XCTAssertEqual(roomy.prefillChunk(requested: 128, vocabSize: 151_936, bytesPerLogit: 2), 128)
        let tighter = MemoryBudget(available: 450 * mb, appLimit: 3000 * mb)
        let chunk = tighter.prefillChunk(requested: 128, vocabSize: 151_936, bytesPerLogit: 4)
        XCTAssertLessThan(chunk, 128)
        XCTAssertGreaterThanOrEqual(chunk, 1)
        let critical = MemoryBudget(available: 100 * mb, appLimit: 3000 * mb)
        XCTAssertEqual(critical.prefillChunk(requested: 128, vocabSize: 151_936, bytesPerLogit: 2), 1)
    }

    func testStatelessLimit() {
        // 空き 1.5GB なら、毎回全体を入力するモデルで 1000 トークン程度まで
        let budget = MemoryBudget(available: 1500 * mb, appLimit: 3000 * mb)
        let limit = budget.statelessTokenLimit(vocabSize: 151_936, bytesPerLogit: 2)
        XCTAssertGreaterThan(limit, 200)
        XCTAssertLessThan(limit, 1500)
        XCTAssertEqual(MemoryBudget(available: 10 * mb, appLimit: 3000 * mb).statelessTokenLimit(vocabSize: 151_936, bytesPerLogit: 2), 0)
    }
}
