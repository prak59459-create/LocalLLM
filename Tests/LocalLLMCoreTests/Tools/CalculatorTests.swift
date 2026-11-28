import XCTest
@testable import LocalLLMCore

/// 「計算」ツールの式の計算（小さなモデルは計算をよく間違えるので、式を書かせてここで計算する）
final class CalculatorToolTests: XCTestCase {
    private func value(_ expression: String) throws -> Double { try Calculator.evaluate(expression) }

    func testArithmeticAndPrecedence() throws {
        XCTAssertEqual(try value("1 + 2 * 3"), 7)
        XCTAssertEqual(try value("(1 + 2) * 3"), 9)
        XCTAssertEqual(try value("10 / 4"), 2.5)
        XCTAssertEqual(try value("7 % 3"), 1)
        XCTAssertEqual(try value("2^10"), 1024)
        // べき乗は右から、符号はべき乗のあと
        XCTAssertEqual(try value("2**3**2"), 512)
        XCTAssertEqual(try value("-3^2"), -9)
        XCTAssertEqual(try value("2^-1"), 0.5)
        XCTAssertEqual(try value("5!"), 120)
        XCTAssertEqual(try value("0!"), 1)
        XCTAssertEqual(try value("1e3 + 1"), 1001)
        XCTAssertEqual(try value("2.5e-1 * 4"), 1)
    }

    func testFunctionsConstantsAndShorthand() throws {
        XCTAssertEqual(try value("sqrt(16) + abs(-2)"), 6)
        XCTAssertEqual(try value("max(1, 5, 3) - min(4, 2)"), 3)
        XCTAssertEqual(try value("log(1000)"), 3, accuracy: 1e-12)
        XCTAssertEqual(try value("ln(e)"), 1, accuracy: 1e-12)
        XCTAssertEqual(try value("pow(2, 8)"), 256)
        XCTAssertEqual(try value("sin(pi / 2)"), 1, accuracy: 1e-12)
        // 掛け算の省略
        XCTAssertEqual(try value("2pi"), 2 * Double.pi, accuracy: 1e-12)
        XCTAssertEqual(try value("3(4 + 5)"), 27)
        XCTAssertEqual(try value("(1 + 1)(2 + 2)"), 8)
        // 桁区切りと「=」
        XCTAssertEqual(try value("1,234,567 + 1 ="), 1_234_568)
    }

    func testJapaneseInput() throws {
        XCTAssertEqual(try value("１２＋３×４"), 24)
        XCTAssertEqual(try value("（１０－２）÷４"), 2)
        XCTAssertEqual(try value("√16"), 4)
        XCTAssertEqual(try value("2π"), 2 * Double.pi, accuracy: 1e-12)
    }

    func testErrors() {
        for bad in ["2 +", "(1 + 2", "1 / 0", "foo(2)", "3 $ 4", "", "1.5!"] {
            XCTAssertThrowsError(try value(bad), bad) { error in
                XCTAssertNotNil((error as? CalculatorError)?.errorDescription, bad)
            }
        }
    }

    func testFormatting() {
        XCTAssertEqual(Calculator.format(42), "42")
        XCTAssertEqual(Calculator.format(-3), "-3")
        XCTAssertEqual(Calculator.format(2.5), "2.5")
        XCTAssertEqual(Calculator.format(1.0 / 3.0), "0.333333333333")
        XCTAssertEqual(Calculator.format(1e20), "1e+20")
    }
}
