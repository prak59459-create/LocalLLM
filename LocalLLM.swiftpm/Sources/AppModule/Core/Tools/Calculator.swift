import Foundation

enum CalculatorError: Error, LocalizedError, Equatable {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let detail):
            return L("式を計算できません: \(detail)", "Cannot calculate the expression: \(detail)")
        }
    }
}

/// 数式を計算する（モデルの「計算」ツール。小さなモデルは計算をよく間違えるので、式を書かせてここで正確に計算する）。
/// ネットにつながなくても動く。四則演算・べき乗（^ と **）・余り（%）・階乗（!）・かっこ・関数（sqrt・sin・log など）・
/// 定数（pi・e）に対応。「2pi」「3(4+5)」のような掛け算の省略と、全角の数字・記号（１２×３）も読む
enum Calculator {
    static func evaluate(_ expression: String) throws -> Double {
        var parser = Parser(tokens: try tokenize(normalized(expression)))
        let value = try parser.expression()
        guard parser.isAtEnd else { throw CalculatorError.invalid(L("余分なところがあります: ", "unexpected ") + parser.rest) }
        guard value.isFinite else { throw CalculatorError.invalid(L("結果が数になりません（0 で割った など）", "the result is not a number (division by zero?)")) }
        return value
    }

    /// 計算の結果を文字列にする（整数ならそのまま、そうでなければ有効数字 12 桁まで）
    static func format(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e15 { return String(Int64(value)) }
        var text = String(format: "%.12g", value)
        if text.contains("e") { return text }
        if text.contains(".") {
            while text.hasSuffix("0") { text.removeLast() }
            if text.hasSuffix(".") { text.removeLast() }
        }
        return text
    }

    // MARK: - 字句

    private enum Token: Equatable {
        case number(Double)
        case name(String)
        case symbol(String)
    }

    /// 全角の数字・記号を半角に、掛け算・割り算の記号を * と / に、桁区切りの「,」（数字の 3 桁ごと）は取る
    private static func normalized(_ text: String) -> String {
        var output = ""
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0xFF10...0xFF19: output.unicodeScalars.append(Unicode.Scalar(scalar.value - 0xFF10 + 0x30)!)
            case 0xFF41...0xFF5A, 0xFF21...0xFF3A: output.unicodeScalars.append(Unicode.Scalar(scalar.value - 0xFEE0)!)
            default:
                switch scalar {
                case "＋": output += "+"
                case "－", "−", "ー": output += "-"
                case "＊", "×", "✕", "·": output += "*"
                case "／", "÷": output += "/"
                case "（": output += "("
                case "）": output += ")"
                case "．": output += "."
                case "，", "、": output += ","
                case "＾": output += "^"
                case "％": output += "%"
                case "！": output += "!"
                case "π": output += " pi "
                case "√": output += " sqrt "
                default: output.unicodeScalars.append(scalar)
                }
            }
        }
        return output
    }

    private static func tokenize(_ text: String) throws -> [Token] {
        var tokens: [Token] = []
        let characters = Array(text)
        var index = 0
        while index < characters.count {
            let character = characters[index]
            if character.isWhitespace || character == "=" || character == "_" {
                index += 1
                continue
            }
            if isDigit(character) || character == "." {
                var end = index
                var digits = ""
                while end < characters.count {
                    let c = characters[end]
                    if isDigit(c) || c == "." {
                        digits.append(c)
                    } else if c == ",", isThousands(characters, at: end) {
                        // 1,234,567 の桁区切り
                    } else if startsExponent(characters, at: end) {
                        digits.append("e")
                        end += 1
                        digits.append(characters[end])
                    } else {
                        break
                    }
                    end += 1
                }
                guard let value = Double(digits) else { throw CalculatorError.invalid(digits) }
                tokens.append(.number(value))
                index = end
                continue
            }
            if character.isLetter {
                var end = index
                while end < characters.count, characters[end].isLetter || isDigit(characters[end]) { end += 1 }
                tokens.append(.name(String(characters[index..<end]).lowercased()))
                index = end
                continue
            }
            if character == "*", index + 1 < characters.count, characters[index + 1] == "*" {
                tokens.append(.symbol("^"))
                index += 2
                continue
            }
            guard "+-*/%^()!,".contains(character) else { throw CalculatorError.invalid(String(character)) }
            tokens.append(.symbol(String(character)))
            index += 1
        }
        return tokens
    }

    private static func isDigit(_ character: Character) -> Bool { character.isASCII && character.isNumber }

    /// 1e3・2.5e-1 の「e」か（e のあとが数字、または符号と数字）
    private static func startsExponent(_ characters: [Character], at index: Int) -> Bool {
        let letter: Character = characters[index]
        guard letter == Character("e") || letter == Character("E"), index + 1 < characters.count else { return false }
        let next: Character = characters[index + 1]
        if isDigit(next) { return true }
        let signed: Bool = next == Character("-") || next == Character("+")
        return signed && index + 2 < characters.count && isDigit(characters[index + 2])
    }

    /// 「,」のあとがちょうど 3 桁の数字で、その次が数字でなければ桁区切り（1,234。引数の区切りの「1, 234」は空白があるので違う）
    private static func isThousands(_ characters: [Character], at comma: Int) -> Bool {
        guard comma + 3 < characters.count else { return false }
        for offset in 1...3 where !isDigit(characters[comma + offset]) { return false }
        let after = comma + 4
        return after >= characters.count || !(isDigit(characters[after]) || characters[after] == ".")
    }

    // MARK: - 構文

    private struct Parser {
        let tokens: [Token]
        var position = 0

        var isAtEnd: Bool { position >= tokens.count }

        var rest: String {
            tokens[position...].map { token -> String in
                switch token {
                case .number(let value): return Calculator.format(value)
                case .name(let name): return name
                case .symbol(let symbol): return symbol
                }
            }.joined()
        }

        private var current: Token? { position < tokens.count ? tokens[position] : nil }

        private mutating func take(_ symbol: String) -> Bool {
            guard current == .symbol(symbol) else { return false }
            position += 1
            return true
        }

        /// 足し算・引き算
        mutating func expression() throws -> Double {
            var value = try term()
            while true {
                if take("+") { value += try term() } else if take("-") { value -= try term() } else { return value }
            }
        }

        /// 掛け算・割り算・余り（「2pi」「3(4+5)」のような省略した掛け算も）
        private mutating func term() throws -> Double {
            var value = try unary()
            while true {
                if take("*") {
                    value *= try unary()
                } else if take("/") {
                    value /= try unary()
                } else if take("%") {
                    value = value.truncatingRemainder(dividingBy: try unary())
                } else if let token = current, startsFactor(token) {
                    value *= try power()
                } else {
                    return value
                }
            }
        }

        private func startsFactor(_ token: Token) -> Bool {
            switch token {
            case .number, .name: return true
            case .symbol(let symbol): return symbol == "("
            }
        }

        /// 符号（-3^2 は -(3^2) = -9）
        private mutating func unary() throws -> Double {
            if take("-") { return -(try unary()) }
            if take("+") { return try unary() }
            return try power()
        }

        /// べき乗（右から: 2^3^2 = 2^9）
        private mutating func power() throws -> Double {
            let base = try postfix()
            if take("^") { return pow(base, try unary()) }
            return base
        }

        /// 階乗
        private mutating func postfix() throws -> Double {
            var value = try primary()
            while take("!") {
                guard value >= 0, value == value.rounded(), value <= 170 else {
                    throw CalculatorError.invalid(L("階乗は 0〜170 の整数だけ", "factorial needs an integer from 0 to 170"))
                }
                var product: Double = 1
                for factor in 1...max(1, Int(value)) { product *= Double(factor) }
                value = product
            }
            return value
        }

        private mutating func primary() throws -> Double {
            guard let token = current else { throw CalculatorError.invalid(L("式が途中で終わっています", "the expression ends too early")) }
            switch token {
            case .number(let value):
                position += 1
                return value
            case .symbol("("):
                position += 1
                let value = try expression()
                guard take(")") else { throw CalculatorError.invalid(L("「)」が足りません", "missing \")\"")) }
                return value
            case .name(let name):
                position += 1
                if let constant = Self.constants[name] { return constant }
                guard take("(") else {
                    // かっこの無い関数（sqrt 2）
                    guard Self.functions[name] != nil else { throw CalculatorError.invalid(L("知らない名前: ", "unknown name: ") + name) }
                    return try call(name, [try power()])
                }
                var arguments: [Double] = []
                if !take(")") {
                    repeat { arguments.append(try expression()) } while take(",")
                    guard take(")") else { throw CalculatorError.invalid(L("「)」が足りません", "missing \")\"")) }
                }
                return try call(name, arguments)
            case .symbol(let symbol):
                throw CalculatorError.invalid(L("ここに「\(symbol)」は置けません", "unexpected \"\(symbol)\""))
            }
        }

        private func call(_ name: String, _ arguments: [Double]) throws -> Double {
            if let function = Self.functions[name] {
                guard arguments.count == 1 else { throw CalculatorError.invalid(L("\(name) の引数は 1 つ", "\(name) takes one argument")) }
                return function(arguments[0])
            }
            switch name {
            case "pow":
                guard arguments.count == 2 else { throw CalculatorError.invalid(L("pow の引数は 2 つ", "pow takes two arguments")) }
                return Foundation.pow(arguments[0], arguments[1])
            case "min", "max":
                guard !arguments.isEmpty else { throw CalculatorError.invalid(L("\(name) に引数がありません", "\(name) needs arguments")) }
                return name == "min" ? arguments.min()! : arguments.max()!
            case "hypot":
                var squares: Double = 0
                for argument in arguments { squares += argument * argument }
                return squares.squareRoot()
            case "log" where arguments.count == 2:
                return Foundation.log(arguments[0]) / Foundation.log(arguments[1])
            case "mod":
                guard arguments.count == 2 else { throw CalculatorError.invalid(L("mod の引数は 2 つ", "mod takes two arguments")) }
                return arguments[0].truncatingRemainder(dividingBy: arguments[1])
            default:
                throw CalculatorError.invalid(L("知らない関数: ", "unknown function: ") + name)
            }
        }

        static let constants: [String: Double] = ["pi": Double.pi, "e": Foundation.exp(1.0), "tau": 2 * Double.pi]

        static let functions: [String: (Double) -> Double] = [
            "sqrt": { $0.squareRoot() }, "cbrt": { Foundation.cbrt($0) }, "abs": { Swift.abs($0) },
            "sin": { Foundation.sin($0) }, "cos": { Foundation.cos($0) }, "tan": { Foundation.tan($0) },
            "asin": { Foundation.asin($0) }, "acos": { Foundation.acos($0) }, "atan": { Foundation.atan($0) },
            "sinh": { Foundation.sinh($0) }, "cosh": { Foundation.cosh($0) }, "tanh": { Foundation.tanh($0) },
            "exp": { Foundation.exp($0) }, "ln": { Foundation.log($0) }, "log": { Foundation.log10($0) },
            "log10": { Foundation.log10($0) }, "log2": { Foundation.log2($0) },
            "floor": { $0.rounded(.down) }, "ceil": { $0.rounded(.up) }, "round": { $0.rounded() },
            "trunc": { $0.rounded(.towardZero) },
        ]
    }
}
