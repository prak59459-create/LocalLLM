import Foundation

// MARK: - 構文木

/// 式
indirect enum JinjaExpr {
    case literal(JinjaValue)
    case name(String)
    case list([JinjaExpr])
    case tuple([JinjaExpr])
    case dict([(JinjaExpr, JinjaExpr)])
    case attribute(JinjaExpr, String)
    case item(JinjaExpr, JinjaExpr)
    case slice(JinjaExpr, JinjaExpr?, JinjaExpr?, JinjaExpr?)
    case call(JinjaExpr, [JinjaExpr], [(String, JinjaExpr)])
    case filter(JinjaExpr, String, [JinjaExpr], [(String, JinjaExpr)])
    case test(JinjaExpr, String, [JinjaExpr], negated: Bool)
    case not(JinjaExpr)
    case negate(JinjaExpr)
    case plus(JinjaExpr)
    case binary(String, JinjaExpr, JinjaExpr)
    case and(JinjaExpr, JinjaExpr)
    case or(JinjaExpr, JinjaExpr)
    case compare(JinjaExpr, [(String, JinjaExpr)])
    case conditional(JinjaExpr, JinjaExpr, JinjaExpr?)   // (条件, 真のとき, 偽のとき)
}

/// 代入先
enum JinjaTarget {
    case name(String)
    case attribute(String, String)   // ns.x
    case tuple([String])
}

/// 文
indirect enum JinjaNode {
    case text(String)
    case output(JinjaExpr)
    case ifBlock([(JinjaExpr, [JinjaNode])], [JinjaNode]?)
    case forBlock(JinjaTarget, JinjaExpr, JinjaExpr?, [JinjaNode], [JinjaNode]?)
    case set(JinjaTarget, JinjaExpr)
    case setBlock(String, [JinjaNode])
    case macro(String, [(String, JinjaExpr?)], [JinjaNode])
    case filterBlock(String, [JinjaExpr], [(String, JinjaExpr)], [JinjaNode])
    case breakLoop
    case continueLoop
}

// MARK: - 字句解析

/// 式の中のトークン
enum JinjaToken: Equatable {
    case name(String)
    case string(String)
    case integer(Int)
    case float(Double)
    case op(String)
}

/// テンプレートを「文字列」と「タグ」に分けたもの
enum JinjaSegment {
    case text(String)
    case variable([JinjaToken], line: Int)
    case block([JinjaToken], line: Int)
}

struct JinjaLexer {
    private let scalars: [Unicode.Scalar]
    private let trimBlocks: Bool
    private let lstripBlocks: Bool
    private var position = 0
    private var line = 1

    init(_ source: String, trimBlocks: Bool, lstripBlocks: Bool) {
        // 改行コードは \n にそろえる（Jinja も同じ）
        var view = String.UnicodeScalarView()
        var previousCR = false
        for scalar in source.unicodeScalars {
            if scalar == "\r" {
                view.append("\n")
                previousCR = true
                continue
            }
            if scalar == "\n" && previousCR {
                previousCR = false
                continue
            }
            previousCR = false
            view.append(scalar)
        }
        // keep_trailing_newline=False（Jinja の既定）: 最後の改行を 1 つ取り除く
        if view.last == "\n" { view.removeLast() }
        scalars = Array(view)
        self.trimBlocks = trimBlocks
        self.lstripBlocks = lstripBlocks
    }

    mutating func tokenize() throws -> [JinjaSegment] {
        var segments: [JinjaSegment] = []
        var lineStarting = true
        var stripNextLeading = false   // 直前のタグが -}} / -%} だった
        var trimNextNewline = false    // trim_blocks で次の改行を 1 つ消す

        while position < scalars.count {
            // 次のタグの開始を探す
            var tagStart = position
            var kind: Character = " "
            while tagStart + 1 < scalars.count {
                if scalars[tagStart] == "{" {
                    let next = scalars[tagStart + 1]
                    if next == "{" || next == "%" || next == "#" {
                        kind = Character(next)
                        break
                    }
                }
                tagStart += 1
            }
            let hasTag = kind != " "
            var text = string(position..<(hasTag ? tagStart : scalars.count))
            if stripNextLeading {
                text = String(text.unicodeScalars.drop { $0.properties.isWhitespace })
                stripNextLeading = false
            } else if trimNextNewline {
                if text.unicodeScalars.first == "\n" { text = String(text.unicodeScalars.dropFirst()) }
                trimNextNewline = false
            }
            guard hasTag else {
                if !text.isEmpty { segments.append(.text(text)) }
                break
            }

            // 開きタグの直後の - / +
            var cursor = tagStart + 2
            var leftModifier: Unicode.Scalar?
            if cursor < scalars.count, scalars[cursor] == "-" || scalars[cursor] == "+" {
                leftModifier = scalars[cursor]
                cursor += 1
            }
            // タグの前の空白の扱い
            if leftModifier == "-" {
                text = trimTrailingWhitespace(text)
            } else if leftModifier != "+" && lstripBlocks && kind != "{" {
                text = lstripBeforeBlock(text, lineStarting: lineStarting)
            }
            if !text.isEmpty { segments.append(.text(text)) }
            countLines(position..<tagStart)
            let tagLine = line

            // タグの中身
            switch kind {
            case "#":
                // コメント: #} を探す
                guard let end = find("#}", from: cursor) else {
                    throw JinjaError.syntax("unclosed comment (line \(tagLine))")
                }
                let rightModifier: Unicode.Scalar? = end > cursor && (scalars[end - 1] == "-" || scalars[end - 1] == "+") ? scalars[end - 1] : nil
                countLines(tagStart..<(end + 2))
                position = end + 2
                applyRight(rightModifier, isBlock: true, strip: &stripNextLeading, trim: &trimNextNewline)
            default:
                let closing: [Unicode.Scalar] = kind == "{" ? ["}", "}"] : ["%", "}"]
                var tokens: [JinjaToken] = []
                let rightModifier = try readTokens(from: cursor, closing: closing, into: &tokens, line: tagLine)
                countLines(tagStart..<position)
                if kind == "{" {
                    segments.append(.variable(tokens, line: tagLine))
                    if rightModifier == "-" { stripNextLeading = true }
                } else {
                    // raw ブロックはそのまま
                    if tokens == [.name("raw")] {
                        applyRight(rightModifier, isBlock: true, strip: &stripNextLeading, trim: &trimNextNewline)
                        let rawText = try readRaw()
                        segments.append(.text(rawText))
                        continue
                    }
                    segments.append(.block(tokens, line: tagLine))
                    applyRight(rightModifier, isBlock: true, strip: &stripNextLeading, trim: &trimNextNewline)
                }
            }
            // 直前に読んだ部分が改行で終わったか（lstrip_blocks の判定用）
            lineStarting = trimNextNewline && position < scalars.count && scalars[position] == "\n"
            if trimNextNewline && position < scalars.count && scalars[position] == "\n" {
                // trim_blocks で消す改行はタグの一部とみなす
                position += 1
                line += 1
                trimNextNewline = false
                lineStarting = true
            } else {
                lineStarting = false
            }
        }
        return segments
    }

    private mutating func applyRight(_ modifier: Unicode.Scalar?, isBlock: Bool, strip: inout Bool, trim: inout Bool) {
        if modifier == "-" {
            strip = true
        } else if modifier != "+" && trimBlocks && isBlock {
            trim = true
        }
    }

    /// lstrip_blocks: 行頭からタグまでが空白だけなら、その空白を消す。
    private func lstripBeforeBlock(_ text: String, lineStarting: Bool) -> String {
        let chars = Array(text.unicodeScalars)
        var lineStart = 0
        if let lastNewline = chars.lastIndex(of: "\n") {
            lineStart = lastNewline + 1
        } else if !lineStarting {
            return text
        }
        let tail = chars[lineStart...]
        guard !tail.isEmpty, tail.allSatisfy({ $0.properties.isWhitespace }) else { return text }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: chars[..<lineStart])
        return String(view)
    }

    private func trimTrailingWhitespace(_ text: String) -> String {
        var chars = Array(text.unicodeScalars)[...]
        while let last = chars.last, last.properties.isWhitespace { chars = chars.dropLast() }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: chars)
        return String(view)
    }

    private func string(_ range: Range<Int>) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars[range])
        return String(view)
    }

    private mutating func countLines(_ range: Range<Int>) {
        for index in range where index < scalars.count && scalars[index] == "\n" { line += 1 }
    }

    private func find(_ needle: String, from start: Int) -> Int? {
        let target = Array(needle.unicodeScalars)
        var index = start
        while index + target.count <= scalars.count {
            if Array(scalars[index..<(index + target.count)]) == target { return index }
            index += 1
        }
        return nil
    }

    /// {% endraw %} までをそのまま読む。
    private mutating func readRaw() throws -> String {
        var index = position
        while index + 1 < scalars.count {
            if scalars[index] == "{" && scalars[index + 1] == "%" {
                var cursor = index + 2
                if cursor < scalars.count, scalars[cursor] == "-" || scalars[cursor] == "+" { cursor += 1 }
                while cursor < scalars.count && scalars[cursor].properties.isWhitespace { cursor += 1 }
                let word = Array("endraw".unicodeScalars)
                if cursor + word.count <= scalars.count && Array(scalars[cursor..<(cursor + word.count)]) == word {
                    var text = string(position..<index)
                    if scalars[index + 2] == "-" { text = trimTrailingWhitespace(text) }
                    guard let close = find("%}", from: cursor) else { throw JinjaError.syntax("unclosed raw block") }
                    position = close + 2
                    return text
                }
            }
            index += 1
        }
        throw JinjaError.syntax("missing endraw")
    }

    /// タグの中の式をトークンに分ける。閉じタグの直前の - / + を返す。
    private mutating func readTokens(from start: Int, closing: [Unicode.Scalar], into tokens: inout [JinjaToken],
                                     line: Int) throws -> Unicode.Scalar? {
        var index = start
        // 括弧の深さ（{'a': {'b': 1}} の }} を閉じタグと間違えないため）
        var depth = 0
        func atClosing(_ at: Int) -> Bool {
            depth == 0 && at + 1 < scalars.count && scalars[at] == closing[0] && scalars[at + 1] == closing[1]
        }
        while index < scalars.count {
            let scalar = scalars[index]
            if scalar.properties.isWhitespace {
                index += 1
                continue
            }
            if atClosing(index) {
                position = index + 2
                return nil
            }
            if (scalar == "-" || scalar == "+") && atClosing(index + 1) {
                position = index + 3
                return scalar
            }
            if scalar == "(" || scalar == "[" || scalar == "{" { depth += 1 }
            if (scalar == ")" || scalar == "]" || scalar == "}") && depth > 0 { depth -= 1 }
            // 文字列
            if scalar == "\"" || scalar == "'" {
                var value = String.UnicodeScalarView()
                var cursor = index + 1
                var closed = false
                while cursor < scalars.count {
                    let c = scalars[cursor]
                    if c == scalar {
                        closed = true
                        break
                    }
                    if c == "\\" && cursor + 1 < scalars.count {
                        let escaped = scalars[cursor + 1]
                        switch escaped {
                        case "n": value.append("\n")
                        case "t": value.append("\t")
                        case "r": value.append("\r")
                        case "\\": value.append("\\")
                        case "'": value.append("'")
                        case "\"": value.append("\"")
                        case "0": value.append("\0")
                        default:
                            value.append("\\")
                            value.append(escaped)
                        }
                        cursor += 2
                        continue
                    }
                    value.append(c)
                    cursor += 1
                }
                guard closed else { throw JinjaError.syntax("unterminated string (line \(line))") }
                tokens.append(.string(String(value)))
                index = cursor + 1
                continue
            }
            // 数値
            if scalar.properties.numericType != nil && scalar.isASCII {
                var cursor = index
                var text = ""
                var isFloat = false
                while cursor < scalars.count {
                    let c = scalars[cursor]
                    if c.isASCII && c.properties.numericType != nil || c == "_" {
                        if c != "_" { text.unicodeScalars.append(c) }
                    } else if c == "." && !isFloat && cursor + 1 < scalars.count && scalars[cursor + 1].isASCII
                                && scalars[cursor + 1].properties.numericType != nil {
                        isFloat = true
                        text.unicodeScalars.append(c)
                    } else if (c == "e" || c == "E") && cursor + 1 < scalars.count
                                && (scalars[cursor + 1].properties.numericType != nil || scalars[cursor + 1] == "-" || scalars[cursor + 1] == "+") {
                        isFloat = true
                        text.unicodeScalars.append(c)
                        cursor += 1
                        text.unicodeScalars.append(scalars[cursor])
                    } else {
                        break
                    }
                    cursor += 1
                }
                if isFloat {
                    tokens.append(.float(Double(text) ?? 0))
                } else {
                    tokens.append(.integer(Int(text) ?? 0))
                }
                index = cursor
                continue
            }
            // 名前
            if scalar.properties.isAlphabetic || scalar == "_" {
                var cursor = index
                var text = ""
                while cursor < scalars.count {
                    let c = scalars[cursor]
                    guard c.properties.isAlphabetic || c == "_" || (c.isASCII && c.properties.numericType != nil) else { break }
                    text.unicodeScalars.append(c)
                    cursor += 1
                }
                tokens.append(.name(text))
                index = cursor
                continue
            }
            // 記号（長いものから）
            let two = index + 1 < scalars.count ? String(String.UnicodeScalarView([scalar, scalars[index + 1]])) : ""
            if ["==", "!=", "<=", ">=", "//", "**"].contains(two) {
                tokens.append(.op(two))
                index += 2
                continue
            }
            let one = String(scalar)
            if "+-*/%~<>=()[]{},.:|".unicodeScalars.contains(scalar) {
                tokens.append(.op(one))
                index += 1
                continue
            }
            throw JinjaError.syntax("unexpected character '\(one)' (line \(line))")
        }
        throw JinjaError.syntax("unclosed tag (line \(line))")
    }
}

// MARK: - 構文解析

struct JinjaParser {
    private var segments: [JinjaSegment]
    private var index = 0

    init(segments: [JinjaSegment]) {
        self.segments = segments
    }

    mutating func parse() throws -> [JinjaNode] {
        let (nodes, end) = try parseBody(until: [])
        if let end { throw JinjaError.syntax("unexpected '\(end)'") }
        return nodes
    }

    /// 終わりのキーワード（endif など）まで読む。見つかったキーワードとそのタグのトークンを返す。
    private mutating func parseBody(until terminators: Set<String>) throws -> ([JinjaNode], String?) {
        var nodes: [JinjaNode] = []
        while index < segments.count {
            let segment = segments[index]
            index += 1
            switch segment {
            case .text(let text):
                nodes.append(.text(text))
            case .variable(let tokens, let line):
                var stream = JinjaTokenStream(tokens: tokens, line: line)
                let expr = try stream.parseExpression()
                try stream.expectEnd()
                nodes.append(.output(expr))
            case .block(let tokens, let line):
                guard case .name(let keyword)? = tokens.first else {
                    throw JinjaError.syntax("empty block tag (line \(line))")
                }
                if terminators.contains(keyword) {
                    index -= 1
                    return (nodes, keyword)
                }
                var stream = JinjaTokenStream(tokens: Array(tokens.dropFirst()), line: line)
                nodes.append(try parseStatement(keyword, &stream))
            }
        }
        return (nodes, nil)
    }

    /// 現在の（終わりの）ブロックタグを取り出す
    private mutating func takeBlock() -> (String, JinjaTokenStream) {
        guard case .block(let tokens, let line) = segments[index], case .name(let keyword)? = tokens.first else {
            fatalError("takeBlock は parseBody がキーワードを返したときだけ呼ぶ")
        }
        index += 1
        return (keyword, JinjaTokenStream(tokens: Array(tokens.dropFirst()), line: line))
    }

    private mutating func parseStatement(_ keyword: String, _ stream: inout JinjaTokenStream) throws -> JinjaNode {
        switch keyword {
        case "if":
            var branches: [(JinjaExpr, [JinjaNode])] = []
            var condition = try stream.parseExpression()
            try stream.expectEnd()
            var elseBody: [JinjaNode]?
            while true {
                let (body, end) = try parseBody(until: ["elif", "else", "endif"])
                guard let end else { throw JinjaError.syntax("missing endif (line \(stream.line))") }
                var (_, tail) = takeBlock()
                if elseBody != nil {
                    // else の後
                    elseBody = body
                    guard end == "endif" else { throw JinjaError.syntax("expected endif") }
                    break
                }
                branches.append((condition, body))
                if end == "elif" {
                    condition = try tail.parseExpression()
                    try tail.expectEnd()
                } else if end == "else" {
                    try tail.expectEnd()
                    elseBody = []
                } else {
                    break
                }
            }
            return .ifBlock(branches, elseBody)
        case "for":
            let target = try stream.parseTarget()
            try stream.expectName("in")
            let iterable = try stream.parseExpression(allowConditional: false)
            var filter: JinjaExpr?
            if stream.peekName("if") {
                stream.advance()
                filter = try stream.parseExpression(allowConditional: false)
            }
            if stream.peekName("recursive") { stream.advance() }
            try stream.expectEnd()
            let (body, end) = try parseBody(until: ["else", "endfor"])
            guard let end else { throw JinjaError.syntax("missing endfor (line \(stream.line))") }
            _ = takeBlock()
            var elseBody: [JinjaNode]?
            if end == "else" {
                let (otherwise, close) = try parseBody(until: ["endfor"])
                guard close != nil else { throw JinjaError.syntax("missing endfor") }
                _ = takeBlock()
                elseBody = otherwise
            }
            return .forBlock(target, iterable, filter, body, elseBody)
        case "set":
            let target = try stream.parseTarget()
            if stream.atEnd || stream.peekOp("|") {
                // ブロック set: {% set x %}...{% endset %}
                guard case .name(let name) = target else { throw JinjaError.syntax("invalid block set") }
                let (body, end) = try parseBody(until: ["endset"])
                guard end != nil else { throw JinjaError.syntax("missing endset") }
                _ = takeBlock()
                return .setBlock(name, body)
            }
            try stream.expectOp("=")
            let value = try stream.parseExpressionList()
            try stream.expectEnd()
            return .set(target, value)
        case "macro":
            let name = try stream.expectAnyName()
            try stream.expectOp("(")
            var params: [(String, JinjaExpr?)] = []
            while !stream.peekOp(")") {
                let param = try stream.expectAnyName()
                var defaultValue: JinjaExpr?
                if stream.peekOp("=") {
                    stream.advance()
                    defaultValue = try stream.parseExpression()
                }
                params.append((param, defaultValue))
                if stream.peekOp(",") { stream.advance() } else { break }
            }
            try stream.expectOp(")")
            try stream.expectEnd()
            let (body, end) = try parseBody(until: ["endmacro"])
            guard end != nil else { throw JinjaError.syntax("missing endmacro") }
            _ = takeBlock()
            return .macro(name, params, body)
        case "filter":
            let name = try stream.expectAnyName()
            var args: [JinjaExpr] = []
            var kwargs: [(String, JinjaExpr)] = []
            if stream.peekOp("(") { (args, kwargs) = try stream.parseCallArguments() }
            try stream.expectEnd()
            let (body, end) = try parseBody(until: ["endfilter"])
            guard end != nil else { throw JinjaError.syntax("missing endfilter") }
            _ = takeBlock()
            return .filterBlock(name, args, kwargs, body)
        case "generation":
            // transformers の {% generation %}: 中身をそのまま出す
            let (body, end) = try parseBody(until: ["endgeneration"])
            guard end != nil else { throw JinjaError.syntax("missing endgeneration") }
            _ = takeBlock()
            return .ifBlock([(.literal(.bool(true)), body)], nil)
        case "break":
            return .breakLoop
        case "continue":
            return .continueLoop
        default:
            throw JinjaError.syntax("unknown tag '\(keyword)' (line \(stream.line))")
        }
    }
}

/// 式のトークン列を読む
struct JinjaTokenStream {
    private let tokens: [JinjaToken]
    private var position = 0
    let line: Int

    init(tokens: [JinjaToken], line: Int) {
        self.tokens = tokens
        self.line = line
    }

    var atEnd: Bool { position >= tokens.count }

    func peek(_ offset: Int = 0) -> JinjaToken? {
        position + offset < tokens.count ? tokens[position + offset] : nil
    }

    mutating func advance() { position += 1 }

    func peekOp(_ op: String) -> Bool {
        if case .op(let value)? = peek(), value == op { return true }
        return false
    }

    func peekName(_ name: String) -> Bool {
        if case .name(let value)? = peek(), value == name { return true }
        return false
    }

    mutating func expectOp(_ op: String) throws {
        guard peekOp(op) else { throw error("expected '\(op)'") }
        advance()
    }

    mutating func expectName(_ name: String) throws {
        guard peekName(name) else { throw error("expected '\(name)'") }
        advance()
    }

    mutating func expectAnyName() throws -> String {
        guard case .name(let name)? = peek() else { throw error("expected a name") }
        advance()
        return name
    }

    mutating func expectEnd() throws {
        guard atEnd else { throw error("unexpected token \(String(describing: peek()!))") }
    }

    func error(_ message: String) -> JinjaError {
        JinjaError.syntax("\(message) (line \(line))")
    }

    /// 代入先: name / ns.attr / a, b
    mutating func parseTarget() throws -> JinjaTarget {
        let first = try expectAnyName()
        if peekOp(".") {
            advance()
            let attribute = try expectAnyName()
            return .attribute(first, attribute)
        }
        if peekOp(",") {
            var names = [first]
            while peekOp(",") {
                advance()
                if case .name(let name)? = peek(), name != "in" {
                    advance()
                    names.append(name)
                } else {
                    break
                }
            }
            return .tuple(names)
        }
        return .name(first)
    }

    /// カンマ区切りならタプルにする（set 用）
    mutating func parseExpressionList() throws -> JinjaExpr {
        let first = try parseExpression()
        guard peekOp(",") else { return first }
        var items = [first]
        while peekOp(",") {
            advance()
            if atEnd { break }
            items.append(try parseExpression())
        }
        return .tuple(items)
    }

    mutating func parseExpression(allowConditional: Bool = true) throws -> JinjaExpr {
        let expr = try parseOr()
        if allowConditional && peekName("if") {
            advance()
            let condition = try parseOr()
            var otherwise: JinjaExpr?
            if peekName("else") {
                advance()
                otherwise = try parseExpression()
            }
            return .conditional(condition, expr, otherwise)
        }
        return expr
    }

    private mutating func parseOr() throws -> JinjaExpr {
        var left = try parseAnd()
        while peekName("or") {
            advance()
            left = .or(left, try parseAnd())
        }
        return left
    }

    private mutating func parseAnd() throws -> JinjaExpr {
        var left = try parseNot()
        while peekName("and") {
            advance()
            left = .and(left, try parseNot())
        }
        return left
    }

    private mutating func parseNot() throws -> JinjaExpr {
        if peekName("not") {
            advance()
            return .not(try parseNot())
        }
        return try parseCompare()
    }

    private mutating func parseCompare() throws -> JinjaExpr {
        let first = try parseMath1()
        var rest: [(String, JinjaExpr)] = []
        while true {
            if case .op(let op)? = peek(), ["==", "!=", "<", ">", "<=", ">="].contains(op) {
                advance()
                rest.append((op, try parseMath1()))
            } else if peekName("in") {
                advance()
                rest.append(("in", try parseMath1()))
            } else if peekName("not"), case .name("in")? = peek(1) {
                advance()
                advance()
                rest.append(("notin", try parseMath1()))
            } else {
                break
            }
        }
        return rest.isEmpty ? first : .compare(first, rest)
    }

    private mutating func parseMath1() throws -> JinjaExpr {
        var left = try parseConcat()
        while case .op(let op)? = peek(), op == "+" || op == "-" {
            advance()
            left = .binary(op, left, try parseConcat())
        }
        return left
    }

    private mutating func parseConcat() throws -> JinjaExpr {
        var left = try parseMath2()
        while peekOp("~") {
            advance()
            left = .binary("~", left, try parseMath2())
        }
        return left
    }

    private mutating func parseMath2() throws -> JinjaExpr {
        var left = try parsePow()
        while case .op(let op)? = peek(), ["*", "/", "//", "%"].contains(op) {
            advance()
            left = .binary(op, left, try parsePow())
        }
        return left
    }

    private mutating func parsePow() throws -> JinjaExpr {
        var left = try parseUnary()
        while peekOp("**") {
            advance()
            left = .binary("**", left, try parseUnary())
        }
        return left
    }

    private mutating func parseUnary(withFilter: Bool = true) throws -> JinjaExpr {
        var node: JinjaExpr
        if peekOp("-") {
            advance()
            node = .negate(try parseUnary(withFilter: false))
        } else if peekOp("+") {
            advance()
            node = .plus(try parseUnary(withFilter: false))
        } else {
            node = try parsePrimary()
        }
        node = try parsePostfix(node)
        if withFilter { node = try parseFilterExpr(node) }
        return node
    }

    private mutating func parsePrimary() throws -> JinjaExpr {
        guard let token = peek() else { throw error("unexpected end of expression") }
        switch token {
        case .name(let name):
            advance()
            switch name {
            case "true", "True": return .literal(.bool(true))
            case "false", "False": return .literal(.bool(false))
            case "none", "None": return .literal(.none)
            default: return .name(name)
            }
        case .string(let text):
            advance()
            // 隣り合う文字列はつなげる
            var value = text
            while case .string(let more)? = peek() {
                advance()
                value += more
            }
            return .literal(.string(value))
        case .integer(let value):
            advance()
            return .literal(.int(value))
        case .float(let value):
            advance()
            return .literal(.double(value))
        case .op("("):
            advance()
            if peekOp(")") {
                advance()
                return .tuple([])
            }
            let first = try parseExpression()
            if peekOp(",") {
                var items = [first]
                while peekOp(",") {
                    advance()
                    if peekOp(")") { break }
                    items.append(try parseExpression())
                }
                try expectOp(")")
                return .tuple(items)
            }
            try expectOp(")")
            return first
        case .op("["):
            advance()
            var items: [JinjaExpr] = []
            while !peekOp("]") {
                items.append(try parseExpression())
                if peekOp(",") { advance() } else { break }
            }
            try expectOp("]")
            return .list(items)
        case .op("{"):
            advance()
            var pairs: [(JinjaExpr, JinjaExpr)] = []
            while !peekOp("}") {
                let key = try parseExpression()
                try expectOp(":")
                let value = try parseExpression()
                pairs.append((key, value))
                if peekOp(",") { advance() } else { break }
            }
            try expectOp("}")
            return .dict(pairs)
        default:
            throw error("unexpected token \(token)")
        }
    }

    private mutating func parsePostfix(_ base: JinjaExpr) throws -> JinjaExpr {
        var node = base
        while true {
            if peekOp(".") {
                advance()
                switch peek() {
                case .name(let name)?:
                    advance()
                    node = .attribute(node, name)
                case .integer(let number)?:
                    advance()
                    node = .item(node, .literal(.int(number)))
                default:
                    throw error("expected attribute name")
                }
            } else if peekOp("[") {
                advance()
                node = try parseSubscript(node)
            } else if peekOp("(") {
                let (args, kwargs) = try parseCallArguments()
                node = .call(node, args, kwargs)
            } else {
                return node
            }
        }
    }

    private mutating func parseSubscript(_ base: JinjaExpr) throws -> JinjaExpr {
        // [a], [a:b], [a:b:c], [::-1]
        var parts: [JinjaExpr?] = []
        var current: JinjaExpr?
        var isSlice = false
        while !peekOp("]") {
            if peekOp(":") {
                advance()
                parts.append(current)
                current = nil
                isSlice = true
                continue
            }
            current = try parseExpression()
        }
        try expectOp("]")
        if !isSlice {
            guard let current else { throw error("empty subscript") }
            return .item(base, current)
        }
        parts.append(current)
        while parts.count < 3 { parts.append(nil) }
        return .slice(base, parts[0], parts[1], parts[2])
    }

    mutating func parseCallArguments() throws -> ([JinjaExpr], [(String, JinjaExpr)]) {
        try expectOp("(")
        var args: [JinjaExpr] = []
        var kwargs: [(String, JinjaExpr)] = []
        while !peekOp(")") {
            if case .name(let name)? = peek(), case .op("=")? = peek(1) {
                advance()
                advance()
                kwargs.append((name, try parseExpression()))
            } else {
                args.append(try parseExpression())
            }
            if peekOp(",") { advance() } else { break }
        }
        try expectOp(")")
        return (args, kwargs)
    }

    private mutating func parseFilterExpr(_ base: JinjaExpr) throws -> JinjaExpr {
        var node = base
        while true {
            if peekOp("|") {
                advance()
                var name = try expectAnyName()
                while peekOp(".") {
                    advance()
                    name += "." + (try expectAnyName())
                }
                var args: [JinjaExpr] = []
                var kwargs: [(String, JinjaExpr)] = []
                if peekOp("(") { (args, kwargs) = try parseCallArguments() }
                node = .filter(node, name, args, kwargs)
            } else if peekName("is") {
                advance()
                var negated = false
                if peekName("not") {
                    advance()
                    negated = true
                }
                let name = try expectAnyName()
                var args: [JinjaExpr] = []
                if peekOp("(") {
                    (args, _) = try parseCallArguments()
                } else if let next = peek(), startsTestArgument(next) {
                    args = [try parsePrimary()]
                    args[0] = try parsePostfix(args[0])
                }
                node = .test(node, name, args, negated: negated)
            } else {
                return node
            }
        }
    }

    /// `is divisibleby 3` のように括弧なしで引数を取れるか
    private func startsTestArgument(_ token: JinjaToken) -> Bool {
        switch token {
        case .string, .integer, .float: return true
        case .name(let name): return !["else", "or", "and", "if", "in", "not", "is"].contains(name)
        case .op(let op): return op == "[" || op == "{"
        }
    }
}
