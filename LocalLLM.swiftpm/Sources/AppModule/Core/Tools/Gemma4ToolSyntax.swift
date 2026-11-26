import Foundation

/// Gemma 4（E2B・E4B など）のツールの呼び出しの書き方。
///
///     <|tool_call>call:web_search{query:<|"|>東京 天気<|"|>,limit:3}<tool_call|><|tool_response>
///
/// - キーは引用符なし、文字列は <|"|> で囲む（中にカンマや括弧・" があってもよい）。数・true・false・null はそのまま。
/// - 呼び出しのあと、モデルは <|tool_response> を出して結果を待つ（そこで生成を止める）。
/// - 小さなモデルが崩した書き方（普通の " の文字列、閉じ忘れ、JSON そのもの）もできるだけ読む。
enum Gemma4ToolSyntax {
    static let open = "<|tool_call>"
    static let close = "<tool_call|>"
    static let quote = "<|\"|>"
    /// 呼び出しのあとに出る、結果を待つ印
    static let responseMarker = "<|tool_response>"

    /// 1 つの <|tool_call> … <tool_call|> の中身（"call:名前{…}"）から呼び出しを読む
    static func call(in text: String, knownTools: Set<String>) -> ToolCall? {
        var reader = Reader(Array(text))
        reader.skipSpaces()
        if reader.consume("call:") { reader.skipSpaces() }
        let name = reader.readName()
        guard !name.isEmpty, knownTools.contains(name) else { return nil }
        reader.skipSpaces()
        var arguments = JSONValue.object([])
        if reader.peek == "{" { arguments = reader.readValue() }
        if case .object = arguments {} else { arguments = .object([("input", arguments)]) }
        return ToolCall(name: name, arguments: arguments)
    }

    /// 読み取り（文字ごと）
    private struct Reader {
        let characters: [Character]
        var index = 0

        init(_ characters: [Character]) {
            self.characters = characters
        }

        var peek: Character? { index < characters.count ? characters[index] : nil }

        mutating func skipSpaces() {
            while let character = peek, character.isWhitespace { index += 1 }
        }

        func starts(with text: String) -> Bool {
            var position = index
            for character in text {
                guard position < characters.count, characters[position] == character else { return false }
                position += 1
            }
            return true
        }

        mutating func consume(_ text: String) -> Bool {
            guard starts(with: text) else { return false }
            index += text.count
            return true
        }

        /// ツールの名前（英数字・_・-・.）
        mutating func readName() -> String {
            var name = ""
            while let character = peek, character.isLetter || character.isNumber || "_-.".contains(character) {
                name.append(character)
                index += 1
            }
            return name
        }

        mutating func readValue() -> JSONValue {
            skipSpaces()
            guard let character = peek else { return .null }
            if consume(Gemma4ToolSyntax.quote) { return .string(readUntilQuote()) }
            switch character {
            case "{": return readObject()
            case "[": return readArray()
            case "\"", "'":
                index += 1
                return .string(readQuoted(character))
            default:
                return literal(readBare(stopsAtColon: false))
            }
        }

        /// <|"|> まで（閉じ忘れなら最後まで）
        mutating func readUntilQuote() -> String {
            var text = ""
            while index < characters.count {
                if consume(Gemma4ToolSyntax.quote) { return text }
                text.append(characters[index])
                index += 1
            }
            return text
        }

        /// " や ' の文字列（\ の後ろはそのまま）
        mutating func readQuoted(_ mark: Character) -> String {
            var text = ""
            while let character = peek {
                index += 1
                if character == mark { return text }
                if character == "\\", let next = peek {
                    index += 1
                    switch next {
                    case "n": text.append("\n")
                    case "t": text.append("\t")
                    default: text.append(next)
                    }
                    continue
                }
                text.append(character)
            }
            return text
        }

        /// 区切り（, } ]。キーなら : も）までの、引用符の無い値
        mutating func readBare(stopsAtColon: Bool) -> String {
            var text = ""
            while let character = peek, !",}]".contains(character), !(stopsAtColon && character == ":"),
                  !starts(with: Gemma4ToolSyntax.close) {
                text.append(character)
                index += 1
            }
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        func literal(_ text: String) -> JSONValue {
            switch text {
            case "true", "True": return .bool(true)
            case "false", "False": return .bool(false)
            case "null", "None", "": return .null
            default:
                if let integer = Int(text) { return .int(integer) }
                if let number = Double(text), number.isFinite { return .double(number) }
                return .string(text)
            }
        }

        mutating func readKey() -> String {
            skipSpaces()
            if consume(Gemma4ToolSyntax.quote) { return readUntilQuote() }
            if let character = peek, character == "\"" || character == "'" {
                index += 1
                return readQuoted(character)
            }
            return readBare(stopsAtColon: true)
        }

        mutating func readObject() -> JSONValue {
            index += 1   // {
            var pairs: [(String, JSONValue)] = []
            while true {
                skipSpaces()
                guard let character = peek else { break }
                if character == "}" { index += 1; break }
                if character == "," { index += 1; continue }
                let start = index
                let key = readKey()
                skipSpaces()
                guard consume(":") else {
                    // キーだけで終わっている・崩れている: 先へ進めて終わる
                    if index == start { index += 1 }
                    break
                }
                let value = readValue()
                if !key.isEmpty { pairs.append((key, value)) }
            }
            return .object(pairs)
        }

        mutating func readArray() -> JSONValue {
            index += 1   // [
            var items: [JSONValue] = []
            while true {
                skipSpaces()
                guard let character = peek else { break }
                if character == "]" { index += 1; break }
                if character == "," { index += 1; continue }
                let start = index
                items.append(readValue())
                if index == start { index += 1 }
            }
            return .array(items)
        }
    }
}
