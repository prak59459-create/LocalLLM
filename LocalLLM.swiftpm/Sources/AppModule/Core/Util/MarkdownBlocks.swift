import Foundation

/// 返答テキストを、表示用の「ブロック」（段落・見出し・コード・箇条書きなど）に分ける簡易 Markdown パーサ。
/// 行内の装飾（**太字**、`コード`、リンク）は表示側で AttributedString に任せる。
enum MarkdownBlock: Hashable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case code(language: String?, code: String, isClosed: Bool)
    case listItem(marker: String, text: String, indent: Int)
    case quote(String)
    case rule
    case table([String])
}

enum MarkdownParser {
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var quote: [String] = []
        var table: [String] = []

        func flushParagraph() {
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph.joined(separator: "\n")))
                paragraph.removeAll()
            }
        }
        func flushQuote() {
            if !quote.isEmpty {
                blocks.append(.quote(quote.joined(separator: "\n")))
                quote.removeAll()
            }
        }
        func flushTable() {
            if !table.isEmpty {
                blocks.append(.table(table))
                table.removeAll()
            }
        }
        func flushAll() {
            flushParagraph()
            flushQuote()
            flushTable()
        }

        let lines = text.components(separatedBy: "\n")
        var index = 0
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // コードブロック ``` または ~~~
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushAll()
                let fence = String(trimmed.prefix(3))
                let language = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                var codeLines: [String] = []
                var closed = false
                index += 1
                while index < lines.count {
                    if lines[index].trimmingCharacters(in: .whitespaces).hasPrefix(fence) {
                        closed = true
                        break
                    }
                    codeLines.append(lines[index])
                    index += 1
                }
                blocks.append(.code(language: language.isEmpty ? nil : language,
                                    code: codeLines.joined(separator: "\n"), isClosed: closed))
                index += 1
                continue
            }

            if trimmed.isEmpty {
                flushAll()
                index += 1
                continue
            }

            // 見出し
            if let heading = parseHeading(trimmed) {
                flushAll()
                blocks.append(heading)
                index += 1
                continue
            }

            // 区切り線
            if isRule(trimmed) {
                flushAll()
                blocks.append(.rule)
                index += 1
                continue
            }

            // 引用
            if trimmed.hasPrefix(">") {
                flushParagraph()
                flushTable()
                quote.append(String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces))
                index += 1
                continue
            }

            // 表（| で始まる行）
            if trimmed.hasPrefix("|") {
                flushParagraph()
                flushQuote()
                table.append(trimmed)
                index += 1
                continue
            }

            // 箇条書き
            if let item = parseListItem(line) {
                flushAll()
                blocks.append(item)
                index += 1
                continue
            }

            flushQuote()
            flushTable()
            paragraph.append(line)
            index += 1
        }
        flushAll()
        return blocks
    }

    private static func parseHeading(_ line: String) -> MarkdownBlock? {
        var level = 0
        for character in line {
            if character == "#" { level += 1 } else { break }
        }
        guard (1...6).contains(level) else { return nil }
        let rest = line.dropFirst(level)
        guard rest.first == " " else { return nil }
        return .heading(level: level, text: rest.trimmingCharacters(in: .whitespaces))
    }

    private static func isRule(_ line: String) -> Bool {
        let compact = line.replacingOccurrences(of: " ", with: "")
        guard compact.count >= 3, let first = compact.first, "-*_".contains(first) else { return false }
        return compact.allSatisfy { $0 == first }
    }

    private static func parseListItem(_ line: String) -> MarkdownBlock? {
        let indentCount = line.prefix { $0 == " " || $0 == "\t" }.count
        let body = line.dropFirst(indentCount)
        let indent = indentCount / 2
        for marker in ["- ", "* ", "+ ", "・"] where body.hasPrefix(marker) {
            let text = body.dropFirst(marker.count).trimmingCharacters(in: .whitespaces)
            return .listItem(marker: "•", text: text, indent: indent)
        }
        // 番号付き（1. や 1) ）
        let digits = body.prefix { $0.isASCII && $0.isNumber }
        if !digits.isEmpty, digits.count <= 3 {
            let after = body.dropFirst(digits.count)
            if let punct = after.first, punct == "." || punct == ")", after.dropFirst().first == " " {
                let text = after.dropFirst(2).trimmingCharacters(in: .whitespaces)
                return .listItem(marker: "\(digits).", text: text, indent: indent)
            }
        }
        return nil
    }
}
