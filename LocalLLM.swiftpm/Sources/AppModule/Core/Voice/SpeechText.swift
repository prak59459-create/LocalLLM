import Foundation

/// 返答を読み上げる前の下ごしらえ（Markdown の記号やコードを、耳で聞いて分かる形にする）。
enum SpeechText {
    /// 読み上げ用の文章にする。
    /// - 考えている途中（<think>…</think>）は読まない
    /// - コードブロックは「コードは省略します」に置き換える
    /// - 見出し・強調・リンクなどの記号を取り除く
    static func spoken(from markdown: String, language: String? = nil) -> String {
        var text = removingBlocks(markdown, open: "<think>", close: "</think>", replacement: "")
        let isJapanese = (language ?? Self.language(of: text)).hasPrefix("ja")
        let codeNotice = isJapanese ? "（コードは省略します）" : " (code omitted) "
        text = removingBlocks(text, open: "```", close: "```", replacement: codeNotice)

        var lines: [String] = []
        for rawLine in text.components(separatedBy: "\n") {
            var line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            // 表の区切り線・水平線
            if line.allSatisfy({ "-|: *_=".contains($0) }) { continue }
            // 見出し・引用・箇条書きの印
            while let first = line.first, "#>".contains(first) { line.removeFirst() }
            line = line.trimmingCharacters(in: .whitespaces)
            for marker in ["- ", "* ", "+ ", "- [ ] ", "- [x] "] where line.hasPrefix(marker) {
                line = String(line.dropFirst(marker.count))
                break
            }
            line = line.replacingOccurrences(of: "|", with: " ")
            lines.append(line)
        }
        text = lines.joined(separator: "\n")
        // 画像・リンク: [文字](URL) → 文字
        text = replacing(#"!?\[([^\]]*)\]\([^)]*\)"#, in: text, with: "$1")
        // そのままの URL は読まない
        text = replacing(#"https?://\S+"#, in: text, with: isJapanese ? "（リンク）" : " (link) ")
        // 強調・コードの記号
        for symbol in ["**", "__", "~~", "`", "*"] { text = text.replacingOccurrences(of: symbol, with: "") }
        text = replacing(#"[ \t]+"#, in: text, with: " ")
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 文章の言語（BCP 47）。かな・漢字が一定以上あれば日本語、そうでなければ英語とみなす。
    static func language(of text: String) -> String {
        var japanese = 0, letters = 0
        for scalar in text.unicodeScalars.prefix(600) {
            let value = scalar.value
            if (0x3040...0x30FF).contains(value) || (0x4E00...0x9FFF).contains(value) {
                japanese += 1
                letters += 1
            } else if scalar.properties.isAlphabetic {
                letters += 1
            }
        }
        return letters > 0 && Double(japanese) / Double(letters) > 0.15 ? "ja-JP" : "en-US"
    }

    /// open から close までを取り除く（閉じていなければ最後まで）
    private static func removingBlocks(_ text: String, open: String, close: String, replacement: String) -> String {
        var result = ""
        var rest = Substring(text)
        while let start = rest.range(of: open) {
            result += rest[..<start.lowerBound]
            result += replacement
            let after = rest[start.upperBound...]
            guard let end = after.range(of: close) else { return result }
            rest = after[end.upperBound...]
        }
        return result + rest
    }

    private static func replacing(_ pattern: String, in text: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        return regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: template)
    }
}
