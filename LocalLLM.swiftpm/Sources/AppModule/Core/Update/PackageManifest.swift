import Foundation

/// Package.swift の中の値を読んだり書き換えたりする（Swift の構文をちゃんと解析するのではなく、`key: 値` を探す）。
enum PackageManifest {
    /// 更新のたびにこのアプリが書き換える値
    static let versionKeys = ["displayVersion", "bundleVersion"]
    /// この iPad の Swift Playgrounds で設定される値（チームやアイコンなど）。更新しても、この iPad の値を残す。
    /// bundleIdentifier が変わると別のアプリになってしまう（ダウンロードしたモデルなどが見えなくなる）ので、特に大事。
    static let preservedKeys = ["bundleIdentifier", "teamIdentifier", "appIcon", "accentColor"]

    /// `key:` の値の範囲（`"0.3"` や `.placeholder(icon: .chatMessage)` のような式全体）
    static func valueRange(of key: String, in text: String) -> Range<String.Index>? {
        var searchStart = text.startIndex
        while let found = text.range(of: key + ":", range: searchStart..<text.endIndex) {
            searchStart = found.upperBound
            if found.lowerBound > text.startIndex {
                let previous = text[text.index(before: found.lowerBound)]
                if previous.isLetter || previous.isNumber || previous == "_" { continue }
            }
            if isInComment(found.lowerBound, in: text) { continue }
            return expression(startingAfter: found.upperBound, in: text)
        }
        return nil
    }

    /// 文字列の値（`key: "..."` の中身）
    static func stringValue(of key: String, in text: String) -> String? {
        guard let range = valueRange(of: key, in: text) else { return nil }
        let value = text[range]
        guard value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") else { return nil }
        return String(value.dropFirst().dropLast())
    }

    static func expressionValue(of key: String, in text: String) -> String? {
        valueRange(of: key, in: text).map { String(text[$0]) }
    }

    /// `key:` の値を value（式の文字列そのもの）に置き換える。key が無ければそのまま返す。
    static func replacing(_ key: String, withExpression value: String, in text: String) -> String {
        guard let range = valueRange(of: key, in: text) else { return text }
        return text.replacingCharacters(in: range, with: value)
    }

    /// 版や、この iPad の設定の違いを無視して比べるための文字列
    static func normalized(_ text: String) -> String {
        var result = text.replacingOccurrences(of: "\r\n", with: "\n")
        for key in versionKeys + preservedKeys {
            result = replacing(key, withExpression: "•", in: result)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 版と、この iPad の設定以外は同じか
    static func isEquivalent(_ lhs: String, _ rhs: String) -> Bool {
        normalized(lhs) == normalized(rhs)
    }

    /// GitHub の Package.swift に、この iPad の設定を残して版を入れた文字列。
    /// base（前回の更新で書いた GitHub の Package.swift）があれば、この iPad で変えていない値は GitHub の新しい値にする。
    static func merge(local: String, remote: String, base: String?, version: PackageVersion) -> String {
        var result = remote
        for key in preservedKeys {
            guard let localValue = expressionValue(of: key, in: local), valueRange(of: key, in: result) != nil else { continue }
            if let base, let baseValue = expressionValue(of: key, in: base), baseValue == localValue {
                continue
            }
            result = replacing(key, withExpression: localValue, in: result)
        }
        return version.apply(to: result)
    }

    /// 行の中で `//` より後ろか
    private static func isInComment(_ index: String.Index, in text: String) -> Bool {
        let lineStart = text[..<index].lastIndex(of: "\n").map { text.index(after: $0) } ?? text.startIndex
        return text[lineStart..<index].contains("//")
    }

    /// `:` の後ろから、カンマか閉じかっこ（入れ子の外）までの式の範囲
    private static func expression(startingAfter start: String.Index, in text: String) -> Range<String.Index>? {
        var index = start
        while index < text.endIndex, text[index] == " " || text[index] == "\t" { index = text.index(after: index) }
        let valueStart = index
        var depth = 0
        var inString = false
        var lastNonSpace: String.Index?
        while index < text.endIndex {
            let character = text[index]
            if inString {
                if character == "\\" {
                    index = text.index(after: index)
                    if index < text.endIndex { lastNonSpace = index; index = text.index(after: index) }
                    continue
                }
                if character == "\"" { inString = false }
                lastNonSpace = index
            } else {
                switch character {
                case "\"":
                    inString = true
                case "(", "[", "{":
                    depth += 1
                case ")", "]", "}":
                    if depth == 0 { return range(valueStart, lastNonSpace, in: text) }
                    depth -= 1
                case ",":
                    if depth == 0 { return range(valueStart, lastNonSpace, in: text) }
                case "\n":
                    if depth == 0, lastNonSpace != nil { return range(valueStart, lastNonSpace, in: text) }
                default:
                    break
                }
                if !character.isWhitespace { lastNonSpace = index }
            }
            index = text.index(after: index)
        }
        return range(valueStart, lastNonSpace, in: text)
    }

    private static func range(_ start: String.Index, _ last: String.Index?, in text: String) -> Range<String.Index>? {
        guard let last, last >= start else { return nil }
        return start..<text.index(after: last)
    }
}
