import Foundation

/// エディタで色を付ける言語。
enum CodeLanguage: String, CaseIterable, Sendable {
    case swift, javascript, typescript, python, html, css, json, markdown, shell, c, plain

    static func from(path: String) -> CodeLanguage {
        switch (path as NSString).pathExtension.lowercased() {
        case "swift": return .swift
        case "js", "mjs", "cjs", "jsx": return .javascript
        case "ts", "tsx": return .typescript
        case "py": return .python
        case "html", "htm", "xml", "svg", "plist": return .html
        case "css", "scss": return .css
        case "json", "jsonl", "resolved": return .json
        case "md", "markdown": return .markdown
        case "sh", "bash", "zsh", "yml", "yaml", "toml": return .shell
        case "c", "h", "cpp", "hpp", "m", "mm", "java", "kt", "go", "rs", "cs": return .c
        default: return .plain
        }
    }

    var keywords: Set<String> {
        switch self {
        case .swift:
            return ["actor", "any", "as", "async", "await", "break", "case", "catch", "class", "continue", "default", "defer",
                    "do", "else", "enum", "extension", "fallthrough", "false", "fileprivate", "for", "func", "guard", "if",
                    "import", "in", "init", "inout", "internal", "is", "let", "mutating", "nil", "nonisolated", "open",
                    "operator", "override", "private", "protocol", "public", "repeat", "rethrows", "return", "self", "Self",
                    "some", "static", "struct", "subscript", "super", "switch", "throw", "throws", "true", "try", "typealias",
                    "var", "weak", "where", "while", "lazy", "final", "convenience", "required", "deinit", "get", "set",
                    "didSet", "willSet", "indirect", "consuming", "borrowing", "sending"]
        case .javascript, .typescript:
            return ["async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete",
                    "do", "else", "export", "extends", "false", "finally", "for", "from", "function", "if", "import", "in",
                    "instanceof", "let", "new", "null", "of", "return", "static", "super", "switch", "this", "throw", "true",
                    "try", "typeof", "undefined", "var", "void", "while", "with", "yield", "interface", "type", "enum",
                    "implements", "private", "public", "protected", "readonly", "as", "declare", "namespace"]
        case .python:
            return ["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
                    "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
                    "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "self", "match", "case"]
        case .c:
            return ["auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
                    "float", "for", "goto", "if", "int", "long", "return", "short", "signed", "sizeof", "static", "struct",
                    "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "class", "public", "private",
                    "protected", "new", "delete", "true", "false", "null", "nullptr", "this", "package", "import", "func",
                    "fn", "let", "mut", "impl", "pub", "use", "var", "val", "fun", "go", "defer", "interface", "type"]
        case .shell:
            return ["if", "then", "else", "elif", "fi", "for", "in", "do", "done", "while", "case", "esac", "function",
                    "return", "export", "local", "echo", "true", "false"]
        case .json:
            return ["true", "false", "null"]
        case .css:
            return ["important", "inherit", "initial", "none", "auto", "media", "keyframes", "import", "root"]
        case .html, .markdown, .plain:
            return []
        }
    }

    var lineComment: String? {
        switch self {
        case .swift, .javascript, .typescript, .c, .css: return self == .css ? nil : "//"
        case .python, .shell: return "#"
        default: return nil
        }
    }

    var hasBlockComments: Bool {
        switch self {
        case .swift, .javascript, .typescript, .c, .css: return true
        default: return false
        }
    }
}

/// 色を付ける範囲（UTF-16 の位置。NSAttributedString にそのまま使える）。
struct HighlightToken: Hashable, Sendable {
    enum Kind: Hashable, Sendable {
        case keyword, string, comment, number, type, attribute, tag, heading, punctuation
    }
    var kind: Kind
    var location: Int
    var length: Int
}

/// かんたんな字句解析で色を付ける。最適化なしのビルドでも速いように、UTF-16 の配列を 1 回なめるだけにしている。
enum SyntaxHighlighter {
    /// これより長いテキストには色を付けない（入力が重くならないように）
    static let maxLength = 200_000

    static func tokens(in text: String, language: CodeLanguage) -> [HighlightToken] {
        guard language != .plain else { return [] }
        let units = Array(text.utf16)
        guard units.count <= maxLength else { return [] }
        switch language {
        case .html: return markupTokens(units)
        case .markdown: return markdownTokens(units)
        default: return codeTokens(units, language: language)
        }
    }

    private static func isIdentifierStart(_ c: UInt16) -> Bool {
        (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == 95 || c == 36 || c == 64 || c > 127
    }

    private static func isIdentifierPart(_ c: UInt16) -> Bool {
        isIdentifierStart(c) || (c >= 48 && c <= 57)
    }

    private static func isDigit(_ c: UInt16) -> Bool { c >= 48 && c <= 57 }

    private static func codeTokens(_ u: [UInt16], language: CodeLanguage) -> [HighlightToken] {
        var tokens: [HighlightToken] = []
        let keywords = language.keywords
        let lineComment: [UInt16] = language.lineComment.map { Array($0.utf16) } ?? []
        let blockComments = language.hasBlockComments
        let n = u.count
        var i = 0
        while i < n {
            let c = u[i]
            // 行コメント
            if !lineComment.isEmpty, c == lineComment[0], i + lineComment.count <= n,
               lineComment.count == 1 || u[i + 1] == lineComment[1] {
                let start = i
                while i < n && u[i] != 10 { i += 1 }
                tokens.append(HighlightToken(kind: .comment, location: start, length: i - start))
                continue
            }
            // ブロックコメント /* */
            if blockComments, c == 47, i + 1 < n, u[i + 1] == 42 {
                let start = i
                i += 2
                while i + 1 < n && !(u[i] == 42 && u[i + 1] == 47) { i += 1 }
                i = min(n, i + 2)
                tokens.append(HighlightToken(kind: .comment, location: start, length: i - start))
                continue
            }
            // 文字列
            if c == 34 || c == 39 || (c == 96 && language != .swift) {
                let start = i
                let quote = c
                // """ や ''' の複数行文字列
                if i + 2 < n && u[i + 1] == quote && u[i + 2] == quote && (language == .swift || language == .python) {
                    i += 3
                    while i + 2 < n && !(u[i] == quote && u[i + 1] == quote && u[i + 2] == quote) { i += 1 }
                    i = min(n, i + 3)
                } else {
                    i += 1
                    while i < n && u[i] != quote {
                        if u[i] == 92 { i += 1 }
                        else if u[i] == 10 && quote != 96 { break }
                        i += 1
                    }
                    i = min(n, i + 1)
                }
                if language == .json, isFollowedByColon(u, from: i) {
                    tokens.append(HighlightToken(kind: .attribute, location: start, length: i - start))
                } else {
                    tokens.append(HighlightToken(kind: .string, location: start, length: i - start))
                }
                continue
            }
            // 数
            if isDigit(c) {
                let start = i
                while i < n && (isIdentifierPart(u[i]) || u[i] == 46) { i += 1 }
                tokens.append(HighlightToken(kind: .number, location: start, length: i - start))
                continue
            }
            // 名前
            if isIdentifierStart(c) {
                let start = i
                i += 1
                while i < n && isIdentifierPart(u[i]) { i += 1 }
                let length = i - start
                if c == 64 {
                    tokens.append(HighlightToken(kind: .attribute, location: start, length: length))
                } else if length <= 16, keywords.contains(String(decoding: u[start..<i], as: UTF16.self)) {
                    tokens.append(HighlightToken(kind: .keyword, location: start, length: length))
                } else if c >= 65 && c <= 90 && language != .css {
                    tokens.append(HighlightToken(kind: .type, location: start, length: length))
                } else if language == .css, isFollowedByColon(u, from: i) {
                    tokens.append(HighlightToken(kind: .attribute, location: start, length: length))
                }
                continue
            }
            i += 1
        }
        return tokens
    }

    private static func isFollowedByColon(_ u: [UInt16], from index: Int) -> Bool {
        var j = index
        while j < u.count && (u[j] == 32 || u[j] == 9) { j += 1 }
        return j < u.count && u[j] == 58
    }

    private static func markupTokens(_ u: [UInt16]) -> [HighlightToken] {
        var tokens: [HighlightToken] = []
        let n = u.count
        var i = 0
        while i < n {
            // <!-- コメント -->
            if u[i] == 60, i + 3 < n, u[i + 1] == 33, u[i + 2] == 45, u[i + 3] == 45 {
                let start = i
                i += 4
                while i + 2 < n && !(u[i] == 45 && u[i + 1] == 45 && u[i + 2] == 62) { i += 1 }
                i = min(n, i + 3)
                tokens.append(HighlightToken(kind: .comment, location: start, length: i - start))
                continue
            }
            if u[i] == 60 {
                // タグ名
                let start = i
                i += 1
                if i < n && (u[i] == 47 || u[i] == 33 || u[i] == 63) { i += 1 }
                while i < n && (isIdentifierPart(u[i]) || u[i] == 45 || u[i] == 58) { i += 1 }
                tokens.append(HighlightToken(kind: .tag, location: start, length: i - start))
                // 属性と値
                while i < n && u[i] != 62 {
                    if u[i] == 34 || u[i] == 39 {
                        let quote = u[i]
                        let valueStart = i
                        i += 1
                        while i < n && u[i] != quote { i += 1 }
                        i = min(n, i + 1)
                        tokens.append(HighlightToken(kind: .string, location: valueStart, length: i - valueStart))
                    } else if isIdentifierStart(u[i]) {
                        let nameStart = i
                        while i < n && (isIdentifierPart(u[i]) || u[i] == 45 || u[i] == 58) { i += 1 }
                        tokens.append(HighlightToken(kind: .attribute, location: nameStart, length: i - nameStart))
                    } else {
                        i += 1
                    }
                }
                if i < n {
                    tokens.append(HighlightToken(kind: .tag, location: i, length: 1))
                    i += 1
                }
                continue
            }
            i += 1
        }
        return tokens
    }

    private static func markdownTokens(_ u: [UInt16]) -> [HighlightToken] {
        var tokens: [HighlightToken] = []
        let n = u.count
        var i = 0
        var lineStart = true
        var inFence = false
        // 行全体に色を付けて、改行の次へ進む
        func wholeLine(_ kind: HighlightToken.Kind) {
            let start = i
            while i < n && u[i] != 10 { i += 1 }
            if i > start { tokens.append(HighlightToken(kind: kind, location: start, length: i - start)) }
            i += 1
        }
        while i < n {
            if lineStart {
                // ``` のコードブロック
                if i + 2 < n && u[i] == 96 && u[i + 1] == 96 && u[i + 2] == 96 {
                    wholeLine(.keyword)
                    inFence.toggle()
                    continue
                }
                if inFence {
                    wholeLine(.string)
                    continue
                }
                if u[i] == 35 {
                    wholeLine(.heading)
                    continue
                }
                if (u[i] == 45 || u[i] == 42 || u[i] == 62) && i + 1 < n && u[i + 1] == 32 {
                    tokens.append(HighlightToken(kind: .punctuation, location: i, length: 1))
                }
            }
            lineStart = u[i] == 10
            if u[i] == 96 {
                let start = i
                i += 1
                while i < n && u[i] != 96 && u[i] != 10 { i += 1 }
                i = min(n, i + 1)
                tokens.append(HighlightToken(kind: .string, location: start, length: i - start))
                continue
            }
            if u[i] == 42 && i + 1 < n && u[i + 1] == 42 {
                let start = i
                i += 2
                while i + 1 < n && !(u[i] == 42 && u[i + 1] == 42) && u[i] != 10 { i += 1 }
                i = min(n, i + 2)
                tokens.append(HighlightToken(kind: .keyword, location: start, length: i - start))
                continue
            }
            i += 1
        }
        return tokens
    }
}
