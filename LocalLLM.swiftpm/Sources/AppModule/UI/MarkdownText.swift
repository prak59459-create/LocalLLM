import SwiftUI
import UIKit

/// 簡易 Markdown 表示（コードブロックはコピーボタン付き）。
struct MarkdownText: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        let blocks = MarkdownParser.parse(text)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // 種類ごとに小さなビューに分けている（1 つの switch に全部書くと型チェックが遅い）
    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .paragraph(let text):
            MarkdownParagraph(text: text)
        case .heading(let level, let text):
            MarkdownHeading(level: level, text: text)
        case .code(let language, let code, _):
            CodeBlockView(language: language, code: code)
        case .listItem(let marker, let text, let indent):
            MarkdownListItem(marker: marker, text: text, indent: indent)
        case .quote(let text):
            MarkdownQuote(text: text)
        case .rule:
            Divider()
        case .table(let rows):
            MarkdownTable(rows: rows)
        }
    }

    /// 行内の Markdown（太字・斜体・コード・リンク）を解釈する。失敗したらそのまま表示。
    static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}

private struct MarkdownParagraph: View {
    let text: String

    var body: some View {
        Text(MarkdownText.inline(text))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct MarkdownHeading: View {
    let level: Int
    let text: String

    private var font: Font {
        if level <= 1 { return Font.title2.bold() }
        if level == 2 { return Font.title3.bold() }
        return Font.headline
    }

    var body: some View {
        Text(MarkdownText.inline(text))
            .font(font)
            .textSelection(.enabled)
            .padding(.top, 4)
    }
}

private struct MarkdownListItem: View {
    let marker: String
    let text: String
    let indent: Int

    var body: some View {
        let leading: CGFloat = CGFloat(indent) * 16
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(marker)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            Text(MarkdownText.inline(text))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.leading, leading)
    }
}

private struct MarkdownQuote: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Rectangle()
                .fill(.secondary.opacity(0.5))
                .frame(width: 3)
            Text(MarkdownText.inline(text))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

private struct MarkdownTable: View {
    let rows: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Text(rows.joined(separator: "\n"))
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
        }
    }
}

/// コードブロック。
private struct CodeBlockView: View {
    let language: String?
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    UIPasteboard.general.string = code
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.5))
                        copied = false
                    }
                } label: {
                    Label(copied ? L("コピーしました", "Copied") : L("コピー", "Copy"),
                          systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(uiColor: .tertiarySystemBackground))

            ScrollView(.horizontal, showsIndicators: true) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
