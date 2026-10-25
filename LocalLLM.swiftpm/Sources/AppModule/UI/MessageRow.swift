import SwiftUI
import UIKit

/// メッセージ 1 つの表示。
struct MessageRow: View {
    let message: StoredMessage
    var isStreaming: Bool
    var isLast: Bool
    var onRegenerate: () -> Void
    var onDelete: () -> Void

    var body: some View {
        switch message.role {
        case .user:
            userBubble
        case .tool:
            ToolResultView(message: message)
                .contextMenu { menuItems }
        case .assistant, .system:
            assistantRow
        }
    }

    private var userBubble: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack {
                Spacer(minLength: 48)
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color.accentColor.opacity(0.16), in: RoundedRectangle(cornerRadius: 18))
                    .contextMenu { menuItems }
            }
            sourcesCaption
        }
    }

    @ViewBuilder
    private var sourcesCaption: some View {
        let sources = message.sources
        if !sources.isEmpty {
            SourcesCaption(sources: sources)
        }
    }

    private var assistantRow: some View {
        AssistantMessageView(message: message, isStreaming: isStreaming)
            .contextMenu { menuItems }
            .overlay(alignment: .bottomTrailing) { regenerateButton }
    }

    @ViewBuilder
    private var regenerateButton: some View {
        if isLast && !isStreaming && message.role == .assistant {
            Button(action: onRegenerate) {
                Image(systemName: "arrow.clockwise")
                    .font(.footnote)
                    .padding(6)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(L("作り直す", "Regenerate"))
        }
    }

    @ViewBuilder
    private var menuItems: some View {
        Button {
            UIPasteboard.general.string = ThinkingSplit.split(message.content).answer
        } label: {
            Label(L("コピー", "Copy"), systemImage: "doc.on.doc")
        }
        if message.role == .assistant && isLast && !isStreaming {
            Button(action: onRegenerate) {
                Label(L("作り直す", "Regenerate"), systemImage: "arrow.clockwise")
            }
        }
        Button(role: .destructive, action: onDelete) {
            Label(L("削除", "Delete"), systemImage: "trash")
        }
    }
}

/// アシスタントの返答（思考部分は折りたたみ、本文は Markdown で表示）。
private struct AssistantMessageView: View {
    let message: StoredMessage
    var isStreaming: Bool
    @State private var showThinking = false

    var body: some View {
        let split = ThinkingSplit.split(message.content)
        VStack(alignment: .leading, spacing: 8) {
            if let thinking = split.thinking, !thinking.isEmpty || split.isThinkingOpen {
                DisclosureGroup(isExpanded: Binding(
                    get: { showThinking || (isStreaming && split.isThinkingOpen) },
                    set: { showThinking = $0 }
                )) {
                    Text(thinking)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                } label: {
                    Label(split.isThinkingOpen && isStreaming ? L("考え中…", "Thinking…") : L("思考の過程", "Thoughts"),
                          systemImage: "brain")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
                .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
            }

            if split.answer.isEmpty {
                if isStreaming && !split.isThinkingOpen {
                    TypingIndicator()
                } else if !isStreaming && message.errorText == nil && split.thinking == nil && message.toolCalls.isEmpty {
                    Text(L("（空の返答）", "(empty reply)"))
                        .foregroundStyle(.secondary)
                }
            } else {
                MarkdownText(split.answer)
            }

            let calls = message.toolCalls
            if !calls.isEmpty {
                ToolCallChips(calls: calls)
            }

            if let error = message.errorText {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }

            footer
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var footer: some View {
        if !isStreaming, let stats = message.stats {
            HStack(spacing: 8) {
                if let name = message.modelName {
                    Text(name)
                }
                Text(String(format: "%.1f tok/s", stats.decodeTokensPerSecond))
                Text(L("\(stats.generatedTokens) トークン", "\(stats.generatedTokens) tokens"))
                if let reason = stats.stopReason {
                    Text(reason.label)
                }
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.tertiary)
        }
    }
}

/// モデルが呼んだツールの一覧。
struct ToolCallChips: View {
    let calls: [ToolCall]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(calls) { call in
                Label(call.summary, systemImage: ToolResultView.icon(for: call.name))
                    .font(.caption.monospaced())
                    .lineLimit(2)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.accentColor.opacity(0.1), in: Capsule())
            }
        }
    }
}

/// ツールの実行結果（折りたたみ）。
struct ToolResultView: View {
    let message: StoredMessage
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(message.content)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            Label(title, systemImage: message.errorText == nil ? Self.icon(for: message.toolName ?? "") : "exclamationmark.triangle")
                .font(.callout)
                .foregroundStyle(message.errorText == nil ? Color.secondary : Color.orange)
        }
        .padding(10)
        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private var title: String {
        let name = message.toolName ?? L("ツール", "Tool")
        let lines = message.content.split(separator: "\n").count
        return message.errorText == nil ? L("\(name) の結果（\(lines) 行）", "\(name) result (\(lines) lines)")
                                        : L("\(name) が失敗しました", "\(name) failed")
    }

    static func icon(for name: String) -> String {
        switch name {
        case "web_search": return "globe"
        case "fetch_page": return "doc.richtext"
        case "current_time": return "clock"
        case "read_file", "list_files", "search_files": return "doc.text.magnifyingglass"
        case "write_file", "edit_file", "create_file": return "square.and.pencil"
        case "delete_file", "move_file": return "folder.badge.gearshape"
        case "run_javascript": return "play.rectangle"
        case "roadmap", "roadmap_add", "roadmap_update", "roadmap_list": return "point.3.connected.trianglepath.dotted"
        default: return "wrench.and.screwdriver"
        }
    }
}

/// 参考にした資料の一覧（タップで開く）。
private struct SourcesCaption: View {
    let sources: [String]
    @State private var expanded = false

    var body: some View {
        Button {
            expanded.toggle()
        } label: {
            VStack(alignment: .trailing, spacing: 2) {
                Label(L("参考資料 \(sources.count) 件", "\(sources.count) references"), systemImage: "paperclip")
                if expanded {
                    ForEach(sources, id: \.self) { source in
                        Text(source).font(.caption2.monospaced())
                    }
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
    }
}

extension StopReason {
    var label: String {
        switch self {
        case .endOfTurn: return L("完了", "Done")
        case .maxTokens: return L("最大長で停止", "Hit max tokens")
        case .contextFull: return L("コンテキスト上限", "Context full")
        case .cancelled: return L("停止しました", "Stopped")
        case .error: return L("エラー", "Error")
        case .lowMemory: return L("メモリ不足で停止", "Stopped: low memory")
        }
    }
}

/// 生成待ちの「…」アニメーション。
private struct TypingIndicator: View {
    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.35)) { timeline in
            let step = Int(timeline.date.timeIntervalSinceReferenceDate / 0.35) % 3
            HStack(spacing: 5) {
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .frame(width: 7, height: 7)
                        .opacity(index == step ? 1 : 0.3)
                }
            }
            .foregroundStyle(.secondary)
            .animation(.easeInOut(duration: 0.2), value: step)
        }
    }
}
