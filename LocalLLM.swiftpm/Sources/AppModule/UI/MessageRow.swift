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
                let sources = message.sources
                if !sources.isEmpty {
                    SourcesCaption(sources: sources)
                }
            }
        case .assistant, .system:
            AssistantMessageView(message: message, isStreaming: isStreaming)
                .contextMenu { menuItems }
                .overlay(alignment: .bottomTrailing) {
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
                } else if !isStreaming && message.errorText == nil && split.thinking == nil {
                    Text(L("（空の返答）", "(empty reply)"))
                        .foregroundStyle(.secondary)
                }
            } else {
                MarkdownText(split.answer)
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
