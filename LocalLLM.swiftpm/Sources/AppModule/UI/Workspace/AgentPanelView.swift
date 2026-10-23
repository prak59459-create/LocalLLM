import SwiftUI

/// コーディングエージェントとの対話の欄。
struct AgentPanelView: View {
    @Bindable var session: WorkspaceSession
    @Environment(ChatController.self) private var chat
    @Environment(AppSettings.self) private var settings
    @Environment(ConnectorStore.self) private var connectors

    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            AgentHeader(session: session)
            Divider()
            messageList
            if let approval = session.pendingApproval {
                Divider()
                ApprovalCard(approval: approval, session: session)
            }
            if let status = session.agentStatus {
                AgentStatusLine(text: status)
            }
            Divider()
            composer
        }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if session.messages.isEmpty {
                        AgentSuggestions(onPick: { draft = $0; focused = true })
                    }
                    ForEach(session.messages) { message in
                        AgentMessageRow(message: message, session: session,
                                        isStreaming: session.streamingMessageID == message.id)
                            .id(message.id)
                    }
                    Color.clear.frame(height: 1).id("agent-bottom")
                }
                .padding(12)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: session.messages.last?.content.count) { _, _ in
                proxy.scrollTo("agent-bottom", anchor: .bottom)
            }
            .onChange(of: session.messages.count) { _, _ in
                withAnimation { proxy.scrollTo("agent-bottom", anchor: .bottom) }
            }
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField(L("作りたいもの・直したいところ", "What should I build or fix?"), text: $draft, axis: .vertical)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
                .focused($focused)
            if session.isAgentRunning {
                Button { session.stopAgent() } label: {
                    Image(systemName: "stop.circle.fill").font(.system(size: 30))
                }
                .accessibilityLabel(L("停止", "Stop"))
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill").font(.system(size: 30))
                }
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
                .accessibilityLabel(L("送信", "Send"))
            }
        }
        .padding(10)
    }

    private func send() {
        let text = draft
        draft = ""
        session.sendToAgent(text, chat: chat, settings: settings, connectors: connectors)
    }
}

private struct AgentHeader: View {
    @Bindable var session: WorkspaceSession
    @Environment(ChatController.self) private var chat

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles").foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(L("エージェント", "Agent")).font(.subheadline.weight(.semibold))
                Text(chat.loadedInfo?.displayName ?? L("モデル未読み込み", "No model loaded"))
                    .font(.caption)
                    .foregroundStyle(chat.loadedInfo == nil ? Color.orange : Color.secondary)
                    .lineLimit(1)
            }
            Spacer()
            AgentSettingsMenu(session: session)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }
}

private struct AgentSettingsMenu: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        Menu {
            Picker(L("ファイルを変える前の確認", "Before changing files"), selection: $session.approvalMode) {
                ForEach(AgentApprovalMode.allCases) { Text($0.label).tag($0) }
            }
            Picker(L("渡すツール", "Tools"), selection: $session.toolProfile) {
                ForEach(WorkspaceToolbox.Profile.allCases) { Text($0.label).tag($0) }
            }
            Toggle(L("Web 検索も使う", "Also use web search"), isOn: $session.useWebTools)
            Divider()
            Button(role: .destructive) { session.clearConversation() } label: {
                Label(L("やり取りを消す", "Clear conversation"), systemImage: "trash")
            }
            .disabled(session.isAgentRunning)
        } label: {
            Image(systemName: "slider.horizontal.3")
        }
    }
}

private struct AgentStatusLine: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

/// 最初に出す、依頼の例。
private struct AgentSuggestions: View {
    var onPick: (String) -> Void

    private var examples: [String] {
        [
            L("このプロジェクトの構成を説明して", "Explain how this project is organized"),
            L("ダークモードに対応して", "Add dark mode support"),
            L("バグがないか調べて直して", "Look for bugs and fix them"),
            L("README を書いて", "Write a README"),
            L("次に作るとよい機能をロードマップに足して", "Add good next features to the roadmap"),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L("エージェントに頼めること", "Things to ask the agent"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(examples, id: \.self) { example in
                Button { onPick(example) } label: {
                    Text(example)
                        .font(.callout)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(Color.accentColor.opacity(0.1), in: Capsule())
                }
                .buttonStyle(.plain)
            }
            Text(L("小さなモデルでも動くように、エージェントはファイルを少しずつ読んで直します。ツールの呼び出しに対応したモデル（Qwen2.5-Coder / Qwen3 など）がおすすめです。",
                   "The agent reads and edits files a little at a time so it works with small models. Models trained for tool calling (Qwen2.5-Coder, Qwen3, …) work best."))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.top, 6)
        }
    }
}

/// やり取り 1 件。
private struct AgentMessageRow: View {
    let message: AgentMessage
    @Bindable var session: WorkspaceSession
    let isStreaming: Bool

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer(minLength: 30)
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.accentColor.opacity(0.16), in: RoundedRectangle(cornerRadius: 14))
            }
        case .assistant:
            AgentReplyView(message: message, session: session, isStreaming: isStreaming)
        case .tool:
            AgentToolResultView(message: message)
        }
    }
}

private struct AgentReplyView: View {
    let message: AgentMessage
    @Bindable var session: WorkspaceSession
    let isStreaming: Bool

    var body: some View {
        let split = ThinkingSplit.split(message.content)
        VStack(alignment: .leading, spacing: 6) {
            if let thinking = split.thinking, !thinking.isEmpty {
                DisclosureGroup(L("思考の過程", "Thoughts")) {
                    Text(thinking).font(.caption).foregroundStyle(.secondary)
                }
                .font(.caption)
            }
            if !split.answer.isEmpty {
                MarkdownText(split.answer)
            } else if isStreaming {
                Text("…").foregroundStyle(.secondary)
            }
            if !message.toolCalls.isEmpty {
                ToolCallChips(calls: message.toolCalls)
            }
            footer
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var footer: some View {
        if !isStreaming && message.toolCalls.isEmpty, let batch = message.batchID, session.hasChanges(in: batch),
           isLastReply(in: batch) {
            Button { session.undoBatch(batch) } label: {
                Label(L("この依頼の変更を元に戻す", "Undo this request's changes"), systemImage: "arrow.uturn.backward")
                    .font(.caption)
            }
            .buttonStyle(.bordered)
            .disabled(session.isAgentRunning)
        }
        if !isStreaming, let stats = message.stats {
            Text(String(format: "%.1f tok/s · %ld", stats.decodeTokensPerSecond, stats.generatedTokens))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
    }

    private func isLastReply(in batch: UUID) -> Bool {
        session.messages.last(where: { $0.role == .assistant && $0.batchID == batch })?.id == message.id
    }
}

private struct AgentToolResultView: View {
    let message: AgentMessage
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(message.content)
                .font(.caption2.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            Label(title, systemImage: message.isError ? "exclamationmark.triangle" : ToolResultView.icon(for: message.toolName ?? ""))
                .font(.caption)
                .foregroundStyle(message.isError ? Color.orange : Color.secondary)
                .lineLimit(1)
        }
        .padding(8)
        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
    }

    private var title: String {
        let first = message.content.split(separator: "\n").first.map(String.init) ?? ""
        return "\(message.toolName ?? "tool"): " + (first.count > 60 ? String(first.prefix(60)) + "…" : first)
    }
}

/// ファイルを変える前の確認カード（差分つき）。
private struct ApprovalCard: View {
    let approval: PendingApproval
    @Bindable var session: WorkspaceSession

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(L("この操作を許可しますか？", "Allow this action?"), systemImage: "hand.raised.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.orange)
            Text(approval.call.summary)
                .font(.caption.monospaced())
                .lineLimit(3)
            if let proposal = approval.proposal {
                Text("\(proposal.path) — \(proposal.description)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if proposal.before != nil || proposal.after != nil {
                    ScrollView {
                        DiffView(before: proposal.before ?? "", after: proposal.after ?? "", maxLines: 300)
                    }
                    .frame(maxHeight: 220)
                    .background(Color(uiColor: .systemBackground), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            HStack {
                Button(L("許可しない", "Deny"), role: .destructive) { session.resolveApproval(false) }
                Spacer()
                Button(L("以後は自動で", "Always")) { session.resolveApproval(true, always: true) }
                Button(L("許可", "Allow")) { session.resolveApproval(true) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command, .shift])
            }
            .buttonStyle(.bordered)
            .font(.callout)
        }
        .padding(12)
        .background(Color.orange.opacity(0.08))
    }
}
