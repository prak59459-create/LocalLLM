import SwiftUI

/// コーディングエージェントとの対話の欄。
/// ツールの実行は 1 回の返答ごとに 1 行の「作業の帯」にまとめ、細かい結果はタップしたときだけ見せる。
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
            AgentMessageList(session: session, onPick: { draft = $0; focused = true })
            if let approval = session.pendingApproval {
                ApprovalCard(approval: approval, session: session)
                    .padding(.horizontal, 10)
                    .padding(.bottom, 6)
            }
            if let status = session.agentStatus {
                AgentStatusPill(text: status)
                    .padding(.bottom, 6)
            }
            AgentComposer(draft: $draft, focused: $focused, isRunning: session.isAgentRunning,
                          onSend: send, onStop: { session.stopAgent() })
        }
        .background(CDWSTheme.panel)
    }

    private func send() {
        let text = draft
        draft = ""
        session.sendToAgent(text, chat: chat, settings: settings, connectors: connectors)
    }
}

// MARK: - 見出し

private struct AgentHeader: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        HStack(spacing: 10) {
            CDWSIconTile(symbol: "sparkles", size: 30)
                .cdwsGlow(session.isAgentRunning, radius: 9)
            VStack(alignment: .leading, spacing: 1) {
                Text(L("エージェント", "Agent"))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(CDWSTheme.text)
                AgentModelMenu()
            }
            Spacer()
            AgentSettingsMenu(session: session)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(CDWSTheme.panel)
        .overlay(alignment: .bottom) { Rectangle().fill(CDWSTheme.stroke).frame(height: 1) }
    }
}

/// エージェントに使うモデルの切り替え（ワークスペースから出なくても読み込める）。
private struct AgentModelMenu: View {
    @Environment(ChatController.self) private var chat
    @Environment(ModelLibrary.self) private var library
    @Environment(AppSettings.self) private var settings

    private var title: String {
        switch chat.loadState {
        case .ready(let info): return info.displayName
        case .loading(let name), .compiling(let name): return L("読み込み中: ", "Loading: ") + name
        case .failed: return L("読み込みに失敗しました", "Load failed")
        case .idle: return L("モデルを選ぶ", "Choose a model")
        }
    }

    var body: some View {
        Menu {
            if library.completedModels.isEmpty {
                Text(L("モデルがありません（チャット画面のモデルから入手してください）", "No models (get one from Models in the chat screen)"))
            }
            ForEach(library.completedModels) { model in
                Button {
                    chat.loadModel(id: model.id, library: library, settings: settings)
                } label: {
                    if chat.loadedInfo?.modelID == model.id {
                        Label(model.displayName, systemImage: "checkmark")
                    } else {
                        Text(model.displayName)
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                if chat.isBusyLoading { ProgressView().controlSize(.mini) }
                Text(title).lineLimit(1)
                Image(systemName: "chevron.down").font(.caption2)
            }
            .font(.caption)
            .foregroundStyle(chat.loadedInfo == nil ? CDWSTheme.orange : CDWSTheme.secondaryText)
        }
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
                .foregroundStyle(CDWSTheme.secondaryText)
                .frame(width: 32, height: 32)
                .background(CDWSTheme.raised, in: Circle())
        }
    }
}

// MARK: - やり取り

/// やり取りの一覧（新しいものが来たら下へ送る）。
private struct AgentMessageList: View {
    @Bindable var session: WorkspaceSession
    var onPick: (String) -> Void

    private var lastLength: Int { session.messages.last?.content.count ?? 0 }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                AgentMessageStack(session: session, onPick: onPick)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: lastLength) { _, _ in proxy.scrollTo("agent-bottom", anchor: .bottom) }
            .onChange(of: session.messages.count) { _, _ in
                withAnimation { proxy.scrollTo("agent-bottom", anchor: .bottom) }
            }
        }
    }
}

private struct AgentMessageStack: View {
    @Bindable var session: WorkspaceSession
    var onPick: (String) -> Void

    var body: some View {
        let results = AgentToolIndex(messages: session.messages)
        LazyVStack(alignment: .leading, spacing: 14) {
            if session.messages.isEmpty {
                AgentSuggestions(onPick: onPick)
            }
            ForEach(session.messages) { message in
                if !results.isShownInStrip(message) {
                    AgentMessageRow(message: message, session: session, results: results,
                                    isStreaming: session.streamingMessageID == message.id)
                        .id(message.id)
                }
            }
            Color.clear.frame(height: 1).id("agent-bottom")
        }
        .padding(12)
    }
}

/// ツールの呼び出し ID → 結果（返答の帯にまとめて見せる）
struct AgentToolIndex {
    private var results: [String: AgentMessage] = [:]
    private var callIDs = Set<String>()

    init(messages: [AgentMessage]) {
        for message in messages {
            if message.role == .tool, let id = message.toolCallID { results[id] = message }
            for call in message.toolCalls { callIDs.insert(call.id) }
        }
    }

    func result(for call: ToolCall) -> AgentMessage? { results[call.id] }

    /// 帯の中に出すので、一覧には行として出さない結果
    func isShownInStrip(_ message: AgentMessage) -> Bool {
        guard message.role == .tool, let id = message.toolCallID else { return false }
        return callIDs.contains(id)
    }
}

/// やり取り 1 件。
private struct AgentMessageRow: View {
    let message: AgentMessage
    @Bindable var session: WorkspaceSession
    let results: AgentToolIndex
    let isStreaming: Bool

    var body: some View {
        switch message.role {
        case .user:
            AgentUserBubble(text: message.content)
        case .assistant:
            AgentReplyView(message: message, session: session, results: results, isStreaming: isStreaming)
        case .tool:
            AgentOrphanResult(message: message)
        }
    }
}

private struct AgentUserBubble: View {
    let text: String

    var body: some View {
        HStack {
            Spacer(minLength: 36)
            Text(text)
                .font(.callout)
                .foregroundStyle(.white)
                .textSelection(.enabled)
                .padding(.horizontal, 13)
                .padding(.vertical, 9)
                .background(CDWSTheme.accent, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }
}

private struct AgentReplyView: View {
    let message: AgentMessage
    @Bindable var session: WorkspaceSession
    let results: AgentToolIndex
    let isStreaming: Bool

    var body: some View {
        let split = ThinkingSplit.split(message.content)
        VStack(alignment: .leading, spacing: 8) {
            if let thinking = split.thinking, !thinking.isEmpty {
                AgentThoughts(text: thinking)
            }
            AgentAnswer(text: split.answer, isStreaming: isStreaming)
            if !message.toolCalls.isEmpty {
                AgentActivityStrip(calls: message.toolCalls, results: results)
            }
            AgentReplyFooter(message: message, session: session, isStreaming: isStreaming)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct AgentThoughts: View {
    let text: String

    var body: some View {
        DisclosureGroup {
            Text(text)
                .font(.caption)
                .foregroundStyle(CDWSTheme.secondaryText)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label(L("思考の過程", "Thoughts"), systemImage: "brain")
                .font(.caption.weight(.medium))
                .foregroundStyle(CDWSTheme.secondaryText)
        }
        .tint(CDWSTheme.secondaryText)
    }
}

/// 返答の本文。書いている間は軽い Text で、書き終わったら Markdown で表示する。
private struct AgentAnswer: View {
    let text: String
    let isStreaming: Bool

    var body: some View {
        if text.isEmpty {
            if isStreaming {
                Text("…").foregroundStyle(CDWSTheme.secondaryText)
            }
        } else if isStreaming {
            Text(text)
                .font(.callout)
                .foregroundStyle(CDWSTheme.text)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            MarkdownText(text)
                .foregroundStyle(CDWSTheme.text)
        }
    }
}

private struct AgentReplyFooter: View {
    let message: AgentMessage
    @Bindable var session: WorkspaceSession
    let isStreaming: Bool

    var body: some View {
        if !isStreaming {
            HStack(spacing: 8) {
                if message.toolCalls.isEmpty, let batch = message.batchID, session.hasChanges(in: batch), isLastReply(in: batch) {
                    Button { session.undoBatch(batch) } label: {
                        Label(L("この依頼の変更を元に戻す", "Undo this request"), systemImage: "arrow.uturn.backward")
                            .font(.caption.weight(.medium))
                    }
                    .buttonStyle(.bordered)
                    .tint(CDWSTheme.orange)
                    .disabled(session.isAgentRunning)
                }
                if let stats = message.stats {
                    Text(statsText(stats))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(CDWSTheme.secondaryText.opacity(0.7))
                }
            }
        }
    }

    private func statsText(_ stats: GenerationStats) -> String {
        var text = String(format: "%.1f tok/s · %ld", stats.decodeTokensPerSecond, stats.generatedTokens)
        if stats.stopReason == .lowMemory { text += " · " + L("メモリ不足で停止", "stopped: low memory") }
        return text
    }

    private func isLastReply(in batch: UUID) -> Bool {
        session.messages.last(where: { $0.role == .assistant && $0.batchID == batch })?.id == message.id
    }
}

// MARK: - 作業の帯

/// 1 回の返答で呼んだツールを 1 行にまとめる（タップで結果）。
private struct AgentActivityStrip: View {
    let calls: [ToolCall]
    let results: AgentToolIndex

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(calls) { call in
                    AgentActivityChip(call: call, result: results.result(for: call))
                }
            }
        }
    }
}

private struct AgentActivityChip: View {
    let call: ToolCall
    let result: AgentMessage?
    @State private var showing = false

    private var style: AgentToolStyle { AgentToolStyle(call: call) }

    private var statusColor: Color {
        guard let result else { return CDWSTheme.cyan }
        return result.isError ? CDWSTheme.red : CDWSTheme.green
    }

    var body: some View {
        Button { showing = true } label: {
            HStack(spacing: 5) {
                Image(systemName: style.symbol).font(.caption2.weight(.bold)).foregroundStyle(style.color)
                Text(style.label).font(.caption.monospaced()).foregroundStyle(CDWSTheme.text).lineLimit(1)
                statusMark
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(CDWSTheme.raised, in: Capsule())
            .overlay(Capsule().strokeBorder(statusColor.opacity(0.35)))
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showing) {
            AgentToolDetail(call: call, result: result)
        }
    }

    @ViewBuilder
    private var statusMark: some View {
        if result == nil {
            ProgressView().controlSize(.mini)
        } else {
            Circle().fill(statusColor).frame(width: 6, height: 6)
        }
    }
}

/// ツールの呼び出しと結果（帯の札をタップしたとき）
private struct AgentToolDetail: View {
    let call: ToolCall
    let result: AgentMessage?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Text(call.summary)
                    .font(.caption.monospaced())
                    .foregroundStyle(CDWSTheme.secondaryText)
                Text(resultText)
                    .font(.caption.monospaced())
                    .foregroundStyle(result?.isError == true ? CDWSTheme.red : CDWSTheme.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(14)
        }
        .frame(minWidth: 280, idealWidth: 420, maxHeight: 420)
        .background(CDWSTheme.panel)
    }

    private var resultText: String {
        guard let text = result?.content else { return L("実行中…", "Running…") }
        return text.count > 3000 ? String(text.prefix(3000)) + "…" : text
    }
}

/// ツールごとのアイコン・色・短い名前
struct AgentToolStyle {
    var symbol: String
    var color: Color
    var label: String

    init(call: ToolCall) {
        let path = WorkspaceToolExecutor.path(call).map { ($0 as NSString).lastPathComponent }
        switch call.name {
        case WorkspaceToolbox.Name.readFile:
            self.init(symbol: "doc.text", color: CDWSTheme.cyan, label: path ?? "read")
        case WorkspaceToolbox.Name.writeFile:
            self.init(symbol: "square.and.pencil", color: CDWSTheme.green, label: path ?? "write")
        case WorkspaceToolbox.Name.editFile:
            self.init(symbol: "pencil.line", color: CDWSTheme.violet, label: path ?? "edit")
        case WorkspaceToolbox.Name.searchFiles, WorkspaceToolbox.Name.findFiles, WorkspaceToolbox.Name.listFiles:
            let pattern = call.argument("pattern") ?? call.argument("query") ?? ""
            self.init(symbol: "magnifyingglass", color: CDWSTheme.orange, label: pattern.isEmpty ? L("一覧", "list") : pattern)
        case WorkspaceToolbox.Name.roadmap, WorkspaceToolbox.Name.roadmapAdd, WorkspaceToolbox.Name.roadmapUpdate, WorkspaceToolbox.Name.roadmapList:
            let label = call.argument("title") ?? call.argument("id") ?? L("ロードマップ", "roadmap")
            self.init(symbol: "map", color: CDWSTheme.pink, label: label)
        case WorkspaceToolbox.Name.deleteFile:
            self.init(symbol: "trash", color: CDWSTheme.red, label: path ?? "delete")
        case WorkspaceToolbox.Name.moveFile:
            self.init(symbol: "arrow.right.doc.on.clipboard", color: CDWSTheme.orange, label: call.argument("to") ?? "move")
        case WorkspaceToolbox.Name.runJavaScript:
            self.init(symbol: "play.fill", color: CDWSTheme.green, label: path ?? "JavaScript")
        default:
            self.init(symbol: "wrench.and.screwdriver", color: CDWSTheme.secondaryText, label: call.name)
        }
        if label.count > 28 { label = String(label.prefix(28)) + "…" }
    }

    init(symbol: String, color: Color, label: String) {
        self.symbol = symbol
        self.color = color
        self.label = label
    }
}

/// 前の版で保存された、呼び出しの無いツールの結果（1 行だけ）
private struct AgentOrphanResult: View {
    let message: AgentMessage

    var body: some View {
        Label(title, systemImage: message.isError ? "exclamationmark.triangle" : "checkmark.circle")
            .font(.caption)
            .foregroundStyle(message.isError ? CDWSTheme.red : CDWSTheme.secondaryText)
            .lineLimit(1)
    }

    private var title: String {
        let first = message.content.split(separator: "\n").first.map(String.init) ?? ""
        return "\(message.toolName ?? "tool"): " + (first.count > 60 ? String(first.prefix(60)) + "…" : first)
    }
}

// MARK: - 下の部分

private struct AgentStatusPill: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small).tint(CDWSTheme.cyan)
            Text(text)
                .font(.caption.weight(.medium))
                .foregroundStyle(CDWSTheme.text)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(CDWSTheme.raised, in: Capsule())
        .cdwsGlow(true, radius: 20)
    }
}

private struct AgentComposer: View {
    @Binding var draft: String
    var focused: FocusState<Bool>.Binding
    let isRunning: Bool
    var onSend: () -> Void
    var onStop: () -> Void

    private var canSend: Bool { !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField(L("作りたいもの・直したいところ", "What should I build or fix?"), text: $draft, axis: .vertical)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .foregroundStyle(CDWSTheme.text)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(CDWSTheme.raised, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(focused.wrappedValue ? CDWSTheme.violet.opacity(0.7) : CDWSTheme.stroke))
                .focused(focused)
            actionButton
        }
        .padding(10)
        .background(CDWSTheme.panel)
        .overlay(alignment: .top) { Rectangle().fill(CDWSTheme.stroke).frame(height: 1) }
    }

    @ViewBuilder
    private var actionButton: some View {
        if isRunning {
            Button(action: onStop) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .background(CDWSTheme.red, in: Circle())
            }
            .accessibilityLabel(L("停止", "Stop"))
        } else {
            Button(action: onSend) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .background(CDWSTheme.accent, in: Circle())
                    .opacity(canSend ? 1 : 0.4)
            }
            .disabled(!canSend)
            .keyboardShortcut(.return, modifiers: .command)
            .accessibilityLabel(L("送信", "Send"))
        }
    }
}

/// 最初に出す、依頼の例。
private struct AgentSuggestions: View {
    var onPick: (String) -> Void

    private var examples: [(String, String)] {
        [
            ("map", L("このプロジェクトの構成を説明して", "Explain how this project is organized")),
            ("moon.stars", L("ダークモードに対応して", "Add dark mode support")),
            ("ladybug", L("バグがないか調べて直して", "Look for bugs and fix them")),
            ("doc.richtext", L("README を書いて", "Write a README")),
            ("point.3.connected.trianglepath.dotted", L("次に作るとよい機能をロードマップに足して", "Add good next features to the roadmap")),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L("エージェントに頼めること", "Things to ask the agent"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(CDWSTheme.secondaryText)
            ForEach(examples, id: \.1) { example in
                AgentSuggestionButton(symbol: example.0, text: example.1, onPick: onPick)
            }
            Text(L("エージェントはファイルを少しずつ読んで直すので、小さなモデルでも動きます。おすすめの「コーディング専用」のモデル（Qwen2.5 Coder など）が向いています。",
                   "The agent reads and edits files a little at a time, so small models work too. The recommended coding models (Qwen2.5 Coder, …) fit best."))
                .font(.caption2)
                .foregroundStyle(CDWSTheme.secondaryText)
                .padding(.top, 4)
        }
    }
}

private struct AgentSuggestionButton: View {
    let symbol: String
    let text: String
    var onPick: (String) -> Void

    var body: some View {
        Button { onPick(text) } label: {
            HStack(spacing: 10) {
                Image(systemName: symbol)
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(CDWSTheme.cyan)
                    .frame(width: 22)
                Text(text).font(.callout).foregroundStyle(CDWSTheme.text)
                Spacer(minLength: 0)
                Image(systemName: "arrow.up.left").font(.caption2).foregroundStyle(CDWSTheme.secondaryText)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .cdwsPanel(radius: 12, fill: CDWSTheme.raised)
        }
        .buttonStyle(.plain)
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
                .foregroundStyle(CDWSTheme.orange)
            Text(approval.call.summary)
                .font(.caption.monospaced())
                .foregroundStyle(CDWSTheme.text)
                .lineLimit(3)
            if let proposal = approval.proposal {
                ApprovalProposal(proposal: proposal)
            }
            ApprovalButtons(session: session)
        }
        .padding(12)
        .cdwsPanel(radius: 14, fill: CDWSTheme.raised)
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(CDWSTheme.orange.opacity(0.45)))
    }
}

private struct ApprovalProposal: View {
    let proposal: ProposedChange

    var body: some View {
        Text("\(proposal.path) — \(proposal.description)")
            .font(.caption)
            .foregroundStyle(CDWSTheme.secondaryText)
        if proposal.before != nil || proposal.after != nil {
            ScrollView {
                DiffView(before: proposal.before ?? "", after: proposal.after ?? "", maxLines: 300)
            }
            .frame(maxHeight: 220)
            .background(CDWSTheme.canvas, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

private struct ApprovalButtons: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        HStack {
            Button(L("許可しない", "Deny"), role: .destructive) { session.resolveApproval(false) }
            Spacer()
            Button(L("以後は自動で", "Always")) { session.resolveApproval(true, always: true) }
            Button(L("許可", "Allow")) { session.resolveApproval(true) }
                .buttonStyle(.borderedProminent)
                .tint(CDWSTheme.violet)
                .keyboardShortcut(.return, modifiers: [.command, .shift])
        }
        .buttonStyle(.bordered)
        .font(.callout)
    }
}
