import SwiftUI

/// 開いたプロジェクトの画面: 左にファイル、真ん中にエディタなど、右にエージェント、下に状態の帯。
struct WorkspaceView: View {
    @Bindable var session: WorkspaceSession
    var onClose: () -> Void

    @Environment(\.horizontalSizeClass) private var sizeClass
    @State private var showExplorer = true
    @State private var showAgent = true
    @State private var compactSheet: CompactSheet?

    private enum CompactSheet: String, Identifiable {
        case files, agent
        var id: String { rawValue }
    }

    private var isCompact: Bool { sizeClass == .compact }

    var body: some View {
        VStack(spacing: 0) {
            if let notice = session.notice {
                NoticeBar(text: notice) { session.notice = nil }
            }
            WorkspaceColumns(session: session, showExplorer: showExplorer && !isCompact, showAgent: showAgent && !isCompact)
            WorkspaceStatusBar(session: session)
        }
        .background(CDWSTheme.canvas)
        .navigationTitle(session.project.name)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbarBackground(CDWSTheme.panel, for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        .toolbar { toolbar }
        .onChange(of: session.agentPanelRequest) { _, _ in
            if isCompact { compactSheet = .agent } else { withAnimation { showAgent = true } }
        }
        .sheet(item: $compactSheet) { sheet in
            WorkspaceCompactSheet(session: session, showsAgent: sheet == .agent, onClose: { compactSheet = nil })
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Button(action: onClose) {
                Label(L("プロジェクト", "Projects"), systemImage: "chevron.backward")
            }
        }
        ToolbarItem(placement: .topBarLeading) {
            Button {
                if isCompact { compactSheet = .files } else { withAnimation(.snappy) { showExplorer.toggle() } }
            } label: {
                Label(L("ファイル", "Files"), systemImage: "sidebar.left")
            }
        }
        ToolbarItem(placement: .principal) {
            CDWSPaneSwitcher(selection: $session.pane)
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button {
                if isCompact { compactSheet = .agent } else { withAnimation(.snappy) { showAgent.toggle() } }
            } label: {
                Label(L("エージェント", "Agent"), systemImage: session.isAgentRunning ? "sparkles" : "sidebar.right")
                    .symbolEffect(.pulse, isActive: session.isAgentRunning)
            }
        }
    }
}

/// 左・真ん中・右の 3 つの欄
private struct WorkspaceColumns: View {
    @Bindable var session: WorkspaceSession
    let showExplorer: Bool
    let showAgent: Bool

    var body: some View {
        HStack(spacing: 8) {
            if showExplorer {
                FileExplorerView(session: session)
                    .frame(width: 250)
                    .cdwsPanel()
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }
            WorkspaceCenterPane(session: session)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .cdwsPanel()
            if showAgent {
                AgentPanelView(session: session)
                    .frame(width: 380)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .cdwsPanel()
                    .cdwsGlow(session.isAgentRunning)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .padding(8)
    }
}

/// iPhone など狭い画面: ファイルとエージェントはシートで開く
private struct WorkspaceCompactSheet: View {
    @Bindable var session: WorkspaceSession
    let showsAgent: Bool
    var onClose: () -> Void

    var body: some View {
        NavigationStack {
            Group {
                if showsAgent {
                    AgentPanelView(session: session)
                } else {
                    FileExplorerView(session: session)
                }
            }
            .background(CDWSTheme.panel)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close"), action: onClose)
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

/// 真ん中の表示（エディタ・プレビュー・ロードマップ・変更履歴）。
private struct WorkspaceCenterPane: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        switch session.pane {
        case .editor:
            EditorPane(session: session)
        case .preview:
            PreviewPane(session: session)
        case .roadmap:
            RoadmapBoardView(session: session)
        case .changes:
            ChangesView(session: session)
        }
    }
}

/// 下の帯: モデル・コンテキストの使い方・メモリの残り・エージェントの状態
private struct WorkspaceStatusBar: View {
    @Bindable var session: WorkspaceSession
    @Environment(ChatController.self) private var chat

    var body: some View {
        HStack(spacing: 16) {
            WorkspaceAgentState(session: session)
            if let info = chat.loadedInfo {
                Label(info.displayName, systemImage: "cpu")
                    .lineLimit(1)
                    .foregroundStyle(CDWSTheme.secondaryText)
                if session.lastPromptTokens > 0 {
                    CDWSMeter(label: L("文脈", "Context"),
                              fraction: Double(session.lastPromptTokens) / Double(max(info.maxContext, 1)),
                              detail: "\(session.lastPromptTokens)/\(info.maxContext)")
                }
            }
            Spacer(minLength: 8)
            WorkspaceMemoryMeter()
            Text(session.pane.label).foregroundStyle(CDWSTheme.secondaryText)
        }
        .font(.caption2)
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(CDWSTheme.panel)
        .overlay(alignment: .top) { Rectangle().fill(CDWSTheme.stroke).frame(height: 1) }
    }
}

private struct WorkspaceAgentState: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(session.isAgentRunning ? CDWSTheme.cyan : CDWSTheme.green)
                .frame(width: 7, height: 7)
                .shadow(color: session.isAgentRunning ? CDWSTheme.cyan : .clear, radius: 4)
            Text(session.isAgentRunning ? (session.agentStatus ?? L("作業中", "Working")) : L("待機中", "Idle"))
                .lineLimit(1)
                .foregroundStyle(CDWSTheme.text)
        }
    }
}

private struct WorkspaceMemoryMeter: View {
    @Environment(ChatController.self) private var chat

    var body: some View {
        let used = Double(chat.memoryFootprint ?? 0)
        let total = used + Double(chat.memoryAvailable)
        if total > 0 {
            CDWSMeter(label: L("メモリ", "Memory"), fraction: used / total,
                      detail: L("残り ", "free ") + ByteFormat.string(chat.memoryAvailable))
        }
    }
}
