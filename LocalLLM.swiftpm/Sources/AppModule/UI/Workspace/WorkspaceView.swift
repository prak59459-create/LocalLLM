import SwiftUI

/// 開いたプロジェクトの画面: 左にファイル、真ん中にエディタなど、右にエージェント。
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
            HStack(spacing: 0) {
                if showExplorer && !isCompact {
                    FileExplorerView(session: session)
                        .frame(width: 250)
                    Divider()
                }
                WorkspaceCenterPane(session: session)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                if showAgent && !isCompact {
                    Divider()
                    AgentPanelView(session: session)
                        .frame(width: 380)
                }
            }
        }
        .navigationTitle(session.project.name)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar { toolbar }
        .onChange(of: session.agentPanelRequest) { _, _ in
            if isCompact { compactSheet = .agent } else { withAnimation { showAgent = true } }
        }
        .sheet(item: $compactSheet) { sheet in
            NavigationStack {
                Group {
                    switch sheet {
                    case .files: FileExplorerView(session: session)
                    case .agent: AgentPanelView(session: session)
                    }
                }
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button(L("閉じる", "Close")) { compactSheet = nil }
                    }
                }
            }
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
                if isCompact { compactSheet = .files } else { withAnimation { showExplorer.toggle() } }
            } label: {
                Label(L("ファイル", "Files"), systemImage: "sidebar.left")
            }
        }
        ToolbarItem(placement: .principal) {
            Picker(L("表示", "View"), selection: $session.pane) {
                ForEach(WorkspacePane.allCases) { pane in
                    Label(pane.label, systemImage: pane.symbolName).tag(pane)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 460)
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button {
                if isCompact { compactSheet = .agent } else { withAnimation { showAgent.toggle() } }
            } label: {
                Label(L("エージェント", "Agent"), systemImage: session.isAgentRunning ? "sparkles" : "sidebar.right")
                    .symbolEffect(.pulse, isActive: session.isAgentRunning)
            }
        }
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
