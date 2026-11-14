import SwiftUI
import WebKit

/// Web アプリのプレビューと、JavaScript のコンソール。
struct PreviewPane: View {
    @Bindable var session: WorkspaceSession
    @State private var showConsole = true

    var body: some View {
        VStack(spacing: 0) {
            PreviewToolbar(session: session, showConsole: $showConsole)
            Divider()
            if let entry = session.previewEntry {
                WebPreview(session: session, entry: entry, revision: session.previewRevision)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "play.rectangle")
                        .font(.system(size: 40))
                        .foregroundStyle(.secondary)
                    Text(L("HTML のファイルがありません。JavaScript はファイルの「実行」で動かせます。",
                           "No HTML file. Run JavaScript files with \"Run\"."))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(40)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if showConsole {
                Divider()
                ConsoleView(session: session)
                    .frame(height: 180)
            }
        }
    }
}

private struct PreviewToolbar: View {
    @Bindable var session: WorkspaceSession
    @Binding var showConsole: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "globe").foregroundStyle(.secondary)
            Text(session.previewEntry ?? L("プレビューなし", "No preview"))
                .font(.callout.monospaced())
                .lineLimit(1)
            Spacer()
            Button {
                session.saveAll()
                session.reloadPreview()
            } label: {
                Label(L("再読み込み", "Reload"), systemImage: "arrow.clockwise")
            }
            .keyboardShortcut("r", modifiers: [.command, .shift])
            Toggle(isOn: $showConsole) {
                Label(L("コンソール", "Console"), systemImage: "terminal")
            }
            .toggleStyle(.button)
        }
        .labelStyle(.iconOnly)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

private struct ConsoleView: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(L("コンソール", "Console")).font(.caption.weight(.semibold))
                Spacer()
                Button(L("消去", "Clear")) { session.clearConsole() }
                    .font(.caption)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            ConsoleLines(lines: session.consoleLines)
        }
        .background(CDWSTheme.panel)
    }
}

private struct ConsoleLines: View {
    let lines: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(lines.indices, id: \.self) { index in
                        ConsoleLine(text: lines[index])
                            .id(index)
                    }
                }
                .padding(.horizontal, 12)
            }
            .onChange(of: lines.count) { _, count in
                proxy.scrollTo(count - 1, anchor: .bottom)
            }
        }
    }
}

private struct ConsoleLine: View {
    let text: String

    private var color: Color {
        if text.hasPrefix("[error]") { return .red }
        if text.hasPrefix("[warn]") { return .orange }
        return .primary
    }

    var body: some View {
        Text(text)
            .font(.caption.monospaced())
            .foregroundStyle(color)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// プロジェクトの HTML を表示する WKWebView。console の出力はコンソールに送る。
private struct WebPreview: UIViewRepresentable {
    let session: WorkspaceSession
    let entry: String
    let revision: Int

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        let controller = WKUserContentController()
        controller.addUserScript(WKUserScript(source: JavaScriptRunner.consoleBridge, injectionTime: .atDocumentStart, forMainFrameOnly: false))
        let session = session
        controller.add(WeakMessageHandler { message in session.appendConsole(JavaScriptRunner.format(message)) }, name: "console")
        configuration.userContentController = controller
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.isInspectable = true
        view.navigationDelegate = context.coordinator
        context.coordinator.onTerminated = { message in session.appendConsole(message) }
        load(into: view, context: context)
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {
        if context.coordinator.loadedRevision != revision || context.coordinator.loadedEntry != entry {
            load(into: view, context: context)
        }
    }

    private func load(into view: WKWebView, context: Context) {
        context.coordinator.loadedRevision = revision
        context.coordinator.loadedEntry = entry
        guard let folder = session.preparePreviewFolder() else { return }
        let file = folder.appendingPathComponent(entry)
        view.loadFileURL(file, allowingReadAccessTo: folder)
    }

    static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) {
        view.configuration.userContentController.removeAllScriptMessageHandlers()
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var loadedRevision = -1
        var loadedEntry = ""
        var onTerminated: ((String) -> Void)?
        private var recentReloads: [Date] = []

        /// メモリが足りなくなると、iOS はプレビューを動かしているプロセスだけを終了させることがある。
        /// そのままだと真っ白で何も反応しないので、読み込み直す（続けて落ちるときは、やり直しを繰り返さない）。
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            let now = Date()
            recentReloads = recentReloads.filter { now.timeIntervalSince($0) < 30 }
            // 次に画面が更新されたとき、必ず読み込み直す
            loadedRevision = -1
            guard recentReloads.count < 2 else {
                onTerminated?(L("[preview] メモリ不足でプレビューが止まりました。「再読み込み」を押すと、もう一度表示します。",
                                "[preview] The preview stopped because memory ran low. Press Reload to show it again."))
                return
            }
            recentReloads.append(now)
            onTerminated?(L("[preview] メモリ不足でプレビューが止まったので、読み込み直しました。",
                            "[preview] The preview was stopped because memory ran low; it has been reloaded."))
            webView.reload()
        }
    }
}
