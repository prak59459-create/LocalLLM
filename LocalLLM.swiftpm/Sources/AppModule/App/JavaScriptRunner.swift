import Foundation
import WebKit

/// JavaScript を見えない WKWebView で実行して、console の出力と最後の値を返す。
/// Web ページは別のプロセスで動くので、無限ループになってもアプリは止まらない（時間切れで捨てる）。
@MainActor
final class JavaScriptRunner: NSObject, WKNavigationDelegate {
    struct Result: Sendable {
        var output: String
        var isError: Bool
    }

    private var webView: WKWebView?
    private var lines: [String] = []
    private var loadContinuation: CheckedContinuation<Void, Never>?

    /// console.log などを Swift に送るスクリプト
    static let consoleBridge = """
    (function() {
      function send(level, args) {
        try {
          var text = Array.prototype.map.call(args, function(a) {
            if (typeof a === 'string') return a;
            try { return JSON.stringify(a); } catch (e) { return String(a); }
          }).join(' ');
          window.webkit.messageHandlers.console.postMessage(level + '\\t' + text);
        } catch (e) {}
      }
      ['log', 'info', 'warn', 'error', 'debug'].forEach(function(level) {
        var original = console[level];
        console[level] = function() { send(level, arguments); if (original) original.apply(console, arguments); };
      });
      window.addEventListener('error', function(e) { send('error', [e.message + ' (' + (e.filename || '') + ':' + (e.lineno || 0) + ')']); });
      window.addEventListener('unhandledrejection', function(e) { send('error', ['Unhandled promise rejection: ' + e.reason]); });
    })();
    """

    func run(_ code: String, timeout: Double = 8) async -> Result {
        lines = []
        let configuration = WKWebViewConfiguration()
        let controller = WKUserContentController()
        controller.addUserScript(WKUserScript(source: Self.consoleBridge, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        controller.add(WeakMessageHandler { [weak self] message in self?.lines.append(Self.format(message)) }, name: "console")
        configuration.userContentController = controller
        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 320, height: 480), configuration: configuration)
        webView.navigationDelegate = self
        self.webView = webView
        defer {
            webView.stopLoading()
            controller.removeAllScriptMessageHandlers()
            self.webView = nil
        }

        await withCheckedContinuation { continuation in
            loadContinuation = continuation
            webView.loadHTMLString("<!doctype html><html><head><meta charset='utf-8'></head><body></body></html>", baseURL: nil)
        }

        // eval で最後の式の値も返す。例外はメッセージにして返す。
        let literal = (try? String(data: JSONEncoder().encode(code), encoding: .utf8)) ?? "\"\""
        let wrapped = """
        (function() {
          try {
            var __value = (0, eval)(\(literal));
            if (__value === undefined) return '__undefined__';
            if (typeof __value === 'string') return __value;
            try { return JSON.stringify(__value); } catch (e) { return String(__value); }
          } catch (e) {
            return '__error__' + (e && e.stack ? e.stack : String(e));
          }
        })()
        """
        let value = await evaluate(wrapped, in: webView, timeout: timeout)
        // setTimeout などで少し後に出る console.log を待つ
        try? await Task.sleep(nanoseconds: 150_000_000)

        var output = lines.joined(separator: "\n")
        var isError = false
        switch value {
        case .none:
            output += (output.isEmpty ? "" : "\n") + L("時間切れ（\(Int(timeout)) 秒）で止めました。無限ループになっていないか確かめてください。",
                                                       "Stopped after \(Int(timeout)) seconds. Check for infinite loops.")
            isError = true
        case .some(let text) where text.hasPrefix("__error__"):
            output += (output.isEmpty ? "" : "\n") + String(text.dropFirst("__error__".count))
            isError = true
        case .some(let text) where text != "__undefined__":
            output += (output.isEmpty ? "" : "\n") + "→ " + text
        default:
            break
        }
        if lines.contains(where: { $0.hasPrefix("[error]") }) { isError = true }
        if output.isEmpty { output = L("（出力なし）", "(no output)") }
        return Result(output: output, isError: isError)
    }

    private func evaluate(_ script: String, in webView: WKWebView, timeout: Double) async -> String? {
        await withTaskGroup(of: String?.self) { group in
            group.addTask { @MainActor in
                do {
                    let result = try await webView.evaluateJavaScript(script)
                    return (result as? String) ?? String(describing: result)
                } catch {
                    return "__error__" + error.localizedDescription
                }
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    static func format(_ message: String) -> String {
        let parts = message.split(separator: "\t", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return message }
        return parts[0] == "log" ? parts[1] : "[\(parts[0])] \(parts[1])"
    }

    nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            self.loadContinuation?.resume()
            self.loadContinuation = nil
        }
    }

    nonisolated func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in
            self.loadContinuation?.resume()
            self.loadContinuation = nil
        }
    }
}

/// WKUserContentController が強く持たないように、弱い参照で受け渡す。
final class WeakMessageHandler: NSObject, WKScriptMessageHandler {
    private let onMessage: @MainActor (String) -> Void

    init(_ onMessage: @escaping @MainActor (String) -> Void) {
        self.onMessage = onMessage
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        let text = (message.body as? String) ?? String(describing: message.body)
        MainActor.assumeIsolated { onMessage(text) }
    }
}
