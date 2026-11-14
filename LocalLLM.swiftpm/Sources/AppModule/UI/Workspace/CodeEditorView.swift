import SwiftUI
import UIKit

/// エディタの表示: 開いているファイルのタブと、色付きのコードエディタ。
struct EditorPane: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        VStack(spacing: 0) {
            if !session.openTabs.isEmpty {
                EditorTabBar(session: session)
                Rectangle().fill(CDWSTheme.stroke).frame(height: 1)
            }
            if let path = session.selectedPath {
                CodeEditorView(text: session.text(for: path),
                               revision: session.revision(for: path),
                               path: path,
                               onChange: { session.updateBuffer(path, text: $0) })
                    .id(path)
                EditorStatusBar(session: session, path: path)
            } else {
                EditorPlaceholder()
            }
        }
    }
}

private struct EditorTabBar: View {
    @Bindable var session: WorkspaceSession

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 2) {
                ForEach(session.openTabs, id: \.self) { path in
                    tab(path)
                }
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 5)
        }
        .background(CDWSTheme.panel)
    }

    private func tab(_ path: String) -> EditorTab {
        EditorTab(path: path,
                  isSelected: session.selectedPath == path,
                  isDirty: session.isDirty(path),
                  onSelect: { session.selectedPath = path },
                  onClose: { session.close(path) })
    }
}

private struct EditorTab: View {
    let path: String
    let isSelected: Bool
    let isDirty: Bool
    var onSelect: () -> Void
    var onClose: () -> Void

    private var tabBackground: Color { isSelected ? CDWSTheme.raised : Color.clear }

    private var entry: WorkspaceEntry {
        WorkspaceEntry(path: path, isDirectory: false, size: 0)
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: FileIcon.symbol(for: entry))
                .font(.caption)
                .foregroundStyle(FileIcon.color(for: entry))
            Text((path as NSString).lastPathComponent)
                .font(.callout)
                .foregroundStyle(isSelected ? CDWSTheme.text : CDWSTheme.secondaryText)
                .lineLimit(1)
            Button(action: onClose) {
                Image(systemName: isDirty ? "circle.fill" : "xmark")
                    .font(.system(size: isDirty ? 7 : 10, weight: .bold))
                    .frame(width: 16, height: 16)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(tabBackground, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(alignment: .top) {
            if isSelected { Capsule().fill(CDWSTheme.accent).frame(height: 2).padding(.horizontal, 8) }
        }
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
        .help(path)
    }
}

private struct EditorStatusBar: View {
    @Bindable var session: WorkspaceSession
    let path: String

    var body: some View {
        HStack(spacing: 12) {
            Text(path)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Text(CodeLanguage.from(path: path).rawValue)
            if session.isDirty(path) {
                Button(L("保存", "Save")) { session.save(path) }
                    .keyboardShortcut("s", modifiers: .command)
            } else {
                Label(L("保存済み", "Saved"), systemImage: "checkmark")
            }
            if path.lowercased().hasSuffix(".js") {
                Button {
                    session.runJavaScript(path: path)
                    session.pane = .preview
                } label: {
                    Label(L("実行", "Run"), systemImage: "play.fill")
                }
                .keyboardShortcut("r", modifiers: .command)
            }
        }
        .font(.caption)
        .foregroundStyle(CDWSTheme.secondaryText)
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(CDWSTheme.panel)
    }
}

private struct EditorPlaceholder: View {
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(L("左のファイルを選ぶか、右のエージェントに作りたいものを伝えてください。",
                   "Pick a file on the left, or tell the agent on the right what to build."))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// UITextView を使ったコードエディタ（色付け・自動の字下げ・記号キー・検索）。
struct CodeEditorView: UIViewRepresentable {
    let text: String
    /// エージェントが書き換えたときに増える。増えたら中身を入れ替える。
    let revision: Int
    let path: String
    var onChange: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UITextView {
        let view = UITextView()
        view.font = Coordinator.font
        view.backgroundColor = .systemBackground
        view.autocorrectionType = .no
        view.autocapitalizationType = .none
        view.spellCheckingType = .no
        view.smartQuotesType = .no
        view.smartDashesType = .no
        view.smartInsertDeleteType = .no
        view.keyboardType = .asciiCapable
        view.alwaysBounceVertical = true
        view.isFindInteractionEnabled = true
        view.textContainerInset = UIEdgeInsets(top: 12, left: 10, bottom: 40, right: 10)
        view.delegate = context.coordinator
        view.inputAccessoryView = context.coordinator.makeAccessoryBar()
        context.coordinator.textView = view
        view.text = text
        context.coordinator.lastRevision = revision
        context.coordinator.highlight()
        return view
    }

    func updateUIView(_ view: UITextView, context: Context) {
        context.coordinator.parent = self
        if context.coordinator.lastRevision != revision {
            context.coordinator.lastRevision = revision
            let selection = view.selectedRange
            view.text = text
            view.selectedRange = NSRange(location: min(selection.location, (text as NSString).length), length: 0)
            context.coordinator.highlight()
        }
    }

    @MainActor
    final class Coordinator: NSObject, UITextViewDelegate {
        static let font = UIFont.monospacedSystemFont(ofSize: 14, weight: .regular)

        var parent: CodeEditorView
        weak var textView: UITextView?
        var lastRevision = 0
        private var highlightTask: Task<Void, Never>?

        init(_ parent: CodeEditorView) {
            self.parent = parent
        }

        func textViewDidChange(_ textView: UITextView) {
            parent.onChange(textView.text)
            scheduleHighlight()
        }

        /// 改行で前の行の字下げを引き継ぐ。{ や : のあとなら 1 段深くする。
        func textView(_ textView: UITextView, shouldChangeTextIn range: NSRange, replacementText text: String) -> Bool {
            guard text == "\n" else { return true }
            let content = textView.text as NSString
            let lineRange = content.lineRange(for: NSRange(location: range.location, length: 0))
            let line = content.substring(with: NSRange(location: lineRange.location, length: range.location - lineRange.location))
            var indent = String(line.prefix { $0 == " " || $0 == "\t" })
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasSuffix("{") || trimmed.hasSuffix("[") || trimmed.hasSuffix("(") || trimmed.hasSuffix(":") {
                indent += "    "
            }
            guard let start = textView.position(from: textView.beginningOfDocument, offset: range.location),
                  let end = textView.position(from: start, offset: range.length),
                  let textRange = textView.textRange(from: start, to: end) else { return true }
            textView.replace(textRange, withText: "\n" + indent)
            parent.onChange(textView.text)
            scheduleHighlight()
            return false
        }

        private func replaceSelection(with text: String) {
            guard let textView, let range = textView.selectedTextRange else { return }
            textView.replace(range, withText: text)
            parent.onChange(textView.text)
            scheduleHighlight()
        }

        func scheduleHighlight() {
            highlightTask?.cancel()
            highlightTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: 180_000_000)
                guard !Task.isCancelled else { return }
                self?.highlight()
            }
        }

        func highlight() {
            guard let textView else { return }
            let text = textView.text ?? ""
            let tokens = SyntaxHighlighter.tokens(in: text, language: CodeLanguage.from(path: parent.path))
            let storage = textView.textStorage
            let length = storage.length
            storage.beginEditing()
            storage.setAttributes([.font: Self.font, .foregroundColor: UIColor.label], range: NSRange(location: 0, length: length))
            for token in tokens where token.location + token.length <= length {
                storage.addAttribute(.foregroundColor, value: Self.color(for: token.kind),
                                     range: NSRange(location: token.location, length: token.length))
            }
            storage.endEditing()
            textView.typingAttributes = [.font: Self.font, .foregroundColor: UIColor.label]
        }

        static func color(for kind: HighlightToken.Kind) -> UIColor {
            switch kind {
            case .keyword: return .systemPink
            case .string: return .systemRed
            case .comment: return .systemGreen
            case .number: return .systemBlue
            case .type: return .systemTeal
            case .attribute: return .systemPurple
            case .tag: return .systemBlue
            case .heading: return .systemOrange
            case .punctuation: return .secondaryLabel
            }
        }

        /// キーボードの上の記号キー
        func makeAccessoryBar() -> UIView {
            let bar = UIInputView(frame: CGRect(x: 0, y: 0, width: 0, height: 44), inputViewStyle: .keyboard)
            let stack = UIStackView()
            stack.axis = .horizontal
            stack.spacing = 6
            stack.distribution = .fillEqually
            stack.translatesAutoresizingMaskIntoConstraints = false
            let keys = ["⇥", "{", "}", "(", ")", "[", "]", "<", ">", "=", ";", ":", "\"", "'", "/", "⌄"]
            for key in keys {
                let button = UIButton(type: .system)
                button.setTitle(key, for: .normal)
                button.titleLabel?.font = .monospacedSystemFont(ofSize: 17, weight: .medium)
                button.backgroundColor = .secondarySystemBackground
                button.layer.cornerRadius = 6
                button.addAction(UIAction { [weak self] _ in self?.tapKey(key) }, for: .touchUpInside)
                stack.addArrangedSubview(button)
            }
            bar.addSubview(stack)
            NSLayoutConstraint.activate([
                stack.leadingAnchor.constraint(equalTo: bar.leadingAnchor, constant: 8),
                stack.trailingAnchor.constraint(equalTo: bar.trailingAnchor, constant: -8),
                stack.topAnchor.constraint(equalTo: bar.topAnchor, constant: 5),
                stack.bottomAnchor.constraint(equalTo: bar.bottomAnchor, constant: -5),
            ])
            return bar
        }

        private func tapKey(_ key: String) {
            switch key {
            case "⇥": replaceSelection(with: "    ")
            case "⌄": textView?.resignFirstResponder()
            default: replaceSelection(with: key)
            }
        }
    }
}
