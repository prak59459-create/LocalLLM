import SwiftUI
import UniformTypeIdentifiers

/// アプリの更新画面（設定から開く）。
struct UpdateView: View {
    @State private var updater = AppUpdateController()
    @State private var importing: ImportKind?

    private enum ImportKind: Identifiable {
        case target, source
        var id: Self { self }
    }

    private static var swiftpmType: UTType { UTType(filenameExtension: "swiftpm") ?? .package }

    var body: some View {
        Form {
            Section {
                LabeledContent(L("動いている版", "Running version"), value: AppUpdateController.runningVersion)
            } footer: {
                Text(L("新しいソースで Swift Playgrounds のアプリ（.swiftpm）を上書きします。更新するたびに版（bundleVersion と displayVersion）を上げます。変える前のファイルは取っておくので、取り消せます。",
                       "Overwrites the Swift Playgrounds app (.swiftpm) with new source and raises the version every time. The previous files are kept so you can undo."))
            }
            UpdateTargetSection(updater: updater, onPick: { importing = .target })
            UpdateSourceSection(updater: updater, onPick: { importing = .source })
            UpdateActionSection(updater: updater)
        }
        .navigationTitle(L("アプリの更新", "Update the App"))
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
                      allowedContentTypes: importing == .target ? [Self.swiftpmType, .package, .folder]
                                                                 : [.zip, Self.swiftpmType, .package, .folder]) { result in
            guard case .success(let url) = result else { return }
            if importing == .target { updater.setTarget(url) } else { updater.setSourceFile(url) }
            importing = nil
        }
    }
}

private struct UpdateTargetSection: View {
    let updater: AppUpdateController
    var onPick: () -> Void

    var body: some View {
        Section {
            LabeledContent(L("更新するアプリ", "App to update"), value: updater.targetName ?? L("未選択", "Not chosen"))
            Button(updater.targetName == nil ? L("アプリの .swiftpm を選ぶ…", "Choose the app's .swiftpm…")
                                             : L("選び直す…", "Choose again…"), action: onPick)
        } footer: {
            Text(L("Swift Playgrounds で開いているこのアプリのプロジェクト（例: iCloud Drive › Playgrounds › LocalLLM.swiftpm）を選んでください。次からは選ばなくても更新できます。",
                   "Choose this app's project in Swift Playgrounds (e.g. iCloud Drive › Playgrounds › LocalLLM.swiftpm). It is remembered for next time."))
        }
    }
}

private struct UpdateSourceSection: View {
    @Bindable var updater: AppUpdateController
    var onPick: () -> Void

    var body: some View {
        Section {
            Picker(L("新しいソース", "New source"), selection: $updater.source) {
                ForEach(AppUpdateController.Source.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            if updater.source == .github {
                TextField(L("リポジトリ（owner/name）", "Repository (owner/name)"), text: $updater.repository)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField(L("ブランチ", "Branch"), text: $updater.branch)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField(L("トークン（非公開のリポジトリだけ）", "Token (private repositories only)"), text: $updater.token)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            } else {
                LabeledContent(L("ファイル", "File"), value: updater.sourceFileName ?? L("未選択", "Not chosen"))
                Button(L(".zip か .swiftpm を選ぶ…", "Choose a .zip or .swiftpm…"), action: onPick)
            }
        } footer: {
            Text(updater.source == .github
                 ? L("GitHub のブランチの最新のソースを ZIP で取ってきます。", "Downloads the latest source of the branch from GitHub as a ZIP.")
                 : L("GitHub の「Download ZIP」で保存したファイルや、別の .swiftpm を使えます。", "Use a ZIP from GitHub's \"Download ZIP\" or another .swiftpm."))
        }
    }
}

private struct UpdateActionSection: View {
    let updater: AppUpdateController

    var body: some View {
        Section {
            switch updater.phase {
            case .idle:
                EmptyView()
            case .working(let message):
                HStack {
                    ProgressView()
                    Text(message).foregroundStyle(.secondary)
                }
            case .ready(let summary):
                UpdateSummaryView(summary: summary)
                Button {
                    Task { await updater.apply() }
                } label: {
                    Label(L("上書きして更新する", "Overwrite and update"), systemImage: "arrow.down.app.fill")
                }
                .buttonStyle(.borderedProminent)
            case .done(let message):
                Label(message, systemImage: "checkmark.seal.fill").foregroundStyle(.green)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
            Button {
                Task { await updater.prepare() }
            } label: {
                Label(L("更新を確認する", "Check for update"), systemImage: "arrow.triangle.2.circlepath")
            }
            .disabled(!updater.canPrepare)
            if updater.hasBackup {
                Button(role: .destructive) {
                    Task { await updater.restoreLastUpdate() }
                } label: {
                    Label(L("前回の更新を取り消す", "Undo the last update"), systemImage: "arrow.uturn.backward")
                }
            }
        }
    }
}

private struct UpdateSummaryView: View {
    let summary: AppUpdateController.UpdatePlanSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(summary.from).monospacedDigit()
                Image(systemName: "arrow.right")
                Text(summary.to).monospacedDigit().bold()
            }
            .font(.headline)
            Text(L("追加 \(summary.added.count)・変更 \(summary.changed.count)・削除 \(summary.removed.count)・同じ \(summary.unchanged)",
                   "Added \(summary.added.count) · Changed \(summary.changed.count) · Removed \(summary.removed.count) · Same \(summary.unchanged)"))
                .font(.callout)
                .foregroundStyle(.secondary)
            FileListGroup(title: L("追加", "Added"), files: summary.added, color: .green)
            FileListGroup(title: L("変更", "Changed"), files: summary.changed, color: .blue)
            FileListGroup(title: L("削除", "Removed"), files: summary.removed, color: .red)
        }
    }
}

private struct FileListGroup: View {
    let title: String
    let files: [String]
    let color: Color

    var body: some View {
        if !files.isEmpty {
            DisclosureGroup("\(title) (\(files.count))") {
                ForEach(files, id: \.self) { file in
                    Text(file)
                        .font(.caption.monospaced())
                        .foregroundStyle(color)
                }
            }
            .font(.callout)
        }
    }
}
