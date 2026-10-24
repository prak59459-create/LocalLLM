import SwiftUI
import UniformTypeIdentifiers

/// アプリの更新画面（設定から開く）。GitHub とこの iPad の .swiftpm を比べて、変わったファイルだけを書き換える。
struct UpdateView: View {
    @Environment(AppUpdateController.self) private var updater
    @State private var importing: ImportKind?
    @State private var previewing: UpdateFileChange?
    @State private var confirmingRestore = false

    private enum ImportKind: Identifiable {
        case target, source
        var id: Self { self }
    }

    private static var swiftpmType: UTType { UTType(filenameExtension: "swiftpm", conformingTo: .package) ?? .package }

    private var allowedTypes: [UTType] {
        // .swiftpm がフォルダに見えても、パッケージに見えても選べるように、.item まで含める
        importing == .target ? [Self.swiftpmType, .package, .folder, .item] : [.zip, Self.swiftpmType, .package, .folder]
    }

    var body: some View {
        Form {
            UpdateOverviewSection(updater: updater)
            UpdateTargetSection(updater: updater, onPick: { importing = .target })
            UpdateSourceSection(updater: updater, onPick: { importing = .source })
            UpdateActionSection(updater: updater)
            if let result = updater.result, !result.diff.changes.isEmpty || !result.diff.keptLocalFiles.isEmpty {
                UpdateChangesSection(updater: updater, result: result, onPreview: { previewing = $0 })
            }
            UpdateHistorySection(updater: updater, onRestore: { confirmingRestore = true })
            UpdateMaintenanceSection(updater: updater)
        }
        .navigationTitle(L("アプリの更新", "Update the App"))
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
                      allowedContentTypes: allowedTypes) { result in
            handleImport(result)
        }
        .sheet(item: $previewing) { change in
            UpdateFilePreviewView(updater: updater, change: change)
        }
        .confirmationDialog(L("前回の更新を取り消しますか？", "Undo the last update?"), isPresented: $confirmingRestore,
                            titleVisibility: .visible) {
            Button(L("取り消す", "Undo"), role: .destructive) { updater.restoreLatest() }
        } message: {
            Text(L("書き換える前のファイルに戻し、更新で足したファイルを消します。", "Restores the files from before the update and removes files it added."))
        }
        .task {
            updater.refreshTarget()
            updater.refreshCacheSize()
            if updater.source == .github { await updater.refreshRateLimit() }
        }
    }

    private func handleImport(_ result: Result<URL, Error>) {
        let kind = importing
        importing = nil
        guard case .success(let url) = result else { return }
        if kind == .target { updater.pickTarget(url) } else { updater.pickSourceFile(url) }
    }
}

// MARK: - 概要

private struct UpdateOverviewSection: View {
    let updater: AppUpdateController

    var body: some View {
        Section {
            LabeledContent(L("動いている版", "Running version"), value: AppUpdateController.runningVersion)
            if let version = updater.targetSummary?.version, version.description != AppUpdateController.runningVersion {
                LabeledContent(L(".swiftpm の版", "Version in .swiftpm"), value: version.description)
            }
            if let commit = updater.availableUpdate {
                UpdateAvailableBanner(commit: commit)
            }
        } footer: {
            Text(L("GitHub のファイルとこの iPad の .swiftpm を 1 つずつ比べ、変わったファイルだけをダウンロードして書き換えます。同じファイルには触りません。更新するたびに版を上げ、書き換える前のファイルは残すので取り消せます。",
                   "Compares each file on GitHub with the .swiftpm on this iPad and downloads and rewrites only the files that changed. Unchanged files are left alone. The version goes up every time, and the previous files are kept so you can undo."))
        }
    }
}

private struct UpdateAvailableBanner: View {
    let commit: GitHubCommit

    var body: some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text(L("新しい版があります", "A new version is available")).bold()
                Text("\(commit.shortSHA) · \(commit.title)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        } icon: {
            Image(systemName: "sparkles").foregroundStyle(.orange)
        }
    }
}

// MARK: - 更新するアプリ

private struct UpdateTargetSection: View {
    let updater: AppUpdateController
    var onPick: () -> Void

    var body: some View {
        Section {
            switch updater.targetStatus {
            case .none:
                Text(L("まだ選んでいません", "Not chosen yet")).foregroundStyle(.secondary)
            case .ok(let summary):
                UpdateTargetSummaryView(summary: summary)
            case .choosing(let folder, let candidates):
                UpdateCandidateList(updater: updater, folder: folder, candidates: candidates)
            case .problem(let message):
                Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
            Button(action: onPick) {
                Label(pickTitle, systemImage: "folder")
            }
            if updater.targetSummary != nil {
                Button(L("選ぶのをやめる", "Forget this app"), role: .destructive) { updater.forgetTarget() }
            }
        } header: {
            Text(L("更新するアプリ", "App to update"))
        } footer: {
            Text(L("「ファイル」で iCloud Drive › Playgrounds › LocalLLM.swiftpm を選んでください。.swiftpm がうまく選べないときは、Playgrounds フォルダを開いて右上の「開く」を押すと、中のアプリから選べます。次からは選ばなくても更新できます。",
                   "In Files, choose iCloud Drive › Playgrounds › LocalLLM.swiftpm. If the .swiftpm cannot be selected, open the Playgrounds folder and tap Open to pick the app inside it. It is remembered for next time."))
        }
    }

    private var pickTitle: String {
        updater.targetSummary == nil ? L("アプリの .swiftpm を選ぶ…", "Choose the app's .swiftpm…") : L("選び直す…", "Choose again…")
    }
}

private struct UpdateTargetSummaryView: View {
    let summary: PackageSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(summary.folderName, systemImage: summary.looksLikeThisApp ? "checkmark.seal.fill" : "questionmark.folder")
                .font(.headline)
                .foregroundStyle(summary.looksLikeThisApp ? Color.green : Color.orange)
            Text(summary.location)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(details)
                .font(.caption)
                .foregroundStyle(.secondary)
            if !summary.looksLikeThisApp {
                Text(L("LocalLLM のプロジェクトではないようです。別のアプリを上書きしないよう、選んだものを確かめてください。",
                       "This does not look like the LocalLLM project. Make sure you are not overwriting another app."))
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 2)
    }

    private var details: String {
        let version = summary.version?.description ?? "?"
        return L("版 \(version)・Swift ファイル \(summary.swiftFileCount) 個", "Version \(version) · \(summary.swiftFileCount) Swift files")
    }
}

private struct UpdateCandidateList: View {
    let updater: AppUpdateController
    let folder: String
    let candidates: [String]

    var body: some View {
        Text(L("「\(folder)」の中のどのアプリを更新しますか？", "Which app in \"\(folder)\" should be updated?"))
            .font(.callout)
        ForEach(candidates, id: \.self) { path in
            Button {
                updater.chooseCandidate(path)
            } label: {
                Label(path, systemImage: "shippingbox")
            }
        }
    }
}

// MARK: - 新しいソース

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
                UpdateGitHubFields(updater: updater)
            } else {
                LabeledContent(L("ファイル", "File"), value: updater.sourceFileName ?? L("未選択", "Not chosen"))
                Button(L(".zip か .swiftpm を選ぶ…", "Choose a .zip or .swiftpm…"), action: onPick)
            }
        } header: {
            Text(L("新しいソース", "New source"))
        } footer: {
            Text(footer)
        }
    }

    private var footer: String {
        updater.source == .github
            ? L("ブランチの最新のコミットのファイル一覧を取って比べ、違うファイルだけをダウンロードします（中身は GitHub の一覧と同じか確かめます）。",
                "Fetches the file list of the branch's latest commit, compares it, and downloads only the files that differ (each is verified against GitHub's listing).")
            : L("GitHub の「Download ZIP」で保存したファイルや、別の .swiftpm と比べます。", "Compares with a ZIP from GitHub's \"Download ZIP\" or another .swiftpm.")
    }
}

private struct UpdateGitHubFields: View {
    @Bindable var updater: AppUpdateController

    var body: some View {
        TextField(L("リポジトリ（owner/name）", "Repository (owner/name)"), text: $updater.repository)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
        HStack {
            TextField(L("ブランチ", "Branch"), text: $updater.branch)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            UpdateBranchMenu(updater: updater)
        }
        SecureField(L("トークン（非公開のリポジトリだけ）", "Token (private repositories only)"), text: $updater.token)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
        Toggle(L("起動したときに新しい版を確かめる", "Check for a new version at launch"), isOn: $updater.checkAutomatically)
        if let limit = updater.rateLimit {
            UpdateRateLimitRow(limit: limit)
        }
    }
}

private struct UpdateBranchMenu: View {
    let updater: AppUpdateController

    var body: some View {
        Menu {
            if updater.branches.isEmpty {
                Text(L("「読み込む」を押してください", "Tap Load"))
            }
            ForEach(updater.branches, id: \.self) { name in
                Button(name) { updater.branch = name }
            }
            Divider()
            Button(L("ブランチを読み込む", "Load branches")) {
                Task { await updater.loadBranches() }
            }
        } label: {
            Image(systemName: "arrow.triangle.branch")
        }
    }
}

private struct UpdateRateLimitRow: View {
    let limit: GitHubRateLimit

    var body: some View {
        LabeledContent(L("GitHub の残り回数", "GitHub requests left")) {
            Text(text).monospacedDigit().foregroundStyle(limit.remaining < 5 ? Color.orange : Color.secondary)
        }
        .font(.caption)
    }

    private var text: String {
        let reset = limit.reset.map { DateFormatter.localizedString(from: $0, dateStyle: .none, timeStyle: .short) } ?? "?"
        return L("\(limit.remaining)/\(limit.limit)（\(reset) に回復）", "\(limit.remaining)/\(limit.limit) (resets \(reset))")
    }
}

// MARK: - 確認と更新

private struct UpdateActionSection: View {
    let updater: AppUpdateController

    var body: some View {
        Section {
            UpdatePhaseView(updater: updater)
            if updater.isWorking {
                Button(L("やめる", "Cancel"), role: .cancel) { updater.cancel() }
            } else {
                Button {
                    updater.check()
                } label: {
                    Label(L("更新を確認する", "Check for update"), systemImage: "arrow.triangle.2.circlepath")
                }
                .disabled(!updater.canCheck)
            }
        }
    }
}

private struct UpdatePhaseView: View {
    let updater: AppUpdateController

    var body: some View {
        switch updater.phase {
        case .idle:
            EmptyView()
        case .working(let message, let fraction):
            UpdateProgressRow(message: message, fraction: fraction)
        case .upToDate:
            if let result = updater.result {
                UpdateUpToDateView(result: result)
                if !updater.selectedChanges.isEmpty { UpdateApplyButton(updater: updater) }
            }
        case .ready:
            if let result = updater.result { UpdateProposalView(updater: updater, result: result) }
        case .applied(let record):
            UpdateAppliedView(record: record)
        case .restored(let message):
            Label(message, systemImage: "arrow.uturn.backward.circle.fill").foregroundStyle(.blue)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
    }
}

private struct UpdateProgressRow: View {
    let message: String
    let fraction: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                ProgressView()
                Text(message).foregroundStyle(.secondary)
            }
            if let fraction {
                ProgressView(value: fraction)
            }
        }
    }
}

private struct UpdateCommitView: View {
    let commit: GitHubCommit

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(commit.title).font(.callout).lineLimit(3)
            Text(detail).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var detail: String {
        let date = commit.date.map { DateFormatter.localizedString(from: $0, dateStyle: .medium, timeStyle: .short) } ?? ""
        return [commit.shortSHA, commit.author ?? "", date].filter { !$0.isEmpty }.joined(separator: " · ")
    }
}

private struct UpdateUpToDateView: View {
    let result: UpdateCheckResult

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(L("最新です。GitHub と同じファイルです。", "Up to date. The files match GitHub."), systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            if let commit = result.commit { UpdateCommitView(commit: commit) }
            Text(L("同じファイル \(result.diff.unchangedCount) 個", "\(result.diff.unchangedCount) identical files"))
                .font(.caption)
                .foregroundStyle(.secondary)
            if !result.diff.changes.isEmpty {
                Text(L("この iPad でだけ変えたファイルが \(result.diff.changes.count) 個あります（下で GitHub の内容に戻せます）。",
                       "\(result.diff.changes.count) files were changed only on this iPad (you can revert them below)."))
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }
}

private struct UpdateProposalView: View {
    let updater: AppUpdateController
    let result: UpdateCheckResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            UpdateVersionLine(from: result.currentVersion?.description ?? "?", to: result.newVersion().description)
            if let commit = result.commit { UpdateCommitView(commit: commit) }
            UpdateCountsLine(diff: result.diff)
            if !result.commitsSinceBase.isEmpty {
                UpdateCommitList(commits: result.commitsSinceBase)
            }
            if result.diff.manifestDiffersOnlyInSettings {
                Text(L("Package.swift は版だけを書き換えます（チームやアイコンなど、この iPad の設定は残します）。",
                       "Only the version in Package.swift is rewritten (this iPad's team, icon and other settings are kept)."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        UpdateApplyButton(updater: updater)
    }
}

private struct UpdateApplyButton: View {
    let updater: AppUpdateController

    var body: some View {
        Button {
            updater.apply()
        } label: {
            Label(title, systemImage: "arrow.down.app.fill")
        }
        .buttonStyle(.borderedProminent)
        .disabled(updater.selectedChanges.isEmpty)
    }

    private var title: String {
        let count = updater.selectedChanges.count
        let version = updater.plannedVersion?.description ?? "?"
        return L("選んだ \(count) 個のファイルを更新する（\(version)）", "Update the \(count) selected files (\(version))")
    }
}

private struct UpdateVersionLine: View {
    let from: String
    let to: String

    var body: some View {
        HStack {
            Text(from).monospacedDigit()
            Image(systemName: "arrow.right")
            Text(to).monospacedDigit().bold()
        }
        .font(.headline)
    }
}

private struct UpdateCountsLine: View {
    let diff: UpdateDiff

    var body: some View {
        Text(text).font(.callout).foregroundStyle(.secondary)
    }

    private var text: String {
        let modified = diff.changes(of: .modified).count
        let added = diff.changes(of: .added).count
        let removed = diff.changes(of: .removed).count
        let local = diff.changes(of: .localEdit).count + diff.changes(of: .localDeleted).count
        var parts = [L("変更 \(modified)", "Changed \(modified)"), L("追加 \(added)", "Added \(added)"),
                     L("削除 \(removed)", "Removed \(removed)"), L("同じ \(diff.unchangedCount)", "Same \(diff.unchangedCount)")]
        if local > 0 { parts.append(L("この iPad だけ \(local)", "Only here \(local)")) }
        return parts.joined(separator: "・")
    }
}

private struct UpdateCommitList: View {
    let commits: [GitHubCommit]

    var body: some View {
        DisclosureGroup(L("前回の更新からのコミット（\(commits.count)）", "Commits since the last update (\(commits.count))")) {
            ForEach(commits.prefix(50), id: \.sha) { commit in
                UpdateCommitView(commit: commit)
            }
        }
        .font(.callout)
    }
}

private struct UpdateAppliedView: View {
    let record: UpdateRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(L("\(record.newVersion.description) に更新しました", "Updated to \(record.newVersion.description)"),
                  systemImage: "checkmark.seal.fill")
                .font(.headline)
                .foregroundStyle(.green)
            Text(L("\(record.changedFileCount) 個のファイルを書き換え、中身が新しいソースと同じことを確かめました。ほかのファイルには触っていません。",
                   "Rewrote \(record.changedFileCount) files and verified they match the new source. No other files were touched."))
                .font(.callout)
            Text(L("新しい版を動かすには: ① このアプリを閉じる ② Swift Playgrounds で LocalLLM を開き直す ③ ▶︎ を押す",
                   "To run the new version: ① close this app ② reopen LocalLLM in Swift Playgrounds ③ press ▶︎"))
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - 変わったファイル

private struct UpdateChangesSection: View {
    let updater: AppUpdateController
    let result: UpdateCheckResult
    var onPreview: (UpdateFileChange) -> Void

    var body: some View {
        Section {
            UpdateSelectionButtons(updater: updater)
            ForEach(remoteChanges) { change in
                UpdateChangeRow(updater: updater, change: change, onPreview: onPreview)
            }
        } header: {
            Text(L("新しいソースで変わったファイル（\(remoteChanges.count)）", "Changed in the new source (\(remoteChanges.count))"))
        } footer: {
            Text(L("チェックしたファイルだけを書き換えます。行をタップすると差分を見られます。「重なり」はこの iPad でも書き換えていたファイルです。",
                   "Only checked files are rewritten. Tap a row to see the diff. \"Overlap\" means the file was also edited on this iPad."))
        }
        if !localChanges.isEmpty {
            Section {
                ForEach(localChanges) { change in
                    UpdateChangeRow(updater: updater, change: change, onPreview: onPreview)
                }
            } header: {
                Text(L("この iPad でだけ変わったファイル（\(localChanges.count)）", "Changed only on this iPad (\(localChanges.count))"))
            } footer: {
                Text(L("新しいソースは前回の更新から変わっていません。チェックすると新しいソースの内容に戻します。",
                       "The new source has not changed since the last update. Check a file to revert it to the new source."))
            }
        }
        if !result.diff.keptLocalFiles.isEmpty {
            Section {
                ForEach(result.diff.keptLocalFiles, id: \.self) { path in
                    Text(path).font(.caption.monospaced())
                }
            } header: {
                Text(L("この iPad で足したファイル（残します）", "Files added on this iPad (kept)"))
            }
        }
    }

    private var remoteChanges: [UpdateFileChange] {
        result.diff.changes.filter { $0.kind.selectedByDefault }
    }

    private var localChanges: [UpdateFileChange] {
        result.diff.changes.filter { !$0.kind.selectedByDefault }
    }
}

private struct UpdateSelectionButtons: View {
    let updater: AppUpdateController

    var body: some View {
        HStack {
            Button(L("おすすめ", "Recommended")) { updater.selectRecommended() }
            Spacer()
            Button(L("全部", "All")) { updater.selectAll() }
            Spacer()
            Button(L("なし", "None")) { updater.selectNone() }
        }
        .buttonStyle(.borderless)
        .font(.callout)
    }
}

private struct UpdateChangeRow: View {
    let updater: AppUpdateController
    let change: UpdateFileChange
    var onPreview: (UpdateFileChange) -> Void

    var body: some View {
        HStack(spacing: 10) {
            Toggle("", isOn: isOn).labelsHidden()
            Button {
                onPreview(change)
            } label: {
                UpdateChangeLabel(change: change)
            }
            .buttonStyle(.plain)
        }
    }

    private var isOn: Binding<Bool> {
        Binding(get: { updater.selection.contains(change.path) }, set: { updater.setSelected(change.path, $0) })
    }
}

private struct UpdateChangeLabel: View {
    let change: UpdateFileChange

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: UpdateKindStyle.icon(change.kind)).foregroundStyle(UpdateKindStyle.color(change.kind))
            VStack(alignment: .leading, spacing: 1) {
                Text(fileName).font(.callout.monospaced()).lineLimit(1)
                Text(folder).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 4)
            if change.editedHere {
                UpdateBadge(text: L("重なり", "Overlap"), color: .orange)
            }
            UpdateBadge(text: UpdateKindStyle.label(change.kind), color: UpdateKindStyle.color(change.kind))
            Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
        }
        .contentShape(Rectangle())
    }

    private var fileName: String { change.path.split(separator: "/").last.map(String.init) ?? change.path }
    private var folder: String { change.path.split(separator: "/").dropLast().joined(separator: "/") }
}

private struct UpdateBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

private enum UpdateKindStyle {
    static func label(_ kind: FileChangeKind) -> String {
        switch kind {
        case .added: return L("追加", "Added")
        case .modified: return L("変更", "Changed")
        case .removed: return L("削除", "Removed")
        case .localEdit: return L("ここで変更", "Edited here")
        case .localDeleted: return L("ここで削除", "Deleted here")
        }
    }

    static func icon(_ kind: FileChangeKind) -> String {
        switch kind {
        case .added: return "plus.circle.fill"
        case .modified: return "pencil.circle.fill"
        case .removed: return "minus.circle.fill"
        case .localEdit: return "ipad"
        case .localDeleted: return "trash.circle"
        }
    }

    static func color(_ kind: FileChangeKind) -> Color {
        switch kind {
        case .added: return .green
        case .modified: return .blue
        case .removed: return .red
        case .localEdit, .localDeleted: return .orange
        }
    }
}

// MARK: - 差分

private struct UpdateFilePreviewView: View {
    let updater: AppUpdateController
    let change: UpdateFileChange
    @State private var preview: AppUpdateController.FilePreview?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            UpdatePreviewScroll(path: change.path, preview: preview)
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { closeButton }
                .task { preview = await updater.preview(change) }
        }
    }

    private var title: String { PackageScanner.lastComponent(change.path) }

    private var closeButton: some ToolbarContent {
        ToolbarItem(placement: .confirmationAction) {
            Button(L("閉じる", "Close")) { dismiss() }
        }
    }
}

private struct UpdatePreviewScroll: View {
    let path: String
    let preview: AppUpdateController.FilePreview?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Text(path).font(.caption.monospaced()).foregroundStyle(.secondary)
                UpdatePreviewContent(preview: preview)
            }
            .padding()
        }
    }
}

private struct UpdatePreviewContent: View {
    let preview: AppUpdateController.FilePreview?

    var body: some View {
        if let preview {
            if let note = preview.note {
                Text(note).foregroundStyle(.secondary)
            } else {
                Text(stats(preview)).font(.caption).foregroundStyle(.secondary)
                DiffView(before: preview.before, after: preview.after, maxLines: 2000)
            }
        } else {
            ProgressView()
        }
    }

    private func stats(_ preview: AppUpdateController.FilePreview) -> String {
        let counts = TextDiff.stats(old: preview.before, new: preview.after)
        return L("+\(counts.added) 行・−\(counts.removed) 行", "+\(counts.added) lines · −\(counts.removed) lines")
    }
}

// MARK: - 履歴

private struct UpdateHistorySection: View {
    let updater: AppUpdateController
    var onRestore: () -> Void

    var body: some View {
        if !updater.history.isEmpty {
            Section {
                ForEach(updater.history.prefix(10)) { record in
                    UpdateHistoryRow(record: record)
                }
                if let restorable = updater.restorable {
                    Button(role: .destructive, action: onRestore) {
                        Label(L("\(restorable.newVersion.description) の更新を取り消す", "Undo the update to \(restorable.newVersion.description)"),
                              systemImage: "arrow.uturn.backward")
                    }
                    .disabled(updater.isWorking)
                }
            } header: {
                Text(L("更新の履歴", "Update history"))
            } footer: {
                Text(L("新しい更新から順に取り消せます。", "Updates can be undone starting from the newest."))
            }
        }
    }
}

private struct UpdateHistoryRow: View {
    let record: UpdateRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("\(record.previousVersion?.description ?? "?") → \(record.newVersion.description)").monospacedDigit().bold()
                if record.restored {
                    UpdateBadge(text: L("取り消し済み", "Undone"), color: .secondary)
                }
                Spacer()
                Text(record.date, style: .date).font(.caption).foregroundStyle(.secondary)
            }
            Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
        }
    }

    private var detail: String {
        let commit = record.commit.map { "\($0.shortSHA) \($0.title)" } ?? record.sourceLabel
        return L("\(record.changedFileCount) 個のファイル・\(commit)", "\(record.changedFileCount) files · \(commit)")
    }
}

// MARK: - 後片付け

private struct UpdateMaintenanceSection: View {
    let updater: AppUpdateController

    var body: some View {
        Section {
            LabeledContent(L("ダウンロードしたファイル", "Downloaded files"),
                           value: ByteCountFormatter.string(fromByteCount: updater.cacheSize, countStyle: .file))
            Button(L("ダウンロードしたファイルを消す", "Delete downloaded files"), role: .destructive) { updater.clearCache() }
                .disabled(updater.isWorking || updater.cacheSize == 0)
        } footer: {
            Text(L("確認のときにダウンロードした、変わったファイルです。同じ中身は次から取り直しません。",
                   "Changed files downloaded during checks. The same content is not downloaded again."))
        }
    }
}
