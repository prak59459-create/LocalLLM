import Foundation

/// ファイル 1 つの変わり方。
enum FileChangeKind: String, Codable, Sendable, CaseIterable {
    /// GitHub にあって、この iPad に無い
    case added
    /// GitHub で変わった
    case modified
    /// GitHub で消えた
    case removed
    /// この iPad でだけ書き換えた（GitHub は前回の更新から変わっていない）
    case localEdit
    /// この iPad でだけ消した（GitHub は前回の更新から変わっていない）
    case localDeleted

    /// 何も選ばなければ更新に含めるか
    var selectedByDefault: Bool {
        switch self {
        case .added, .modified, .removed: return true
        case .localEdit, .localDeleted: return false
        }
    }

    /// 更新すると GitHub から取ってきたファイルを書くか（removed は消す）
    var writesRemoteFile: Bool { self != .removed }
}

/// 変わったファイル 1 つ。
struct UpdateFileChange: Codable, Hashable, Sendable, Identifiable {
    var path: String
    var kind: FileChangeKind
    /// この iPad のファイルの識別子（無ければ nil）
    var localSHA: String?
    /// GitHub のファイルの識別子（消えたなら nil）
    var remoteSHA: String?
    var remoteSize: Int64?
    /// この iPad でも前回の更新から書き換えている（GitHub の変更と重なっている）
    var editedHere: Bool = false

    var id: String { path }
    var isPackageManifest: Bool { path == "Package.swift" }
    var isSwiftSource: Bool { path.hasSuffix(".swift") }
}

/// GitHub（または選んだファイル）と、この iPad のパッケージの違い。
struct UpdateDiff: Codable, Sendable {
    var changes: [UpdateFileChange]
    /// 同じだったファイルの数
    var unchangedCount: Int
    /// この iPad にだけあるファイル（GitHub に無く、消さないもの）
    var keptLocalFiles: [String]
    /// Package.swift は版や、この iPad の設定（チームなど）だけが違う
    var manifestDiffersOnlyInSettings: Bool

    var isUpToDate: Bool { changes.allSatisfy { !$0.kind.selectedByDefault } }
    var defaultSelection: Set<String> { Set(changes.filter(\.kind.selectedByDefault).map(\.path)) }

    func changes(of kind: FileChangeKind) -> [UpdateFileChange] { changes.filter { $0.kind == kind } }
}

enum UpdateDiffer {
    /// 新しいソースに無ければ消す範囲（これ以外のファイルは消さない）
    static let managedFolders = ["Sources/"]

    /// local（この iPad）と remote（新しいソース）を比べる。
    /// base は前回の更新で書いた内容（相対パス → 識別子）。あれば、この iPad でだけ変えたファイルを見分けられる。
    /// manifestEquivalent は、Package.swift の識別子が違うときに、版やこの iPad の設定以外は同じだったか（呼ぶ側が中身を見て決める）。
    static func diff(local: [String: PackageFile], remote: [String: PackageFile], base: [String: String]?,
                     manifestEquivalent: Bool = false) -> UpdateDiff {
        var changes: [UpdateFileChange] = []
        var unchanged = 0
        var kept: [String] = []
        var manifestOnlySettings = false

        for (path, remoteFile) in remote {
            let baseSHA = base?[path]
            guard let localFile = local[path] else {
                // この iPad に無い
                let deletedHere = baseSHA != nil && baseSHA == remoteFile.sha
                changes.append(UpdateFileChange(path: path, kind: deletedHere ? .localDeleted : .added, localSHA: nil,
                                          remoteSHA: remoteFile.sha, remoteSize: remoteFile.size))
                continue
            }
            if localFile.sha == remoteFile.sha {
                unchanged += 1
                continue
            }
            if path == "Package.swift" {
                if manifestEquivalent {
                    unchanged += 1
                    manifestOnlySettings = true
                } else {
                    changes.append(UpdateFileChange(path: path, kind: .modified, localSHA: localFile.sha,
                                              remoteSHA: remoteFile.sha, remoteSize: remoteFile.size))
                }
                continue
            }
            let kind: FileChangeKind
            var editedHere = false
            if let baseSHA, baseSHA == remoteFile.sha {
                kind = .localEdit
            } else {
                kind = .modified
                editedHere = baseSHA != nil && baseSHA != localFile.sha
            }
            changes.append(UpdateFileChange(path: path, kind: kind, localSHA: localFile.sha, remoteSHA: remoteFile.sha,
                                      remoteSize: remoteFile.size, editedHere: editedHere))
        }

        for (path, localFile) in local where remote[path] == nil {
            guard managedFolders.contains(where: { path.hasPrefix($0) }) else { continue }
            if let base {
                guard let baseSHA = base[path] else {
                    // 前回の更新の後に、この iPad で足したファイル
                    kept.append(path)
                    continue
                }
                changes.append(UpdateFileChange(path: path, kind: .removed, localSHA: localFile.sha, remoteSHA: nil,
                                          remoteSize: nil, editedHere: baseSHA != localFile.sha))
            } else {
                changes.append(UpdateFileChange(path: path, kind: .removed, localSHA: localFile.sha, remoteSHA: nil, remoteSize: nil))
            }
        }

        return UpdateDiff(changes: changes.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending },
                          unchangedCount: unchanged, keptLocalFiles: kept.sorted(),
                          manifestDiffersOnlyInSettings: manifestOnlySettings)
    }

    /// 更新したあとの「前回の更新で書いた内容」。applied は更新に含めたファイル。
    /// 含めなかった GitHub の変更は、前回の内容のままにしておく（次の確認でもまた「変更」として出る）。
    static func manifest(after diff: UpdateDiff, remote: [String: PackageFile], base: [String: String]?,
                         applied: Set<String>) -> [String: String] {
        var result = base ?? [:]
        let changed = Dictionary(uniqueKeysWithValues: diff.changes.map { ($0.path, $0) })
        for (path, file) in remote {
            guard let change = changed[path] else {
                // 同じだった（Package.swift は設定だけ違っても同じとみなす）
                result[path] = file.sha
                continue
            }
            if applied.contains(path) || change.kind == .localEdit || change.kind == .localDeleted {
                // 更新した、または GitHub は前回から変わっていない
                result[path] = file.sha
            }
        }
        // 新しいソースに無いものは忘れる
        for path in result.keys where remote[path] == nil {
            result[path] = nil
        }
        // 消すのを見送ったファイルは、前回の内容（無ければ今の内容）を残す。次の確認でもまた「削除」として出る。
        for change in diff.changes where change.kind == .removed && !applied.contains(change.path) {
            result[change.path] = base?[change.path] ?? change.localSHA
        }
        return result
    }
}
