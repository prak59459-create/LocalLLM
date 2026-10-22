import Foundation
import Observation

/// 取り込みの結果。
struct KnowledgeImportReport: Sendable {
    var added: [String] = []
    var skipped: [String] = []
}

/// マイアシスタントごとの参考資料（検索用インデックス）を保存・検索する。
/// 保存先: Application Support/Knowledge/<アシスタントの ID>.json
@MainActor
@Observable
final class KnowledgeStore {
    /// 変更のたびに増える番号（画面の更新用）。キャッシュ自体は監視対象にしない（表示中に書き換えないため）。
    private(set) var revision = 0
    private(set) var importingAssistantIDs: Set<UUID> = []
    @ObservationIgnored private var indexes: [UUID: KnowledgeIndex] = [:]

    @ObservationIgnored let rootDirectory: URL

    /// 取り込むファイルの拡張子（テキストとして読めるもの）
    nonisolated static let textExtensions: Set<String> = [
        "swift", "m", "mm", "h", "c", "cc", "cpp", "hpp", "py", "js", "mjs", "ts", "tsx", "jsx", "java", "kt", "kts",
        "go", "rs", "rb", "php", "cs", "dart", "lua", "r", "scala", "sh", "zsh", "bash", "sql",
        "html", "htm", "css", "scss", "xml", "json", "yaml", "yml", "toml", "ini", "plist", "strings",
        "md", "markdown", "txt", "text", "csv", "tsv", "rst", "tex", "log",
    ]
    /// 飛ばすフォルダ
    nonisolated static let skippedDirectoryNames: Set<String> = [".build", ".git", "node_modules", "DerivedData", "Pods", "build", ".swiftpm"]
    nonisolated static let maxFileBytes = 1_000_000
    nonisolated static let maxFileCount = 2_000

    init() {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        rootDirectory = base.appendingPathComponent("Knowledge", isDirectory: true)
        try? FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
    }

    /// インデックスを返す（未読み込みならファイルから読む）。
    func index(for assistantID: UUID) -> KnowledgeIndex {
        _ = revision   // 画面がこの値の変化を追えるように読んでおく
        if let cached = indexes[assistantID] { return cached }
        let url = fileURL(for: assistantID)
        let loaded = (try? Data(contentsOf: url)).flatMap { try? JSONDecoder().decode(KnowledgeIndex.self, from: $0) }
            ?? KnowledgeIndex()
        indexes[assistantID] = loaded
        return loaded
    }

    func search(_ query: String, in assistantID: UUID, limit: Int) -> [KnowledgeSearchResult] {
        index(for: assistantID).search(query, limit: limit)
    }

    /// ファイルやフォルダ（fileImporter で選んだ URL）を取り込む。
    func importItems(_ urls: [URL], into assistantID: UUID) async -> KnowledgeImportReport {
        importingAssistantIDs.insert(assistantID)
        defer { importingAssistantIDs.remove(assistantID) }
        let current = index(for: assistantID)
        // ファイルの読み込みとインデックス作成は時間がかかるので、メインスレッドの外で行う
        let result = await Task.detached(priority: .userInitiated) { () -> (KnowledgeIndex, KnowledgeImportReport) in
            var index = current
            var report = KnowledgeImportReport()
            for url in urls {
                let files = KnowledgeStore.collectTextFiles(at: url)
                report.skipped.append(contentsOf: files.skipped)
                for (source, text) in files.files {
                    index.add(source: source, text: text)
                    report.added.append(source)
                }
            }
            return (index, report)
        }.value
        indexes[assistantID] = result.0
        save(assistantID)
        revision += 1
        return result.1
    }

    func remove(source: String, from assistantID: UUID) {
        var index = index(for: assistantID)
        index.remove(source: source)
        indexes[assistantID] = index
        save(assistantID)
        revision += 1
    }

    func deleteAll(for assistantID: UUID) {
        indexes[assistantID] = KnowledgeIndex()
        try? FileManager.default.removeItem(at: fileURL(for: assistantID))
        revision += 1
    }

    // MARK: - 補助

    private func fileURL(for assistantID: UUID) -> URL {
        rootDirectory.appendingPathComponent("\(assistantID.uuidString).json")
    }

    private func save(_ assistantID: UUID) {
        guard let index = indexes[assistantID], let data = try? JSONEncoder().encode(index) else { return }
        try? data.write(to: fileURL(for: assistantID), options: .atomic)
    }

    /// URL がファイルならそれを、フォルダなら中のテキストファイルを集める。
    nonisolated static func collectTextFiles(at url: URL) -> (files: [(String, String)], skipped: [String]) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        var files: [(String, String)] = []
        var skipped: [String] = []
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return ([], [url.lastPathComponent])
        }
        if !isDirectory.boolValue {
            if let text = readText(url) {
                files.append((url.lastPathComponent, text))
            } else {
                skipped.append(url.lastPathComponent)
            }
            return (files, skipped)
        }

        let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys,
                                                              options: [.skipsHiddenFiles]) else {
            return ([], [url.lastPathComponent])
        }
        let basePath = url.standardizedFileURL.path
        let folderName = url.lastPathComponent
        while let item = enumerator.nextObject() as? URL {
            let values = try? item.resourceValues(forKeys: Set(keys))
            if values?.isDirectory == true {
                if skippedDirectoryNames.contains(item.lastPathComponent) { enumerator.skipDescendants() }
                continue
            }
            var relative = item.standardizedFileURL.path
            if relative.hasPrefix(basePath) { relative = String(relative.dropFirst(basePath.count)) }
            let source = folderName + (relative.hasPrefix("/") ? relative : "/" + relative)
            guard textExtensions.contains(item.pathExtension.lowercased()) else { continue }
            if let size = values?.fileSize, size > maxFileBytes {
                skipped.append(source)
                continue
            }
            if files.count >= maxFileCount {
                skipped.append(source)
                continue
            }
            if let text = readText(item) {
                files.append((source, text))
            } else {
                skipped.append(source)
            }
        }
        return (files, skipped)
    }

    /// UTF-8 → Shift_JIS の順で読む。
    nonisolated private static func readText(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url), data.count <= maxFileBytes else { return nil }
        if let text = String(data: data, encoding: .utf8) { return text }
        if let text = String(data: data, encoding: .shiftJIS) { return text }
        return nil
    }
}
