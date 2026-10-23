import Foundation

/// 端末に保存したモデル 1 つ分の記録。モデルフォルダ内の manifest.json に保存する。
struct InstalledModel: Codable, Identifiable, Hashable, Sendable {
    /// "owner/name::variantPath" 形式の ID
    var id: String
    var repoID: String
    var revision: String
    var variantPath: String
    var kind: ModelVariant.Kind
    var displayName: String
    /// Models/ の下のフォルダ名
    var directoryName: String
    /// ダウンロードするファイル（リポジトリ内のパスとサイズ）
    var files: [HFFileEntry]
    var totalBytes: Int64
    var createdAt: Date
    /// すべてのファイルがそろったか
    var isComplete: Bool
    /// コンパイル済みモデル（.mlmodelc）のフォルダ内パス。.mlpackage は初回読み込み時にコンパイルして設定する。
    var compiledModelPath: String?
    /// ダウンロード前に metadata.json から分かった情報（分かれば）
    var maxContext: Int?
    var stateBytes: Int64?
    var weightBytes: Int64?
    /// 分割モデル（kind == .pipeline）の部品（リポジトリ内のパス）
    var pipeline: PipelineDescriptor?
    /// ダウンロード先で名前を変えるフォルダ（拡張子の無いモデルのフォルダなど）
    var renames: [String: String]?

    /// 端末上でのパス（リポジトリ内のパスから）
    func localPath(for repoPath: String) -> String {
        ModelVariant.localPath(for: repoPath, renames: renames ?? [:])
    }

    var kindLabel: String {
        switch kind {
        case .compiled: return ".mlmodelc"
        case .package: return ".mlpackage"
        case .pipeline: return pipeline?.isMonolithic == true ? L("分割形式（一体型）", "Split format (monolithic)")
                                                              : L("分割モデル（\(pipeline?.chunks.count ?? 0) チャンク）", "Split (\(pipeline?.chunks.count ?? 0) chunks)")
        case .archive: return "ZIP"
        }
    }

    static func makeID(repoID: String, variantPath: String) -> String {
        "\(repoID)::\(variantPath)"
    }

    static func makeDirectoryName(repoID: String, variantPath: String) -> String {
        let raw = "\(repoID)__\(variantPath)"
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        return String(raw.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" })
    }

    /// tokenizer.json などの付属ファイルのパス（名前で探す。複数あれば階層の浅いもの）
    func supportFilePath(named name: String) -> String? {
        files.map(\.path)
            .filter { ($0 as NSString).lastPathComponent == name }
            .sorted { $0.split(separator: "/").count < $1.split(separator: "/").count }
            .first
    }

    var shortRepoName: String {
        repoID.split(separator: "/").last.map(String.init) ?? repoID
    }
}
