import Foundation

/// 画像生成モデル（Core ML 版の Stable Diffusion）のリポジトリから、必要なファイルを選ぶ。
///
/// Apple の ml-stable-diffusion で変換したリポジトリは、
/// `split_einsum_v2/compiled/` や `original/compiled/` の下に次のものが入っている:
/// TextEncoder.mlmodelc・Unet.mlmodelc・VAEDecoder.mlmodelc・vocab.json・merges.txt
/// （ほかに安全フィルタ・VAE エンコーダもあるが、文章から画像を作るだけなら要らない）
struct ImageRepoLayout: Hashable, Sendable {
    /// モデルの入っているフォルダ（リポジトリ内。直下なら ""）
    var folder: String
    /// ダウンロードするファイル
    var files: [HFFileEntry]

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }

    static let models = ["TextEncoder.mlmodelc", "Unet.mlmodelc", "VAEDecoder.mlmodelc"]
    static let resources = ["vocab.json", "merges.txt"]
    /// 試す順（Neural Engine 向けの split_einsum を先に）
    static let preferredFolders = ["split_einsum_v2/compiled", "split_einsum/compiled", "original/compiled", "compiled", ""]

    /// リポジトリのファイル一覧から、使えるフォルダを探す（無ければ nil）
    static func find(in entries: [HFFileEntry]) -> ImageRepoLayout? {
        var folders = preferredFolders
        // 決まった名前以外のフォルダに入っていることもある
        for entry in entries where entry.path.hasSuffix("Unet.mlmodelc/coremldata.bin") {
            let folder = String(entry.path.dropLast("Unet.mlmodelc/coremldata.bin".count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            if !folders.contains(folder) { folders.append(folder) }
        }
        for folder in folders {
            let prefix = folder.isEmpty ? "" : folder + "/"
            var chosen: [HFFileEntry] = []
            var found = Set<String>()
            for entry in entries where entry.path.hasPrefix(prefix) {
                let rest = String(entry.path.dropFirst(prefix.count))
                let first = rest.split(separator: "/").first.map(String.init) ?? rest
                if models.contains(first), rest.contains("/") {
                    chosen.append(entry)
                    found.insert(first)
                } else if resources.contains(rest) {
                    chosen.append(entry)
                    found.insert(rest)
                }
            }
            if found.count == models.count + resources.count {
                return ImageRepoLayout(folder: folder, files: chosen.sorted { $0.path < $1.path })
            }
        }
        return nil
    }

    /// 端末に保存するときのパス（フォルダの部分を取り除く）
    func localPath(for entry: HFFileEntry) -> String {
        folder.isEmpty ? entry.path : String(entry.path.dropFirst(folder.count + 1))
    }
}
