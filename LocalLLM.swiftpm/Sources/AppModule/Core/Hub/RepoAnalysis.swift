import Foundation

/// リポジトリ内のモデルの候補。
struct ModelVariant: Codable, Hashable, Identifiable, Sendable {
    enum Kind: String, Codable, Sendable {
        case compiled   // .mlmodelc（拡張子の無いコンパイル済みフォルダも含む）
        case package    // .mlpackage
        case pipeline   // いくつかのモデルをつないで動かす分割モデル（ANEMLL 形式）
        case archive    // ZIP に入ったモデル（ダウンロード後に展開する）
    }

    var id: String { path }
    /// モデルのフォルダ（分割モデルは meta.yaml などの代表のパス、ZIP はファイルのパス）
    var path: String
    var kind: Kind
    var files: [HFFileEntry]
    var totalBytes: Int64
    var displayName: String
    /// 分割モデルの部品
    var pipeline: PipelineDescriptor? = nil
    /// 端末に保存するときに名前を変えるフォルダ（リポジトリ内 → 端末）。拡張子の無いモデルのフォルダに .mlmodelc を付ける。
    var renames: [String: String]? = nil

    /// 端末上でのパス
    func localPath(for repoPath: String) -> String {
        ModelVariant.localPath(for: repoPath, renames: renames ?? [:])
    }

    static func localPath(for repoPath: String, renames: [String: String]) -> String {
        for (from, to) in renames {
            if from.isEmpty {
                // リポジトリ直下に置かれたコンパイル済みモデル
                if RepoAnalysis.isLooseModelFile(repoPath) { return to + "/" + repoPath }
            } else if repoPath == from {
                return to
            } else if repoPath.hasPrefix(from + "/") {
                return to + String(repoPath.dropFirst(from.count))
            }
        }
        return repoPath
    }
}

struct RepoAnalysis: Hashable, Sendable {
    var variants: [ModelVariant]
    var supportFiles: [HFFileEntry]
    var hasTokenizer: Bool

    /// ダウンロードする付属ファイルの名前
    static let supportFileNames: Set<String> = Set([
        "tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json",
        "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "added_tokens.json",
    ]).union(TokenizerSynthesizer.alternativeFileNames)

    /// tokenizer.json がそのまま入っているか（無くても、ほかの形式から作れることがある）
    var hasTokenizerJSON: Bool { supportFiles.contains { ($0.path as NSString).lastPathComponent == "tokenizer.json" } }

    /// コンパイル済みモデルの直下にあるファイル・フォルダ
    static let compiledModelEntries: Set<String> = [
        "coremldata.bin", "model.mil", "metadata.json", "analytics", "weights", "model",
        "neural_network_optionals", "model.espresso.net", "model.espresso.shape", "model.espresso.weights",
    ]

    static func isLooseModelFile(_ path: String) -> Bool {
        let first = path.split(separator: "/").first.map(String.init) ?? path
        return compiledModelEntries.contains(first)
    }

    /// - Parameters:
    ///   - files: リポジトリのファイル一覧
    ///   - metaFiles: ANEMLL の meta.yaml（パス → 中身）。分割モデルの組み立てに使う。
    ///   - repoName: リポジトリ名（直下に置かれたモデルの名前に使う）
    static func analyze(files: [HFFileEntry], metaFiles: [String: String] = [:], repoName: String = "model") -> RepoAnalysis {
        // 1. 拡張子で分かるモデルのフォルダ
        var groups: [String: [HFFileEntry]] = [:]
        var kinds: [String: ModelVariant.Kind] = [:]
        var renames: [String: String] = [:]
        var loose: [HFFileEntry] = []
        for file in files {
            if let root = modelRoot(of: file.path) {
                groups[root, default: []].append(file)
                kinds[root] = root.hasSuffix(".mlmodelc") ? .compiled : .package
            } else {
                loose.append(file)
            }
        }
        // 2. 拡張子は無いが、中身からモデルと分かるフォルダ（coremldata.bin + model.mil、Manifest.json + Data/）
        for (root, kind) in contentRoots(loose) {
            let members = loose.filter { file in
                root.isEmpty ? isLooseModelFile(file.path) : file.path.hasPrefix(root + "/")
            }
            guard !members.isEmpty else { continue }
            groups[root] = members
            kinds[root] = kind
            let name = root.isEmpty ? repoName : (root as NSString).lastPathComponent
            let suffix = kind == .compiled ? ".mlmodelc" : ".mlpackage"
            let parent = root.isEmpty ? "" : (root as NSString).deletingLastPathComponent
            renames[root] = (parent.isEmpty ? "" : parent + "/") + name + suffix
        }
        let rootPaths = Array(groups.keys)

        // 3. 分割モデル（meta.yaml、無ければファイル名から推測）
        var pipelines: [ModelVariant] = []
        var usedRoots = Set<String>()
        var metaByFolder: [String: String] = [:]
        for path in metaFiles.keys { metaByFolder[(path as NSString).deletingLastPathComponent] = path }
        var folders = Set(rootPaths.map { ($0 as NSString).deletingLastPathComponent })
        folders.formUnion(metaByFolder.keys)
        for folder in folders.sorted() {
            var descriptor: PipelineDescriptor?
            var metaPath: String?
            if let path = metaByFolder[folder], let text = metaFiles[path] {
                descriptor = PipelineDescriptor.fromANEMLLMeta(SimpleYAML.parse(text), folder: folder, availableRoots: rootPaths)
                metaPath = path
            }
            if descriptor == nil {
                descriptor = PipelineDescriptor.guess(folder: folder, roots: rootPaths)
            }
            guard let descriptor else { continue }
            var members = descriptor.parts.flatMap { groups[$0] ?? [] }
            if let metaPath, let meta = files.first(where: { $0.path == metaPath }) { members.append(meta) }
            let name = (folder.isEmpty ? repoName : (folder as NSString).lastPathComponent)
            let label = descriptor.isMonolithic ? L("一体型", "monolithic") : L("\(descriptor.chunks.count) チャンク", "\(descriptor.chunks.count) chunks")
            pipelines.append(ModelVariant(path: metaPath ?? (folder.isEmpty ? "[split]" : folder + "/[split]"),
                                          kind: .pipeline,
                                          files: members.sorted { $0.path < $1.path },
                                          totalBytes: members.reduce(Int64(0)) { $0 + $1.size },
                                          displayName: "\(name)（\(label)）",
                                          pipeline: descriptor,
                                          renames: renames.isEmpty ? nil : renames))
            usedRoots.formUnion(descriptor.parts)
        }

        // 4. ひとつで動くモデル（分割モデルのあるフォルダの、使わなかった部品は出さない）
        let pipelineFolders = Set(pipelines.flatMap { $0.pipeline?.parts ?? [] }.map { ($0 as NSString).deletingLastPathComponent })
        var singles: [ModelVariant] = []
        for (root, entries) in groups where !usedRoots.contains(root)
            && !pipelineFolders.contains((root as NSString).deletingLastPathComponent) {
            let kind = kinds[root] ?? .compiled
            let localRoot = renames[root] ?? root
            let name = root.isEmpty ? repoName : ((localRoot as NSString).lastPathComponent as NSString).deletingPathExtension
            singles.append(ModelVariant(path: root, kind: kind, files: entries.sorted { $0.path < $1.path },
                                        totalBytes: entries.reduce(Int64(0)) { $0 + $1.size }, displayName: name,
                                        renames: renames[root].map { [root: $0] }))
        }

        // 5. ZIP に入ったモデル
        let archives = loose.filter { isModelArchive($0.path) }.map { file in
            ModelVariant(path: file.path, kind: .archive, files: [file], totalBytes: file.size,
                         displayName: ((file.path as NSString).lastPathComponent as NSString).deletingPathExtension)
        }

        let order: [ModelVariant.Kind: Int] = [.pipeline: 0, .compiled: 1, .package: 2, .archive: 3]
        let variants = (pipelines + singles + archives).sorted { lhs, rhs in
            if lhs.kind != rhs.kind { return order[lhs.kind, default: 9] < order[rhs.kind, default: 9] }
            return lhs.path < rhs.path
        }
        let modelFiles = Set(groups.values.flatMap { $0.map(\.path) })
        let support = files.filter {
            supportFileNames.contains(($0.path as NSString).lastPathComponent) && !modelFiles.contains($0.path)
        }
        // tokenizer.json か、そこから作れるファイル（tokenizer.model、vocab.json + merges.txt など）があるか
        let hasTokenizer = TokenizerSynthesizer.canBuild(fileNames: Set(support.map { ($0.path as NSString).lastPathComponent }))
        return RepoAnalysis(variants: variants, supportFiles: support.sorted { $0.path < $1.path }, hasTokenizer: hasTokenizer)
    }

    /// パスの中で最初に出てくる .mlmodelc / .mlpackage のフォルダ（入れ子の内側は無視）
    static func modelRoot(of path: String) -> String? {
        let components = path.split(separator: "/").map(String.init)
        for (index, component) in components.enumerated() where index < components.count - 1 {
            if component.hasSuffix(".mlmodelc") || component.hasSuffix(".mlpackage") {
                return components[0...index].joined(separator: "/")
            }
        }
        return nil
    }

    /// 拡張子の無いモデルのフォルダ（"" はリポジトリ直下）
    static func contentRoots(_ files: [HFFileEntry]) -> [(String, ModelVariant.Kind)] {
        var byFolder: [String: Set<String>] = [:]
        for file in files {
            let folder = (file.path as NSString).deletingLastPathComponent
            byFolder[folder, default: []].insert((file.path as NSString).lastPathComponent)
        }
        var candidates: [(String, ModelVariant.Kind)] = []
        for (folder, names) in byFolder {
            if names.contains("coremldata.bin") && (names.contains("model.mil") || names.contains("model.espresso.net")) {
                candidates.append((folder, .compiled))
            } else if names.contains("Manifest.json"),
                      files.contains(where: { $0.path.hasPrefix((folder.isEmpty ? "" : folder + "/") + "Data/com.apple.CoreML/") }) {
                candidates.append((folder, .package))
            }
        }
        // 入れ子は外側だけ
        return candidates.filter { candidate in
            !candidates.contains { other in
                other.0 != candidate.0 && (other.0.isEmpty || candidate.0.hasPrefix(other.0 + "/"))
            }
        }.sorted { $0.0 < $1.0 }
    }

    static func isModelArchive(_ path: String) -> Bool {
        let lower = path.lowercased()
        guard lower.hasSuffix(".zip") else { return false }
        return lower.contains("mlmodelc") || lower.contains("mlpackage") || lower.contains("coreml") || lower.contains("model")
    }

    /// 選んだモデルのダウンロード対象（モデルのファイル + そのモデルに近い付属ファイル）
    func filesToDownload(for variant: ModelVariant) -> [HFFileEntry] {
        let variantParent = (variant.path as NSString).deletingLastPathComponent
        var chosen: [String: HFFileEntry] = [:]
        for file in supportFiles {
            let name = (file.path as NSString).lastPathComponent
            let candidateParent = (file.path as NSString).deletingLastPathComponent
            if let existing = chosen[name] {
                let existingParent = (existing.path as NSString).deletingLastPathComponent
                if Self.distance(from: variantParent, to: candidateParent) < Self.distance(from: variantParent, to: existingParent) {
                    chosen[name] = file
                }
            } else {
                chosen[name] = file
            }
        }
        return variant.files + chosen.values.sorted { $0.path < $1.path }
    }

    /// フォルダどうしの近さ（同じフォルダ 0、祖先は段数、それ以外は大きな値 + 深さ）
    private static func distance(from base: String, to folder: String) -> Int {
        if base == folder { return 0 }
        let baseParts = base.split(separator: "/")
        let folderParts = folder.split(separator: "/")
        if baseParts.starts(with: folderParts) { return baseParts.count - folderParts.count }
        return 100 + folderParts.count
    }

    /// meta.yaml のパス（分割モデルの説明。ダウンロード前に中身を取ってきて analyze に渡す）
    static func metaFilePaths(in files: [HFFileEntry]) -> [String] {
        files.map(\.path).filter { ($0 as NSString).lastPathComponent == "meta.yaml" && modelRoot(of: $0) == nil }
    }
}
