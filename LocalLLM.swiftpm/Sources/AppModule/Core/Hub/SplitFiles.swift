import Foundation

/// 大きすぎて何個かに分けて置かれたファイルを見分けて、ダウンロードしたあとに 1 つにつなげる。
///
/// 見分ける名前:
/// - `weight.bin.part1`、`weight.bin.part01`、`weight.bin.part-1`、`weight.bin.part1of3`
/// - `model.zip.001`（3 桁の番号）
/// - `weight-00001-of-00003.bin`（.bin / .zip / .mil だけ。safetensors の分割は別物なので対象外）
/// - `model.zip.partaa`（split コマンドの名前）
enum SplitParts {
    struct Part: Hashable, Sendable {
        /// つなげたあとのパス
        var base: String
        /// 何番目か（0 から）
        var index: Int
        /// 全部でいくつか（名前から分かるとき）
        var total: Int?
    }

    private static let patterns: [(NSRegularExpression, (NSTextCheckingResult, NSString) -> Part?)] = {
        func regex(_ pattern: String) -> NSRegularExpression {
            // パターンは固定なので失敗しない
            try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        }
        func group(_ match: NSTextCheckingResult, _ text: NSString, _ index: Int) -> String? {
            let range = match.range(at: index)
            return range.location == NSNotFound ? nil : text.substring(with: range)
        }
        return [
            (regex(#"^(.+\.[A-Za-z0-9]+)\.part-?(\d+)(?:of(\d+))?$"#), { match, text in
                guard let base = group(match, text, 1), let number = group(match, text, 2).flatMap({ Int($0) }) else { return nil }
                return Part(base: base, index: number, total: group(match, text, 3).flatMap { Int($0) })
            }),
            (regex(#"^(.+\.[A-Za-z0-9]+)\.(\d{3})$"#), { match, text in
                guard let base = group(match, text, 1), let number = group(match, text, 2).flatMap({ Int($0) }) else { return nil }
                return Part(base: base, index: number, total: nil)
            }),
            (regex(#"^(.+)-(\d{3,5})-of-(\d{3,5})(\.(?:bin|zip|mil))$"#), { match, text in
                guard let stem = group(match, text, 1), let number = group(match, text, 2).flatMap({ Int($0) }),
                      let total = group(match, text, 3).flatMap({ Int($0) }), let suffix = group(match, text, 4) else { return nil }
                return Part(base: stem + suffix, index: number, total: total)
            }),
            (regex(#"^(.+\.[A-Za-z0-9]+)\.part([a-z])([a-z])$"#), { match, text in
                guard let base = group(match, text, 1), let first = group(match, text, 2)?.unicodeScalars.first,
                      let second = group(match, text, 3)?.unicodeScalars.first else { return nil }
                let index = Int(first.value - 97) * 26 + Int(second.value - 97)
                return Part(base: base, index: index, total: nil)
            }),
        ]
    }()

    /// パスが分割の 1 つなら、その情報
    static func parse(_ path: String) -> Part? {
        let text = path as NSString
        let range = NSRange(location: 0, length: text.length)
        for (pattern, make) in patterns {
            if let match = pattern.firstMatch(in: path, range: range), let part = make(match, text) { return part }
        }
        return nil
    }

    /// 分割をつなげたあとのパスごとにまとめる（番号がそろっているものだけ。2 つ以上あるものだけ）
    static func group(_ paths: [String]) -> [String: [String]] {
        var byBase: [String: [(Part, String)]] = [:]
        for path in paths {
            guard let part = parse(path) else { continue }
            byBase[part.base, default: []].append((part, path))
        }
        var result: [String: [String]] = [:]
        for (base, parts) in byBase {
            let sorted = parts.sorted { $0.0.index < $1.0.index }
            let indices = sorted.map(\.0.index)
            guard sorted.count >= 2, let first = indices.first, first <= 1,
                  indices == Array(first..<(first + indices.count)) else { continue }
            if let total = sorted.first?.0.total, total != sorted.count { continue }
            // 元のファイルそのものも置かれていたら、つなげない
            guard !paths.contains(base) else { continue }
            result[base] = sorted.map(\.1)
        }
        return result
    }

    /// ファイルの一覧を、分割をつなげたあとの一覧にする（大きさは合計）
    static func joinedEntries(_ files: [HFFileEntry]) -> (files: [HFFileEntry], groups: [String: [String]]) {
        let groups = group(files.map(\.path))
        guard !groups.isEmpty else { return (files, [:]) }
        let partPaths = Set(groups.values.flatMap { $0 })
        var sizes: [String: Int64] = [:]
        for file in files where partPaths.contains(file.path) {
            if let base = parse(file.path)?.base { sizes[base, default: 0] += file.size }
        }
        let rest = files.filter { !partPaths.contains($0.path) }
        let joined = groups.keys.sorted().map { HFFileEntry(path: $0, size: sizes[$0] ?? 0, lfsSHA256: nil) }
        return (rest + joined, groups)
    }

    /// 部品を順につなげて destination に書き、部品は消す
    static func join(_ parts: [URL], into destination: URL) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".joining-\(UUID().uuidString)")
        fileManager.createFile(atPath: temporary.path, contents: nil)
        let output = try FileHandle(forWritingTo: temporary)
        do {
            for part in parts {
                let input = try FileHandle(forReadingFrom: part)
                defer { try? input.close() }
                // 大きなファイルでもメモリを使わないよう、少しずつ写す
                while true {
                    let chunk = try input.read(upToCount: 8 << 20) ?? Data()
                    if chunk.isEmpty { break }
                    try output.write(contentsOf: chunk)
                }
            }
            try output.close()
        } catch {
            try? output.close()
            try? fileManager.removeItem(at: temporary)
            throw error
        }
        try? fileManager.removeItem(at: destination)
        try fileManager.moveItem(at: temporary, to: destination)
        for part in parts { try? fileManager.removeItem(at: part) }
    }
}

/// ZIP に入ったモデル（X.mlmodelc.zip など）を、その場で展開する。
enum ModelArchiveExpander {
    /// ZIP の名前から、展開したあとのモデルのフォルダの名前（"a/X.mlmodelc.zip" → "a/X.mlmodelc"）
    static func modelRoot(forArchive path: String) -> String? {
        guard path.lowercased().hasSuffix(".zip") else { return nil }
        let base = String(path.dropLast(4))
        let lower = base.lowercased()
        return lower.hasSuffix(".mlmodelc") || lower.hasSuffix(".mlpackage") ? base : nil
    }

    /// zip を展開して、中身が `root`（folder からの相対パス）のフォルダになるようにする。ZIP は消す。
    /// ZIP の中が「X.mlmodelc/…」でも「coremldata.bin …」（フォルダ無し）でもよい。
    static func expand(_ zip: URL, root: String, in folder: URL, inflate: ZipArchive.Inflater?) throws {
        let fileManager = FileManager.default
        let staging = folder.appendingPathComponent(".expand-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }
        _ = try ZipArchive(url: zip).extract(to: staging, inflate: inflate, skip: { path in
            path.hasPrefix("__MACOSX/") || (path as NSString).lastPathComponent == ".DS_Store"
        })
        let name = (root as NSString).lastPathComponent
        let destination = folder.appendingPathComponent(root, isDirectory: true)
        let top = (try? fileManager.contentsOfDirectory(atPath: staging.path))?.filter { !$0.hasPrefix(".") } ?? []
        let source: URL
        if top.contains(name) {
            source = staging.appendingPathComponent(name, isDirectory: true)
        } else if top.count == 1, let only = top.first,
                  (try? staging.appendingPathComponent(only).resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true,
                  only.lowercased().hasSuffix(".mlmodelc") || only.lowercased().hasSuffix(".mlpackage")
                    || fileManager.fileExists(atPath: staging.appendingPathComponent(only).appendingPathComponent("coremldata.bin").path) {
            source = staging.appendingPathComponent(only, isDirectory: true)
        } else {
            source = staging
        }
        try? fileManager.removeItem(at: destination)
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if source == staging {
            try fileManager.moveItem(at: staging, to: destination)
        } else {
            try fileManager.moveItem(at: source, to: destination)
        }
        try? fileManager.removeItem(at: zip)
    }
}
