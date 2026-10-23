import Foundation
#if canImport(Compression)
import Compression
#endif

/// とても小さな YAML の読み取り（ANEMLL の meta.yaml 用）。
/// 「キー: 値」の入れ子、インデントした「- 値」の並び、[a, b] 形式の並び、| の複数行（読み飛ばす）だけに対応する。
indirect enum SimpleYAML: Hashable, Sendable {
    case scalar(String)
    case list([SimpleYAML])
    case map([String: SimpleYAML])

    static func parse(_ text: String) -> SimpleYAML {
        var lines: [(indent: Int, text: String)] = []
        var blockIndent: Int?
        for raw in text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n") {
            let indent = raw.prefix { $0 == " " }.count
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if let block = blockIndent {
                if trimmed.isEmpty || indent > block { continue }
                blockIndent = nil
            }
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            lines.append((indent, trimmed))
            if trimmed.hasSuffix("|") || trimmed.hasSuffix(">") || trimmed.hasSuffix("|-") || trimmed.hasSuffix(">-") {
                blockIndent = indent
            }
        }
        var index = 0
        return parseBlock(lines, &index, indent: lines.first?.indent ?? 0)
    }

    private static func parseBlock(_ lines: [(indent: Int, text: String)], _ index: inout Int, indent: Int) -> SimpleYAML {
        guard index < lines.count else { return .map([:]) }
        if lines[index].text.hasPrefix("- ") || lines[index].text == "-" {
            var items: [SimpleYAML] = []
            while index < lines.count, lines[index].indent == indent,
                  lines[index].text.hasPrefix("- ") || lines[index].text == "-" {
                let value = String(lines[index].text.dropFirst(1)).trimmingCharacters(in: .whitespaces)
                index += 1
                items.append(scalarOrInline(value))
            }
            return .list(items)
        }
        var map: [String: SimpleYAML] = [:]
        while index < lines.count, lines[index].indent == indent {
            let line = lines[index].text
            index += 1
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = unquote(String(line[..<colon]).trimmingCharacters(in: .whitespaces))
            let rest = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            if rest.isEmpty {
                // 子の要素（次の行がより深い、または同じ深さの "- "）
                if index < lines.count, lines[index].indent > indent
                    || (lines[index].indent == indent && lines[index].text.hasPrefix("- ")) {
                    map[key] = parseBlock(lines, &index, indent: lines[index].indent)
                } else {
                    map[key] = .scalar("")
                }
            } else if rest == "|" || rest == ">" || rest == "|-" || rest == ">-" {
                map[key] = .scalar("")
            } else {
                map[key] = scalarOrInline(stripComment(rest))
            }
        }
        return .map(map)
    }

    private static func scalarOrInline(_ value: String) -> SimpleYAML {
        if value.hasPrefix("[") && value.hasSuffix("]") {
            let inner = value.dropFirst().dropLast()
            let parts = inner.split(separator: ",").map { unquote($0.trimmingCharacters(in: .whitespaces)) }
            return .list(parts.filter { !$0.isEmpty }.map { .scalar($0) })
        }
        return .scalar(unquote(value))
    }

    private static func stripComment(_ value: String) -> String {
        if let quote = value.first, quote == "\"" || quote == "'" {
            // 引用符の中の # はコメントではない
            if let close = value.dropFirst().firstIndex(of: quote) { return String(value[...close]) }
            return value
        }
        guard let hash = value.range(of: " #") else { return value }
        return String(value[..<hash.lowerBound]).trimmingCharacters(in: .whitespaces)
    }

    private static func unquote(_ value: String) -> String {
        if value.count >= 2, let first = value.first, let last = value.last,
           (first == "\"" && last == "\"") || (first == "'" && last == "'") {
            return String(value.dropFirst().dropLast())
        }
        return value
    }

    subscript(key: String) -> SimpleYAML? {
        if case .map(let map) = self { return map[key] }
        return nil
    }

    var string: String? {
        if case .scalar(let value) = self { return value }
        return nil
    }

    var int: Int? { string.flatMap { Int($0) } }

    var bool: Bool? {
        switch string?.lowercased() {
        case "true", "yes": return true
        case "false", "no": return false
        default: return nil
        }
    }

    var list: [SimpleYAML]? {
        if case .list(let items) = self { return items }
        return nil
    }
}

/// いくつかの Core ML モデルを順番につないで動かす LLM（ANEMLL 形式）の説明。
/// - 分割型: 埋め込み（input_ids → hidden_states）→ 本体のチャンク（infer / prefill 関数、KV キャッシュを共有）→ 出力層（logits1…N）
/// - 一体型（monolithic）: 1 つのモデルが input_ids を受け取り logits1…N を出す（infer / prefill 関数）
struct PipelineDescriptor: Codable, Hashable, Sendable {
    /// リポジトリ内のパス（.mlmodelc / .mlpackage）
    var embeddings: String?
    var chunks: [String]
    var lmHead: String?
    var contextLength: Int?
    var batchSize: Int?
    var slidingWindow: Int?
    /// 出力層が logits ではなく、分割ごとの最大値の位置と値（argmax_idx / argmax_val）を出す
    var argmaxInModel: Bool
    var splitLMHead: Int?
    var vocabSize: Int?
    var lmHeadChunkSizes: [Int]?
    var architecture: String?

    var isMonolithic: Bool { embeddings == nil && lmHead == nil }

    /// すべての部品のパス
    var parts: [String] {
        [embeddings].compactMap { $0 } + chunks + [lmHead].compactMap { $0 }
    }

    /// argmax_idx の分割ごとの先頭位置
    func argmaxOffsets(count: Int) -> [Int] {
        if let sizes = lmHeadChunkSizes, sizes.count == count {
            var offsets: [Int] = []
            var total = 0
            for size in sizes {
                offsets.append(total)
                total += size
            }
            return offsets
        }
        let size = (vocabSize ?? 0) / max(count, 1)
        return (0..<count).map { $0 * size }
    }

    /// meta.yaml から作る。folder は meta.yaml のあるフォルダ（リポジトリのルートなら ""）。
    /// availableRoots はそのリポジトリにあるモデルのフォルダ（パス）。
    static func fromANEMLLMeta(_ yaml: SimpleYAML, folder: String, availableRoots: [String]) -> PipelineDescriptor? {
        guard let info = yaml["model_info"], let parameters = info["parameters"] else { return nil }
        func path(_ name: String?) -> String? {
            guard let name, !name.isEmpty else { return nil }
            let candidates = [name, name + ".mlmodelc", name + ".mlpackage",
                              (name as NSString).deletingPathExtension + ".mlmodelc",
                              (name as NSString).deletingPathExtension + ".mlpackage"]
            for candidate in candidates {
                let full = folder.isEmpty ? candidate : folder + "/" + candidate
                if availableRoots.contains(full) { return full }
            }
            return nil
        }
        let isMonolithic = info["model_type"]?.string == "monolithic" || parameters["monolithic_model"] != nil
        var descriptor = PipelineDescriptor(
            embeddings: nil, chunks: [], lmHead: nil,
            contextLength: parameters["context_length"]?.int,
            batchSize: parameters["batch_size"]?.int,
            slidingWindow: parameters["sliding_window"]?.int,
            argmaxInModel: parameters["argmax_in_model"]?.bool ?? false,
            splitLMHead: parameters["split_lm_head"]?.int ?? parameters["num_logits"]?.int,
            vocabSize: parameters["vocab_size"]?.int,
            lmHeadChunkSizes: parameters["lm_head_chunk_sizes"]?.list?.compactMap(\.int),
            architecture: info["architecture"]?.string
        )
        if isMonolithic {
            guard let model = path(parameters["monolithic_model"]?.string) else { return nil }
            descriptor.chunks = [model]
            return descriptor
        }
        guard let embeddings = path(parameters["embeddings"]?.string),
              let lmHead = path(parameters["lm_head"]?.string) else { return nil }
        descriptor.embeddings = embeddings
        descriptor.lmHead = lmHead
        // ffn には 1 つ目のチャンク名（…_chunk_01of02）か、チャンク番号のない名前が入っている
        let count = max(parameters["num_chunks"]?.int ?? 1, 1)
        let ffnName = parameters["ffn"]?.string ?? ""
        let base = (ffnName as NSString).deletingPathExtension
        var chunks: [String] = []
        if count == 1, let single = path(ffnName) {
            chunks = [single]
        } else {
            let stem = base.replacingOccurrences(of: #"_chunk_\d+of\d+$"#, with: "", options: .regularExpression)
            for number in 1...count {
                let name = stem + String(format: "_chunk_%02dof%02d", number, count)
                guard let chunk = path(name) else { return nil }
                chunks.append(chunk)
            }
        }
        guard !chunks.isEmpty else { return nil }
        descriptor.chunks = chunks
        return descriptor
    }

    /// meta.yaml が無いとき、ファイル名から推測する（…embeddings…, …FFN…chunk_01of02…, …lm_head…）。
    static func guess(folder: String, roots: [String]) -> PipelineDescriptor? {
        let inFolder = roots.filter { ($0 as NSString).deletingLastPathComponent == folder }
        func name(_ path: String) -> String { ((path as NSString).lastPathComponent as NSString).deletingPathExtension.lowercased() }
        let embeddings = inFolder.filter { name($0).contains("embedding") }
        let heads = inFolder.filter { name($0).contains("lm_head") || name($0).contains("lmhead") }
        let ffn = inFolder.filter { name($0).contains("ffn") || name($0).contains("_chunk_") }
            .filter { !embeddings.contains($0) && !heads.contains($0) }
        guard let embedding = embeddings.sorted(by: { $0.count > $1.count }).first,
              let head = heads.sorted(by: { $0.count > $1.count }).first,
              !ffn.isEmpty else { return nil }
        // チャンク番号順（"chunk_01of02" の 01）。同じ番号が複数あれば名前の長い（量子化済み）ほう
        var byNumber: [Int: String] = [:]
        var total = 0
        for path in ffn {
            let lower = name(path)
            if let range = lower.range(of: #"chunk_(\d+)of(\d+)"#, options: .regularExpression) {
                let digits = lower[range].dropFirst("chunk_".count).split(separator: "o")
                let number = Int(digits.first ?? "") ?? 1
                total = max(total, Int(lower[range].split(separator: "f").last ?? "") ?? 0)
                if let existing = byNumber[number], existing.count >= path.count { continue }
                byNumber[number] = path
            } else if byNumber[1] == nil {
                byNumber[1] = path
            }
        }
        let numbers = byNumber.keys.sorted()
        guard !numbers.isEmpty, total == 0 || numbers == Array(1...total) else { return nil }
        return PipelineDescriptor(embeddings: embedding, chunks: numbers.compactMap { byNumber[$0] }, lmHead: head,
                                  contextLength: nil, batchSize: nil, slidingWindow: nil, argmaxInModel: false,
                                  splitLMHead: nil, vocabSize: nil, lmHeadChunkSizes: nil, architecture: nil)
    }
}

/// ZIP ファイルの読み取り（中央ディレクトリを読んで、1 ファイルずつ取り出す）。
/// 圧縮方式は「無圧縮（0）」と「Deflate（8）」。Deflate の展開は inflate に渡す（iPad では Compression フレームワーク）。
struct ZipArchive {
    struct Entry: Hashable, Sendable {
        var path: String
        var method: UInt16
        var compressedSize: Int
        var uncompressedSize: Int
        var crc32: UInt32
        var localHeaderOffset: Int
        var isDirectory: Bool { path.hasSuffix("/") }
    }

    enum ZipError: Error, LocalizedError {
        case notZip
        case corrupt(String)
        case unsupportedMethod(UInt16)
        case checksumMismatch(String)
        case unsafePath(String)

        var errorDescription: String? {
            switch self {
            case .notZip: return L("ZIP ファイルではありません", "Not a ZIP file")
            case .corrupt(let detail): return L("ZIP ファイルが壊れています（\(detail)）", "The ZIP file is corrupt (\(detail))")
            case .unsupportedMethod(let method): return L("未対応の圧縮方式です（\(method)）", "Unsupported compression method (\(method))")
            case .checksumMismatch(let path): return L("\(path) の CRC が一致しません", "CRC mismatch for \(path)")
            case .unsafePath(let path): return L("安全でないパスが含まれています: \(path)", "Unsafe path in archive: \(path)")
            }
        }
    }

    /// 生の Deflate を展開する関数（data, 展開後のサイズ）→ 展開したデータ
    typealias Inflater = (Data, Int) throws -> Data

    /// この端末で使える Deflate の展開（iPad / Mac では Compression フレームワーク。Linux では無し）
    static var systemInflater: Inflater? {
        #if canImport(Compression)
        return { compressed, size in
            if size == 0 { return Data() }
            var output = Data(count: size)
            let written = output.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) -> Int in
                compressed.withUnsafeBytes { (source: UnsafeRawBufferPointer) -> Int in
                    compression_decode_buffer(destination.bindMemory(to: UInt8.self).baseAddress!, size,
                                              source.bindMemory(to: UInt8.self).baseAddress!, compressed.count,
                                              nil, COMPRESSION_ZLIB)
                }
            }
            guard written == size else { throw ZipError.corrupt("deflate") }
            return output
        }
        #else
        return nil
        #endif
    }

    let data: Data
    let entries: [Entry]

    init(data: Data) throws {
        self.data = data
        self.entries = try Self.readCentralDirectory(data)
    }

    init(url: URL) throws {
        try self.init(data: try Data(contentsOf: url, options: .alwaysMapped))
    }

    private static func readCentralDirectory(_ data: Data) throws -> [Entry] {
        let bytes = [UInt8](data.suffix(min(data.count, 65_557)))
        let offsetBase = data.count - bytes.count
        // 終端レコード（0x06054b50）を後ろから探す
        var eocd = -1
        var position = bytes.count - 22
        while position >= 0 {
            if bytes[position] == 0x50 && bytes[position + 1] == 0x4B && bytes[position + 2] == 0x05 && bytes[position + 3] == 0x06 {
                eocd = position
                break
            }
            position -= 1
        }
        guard eocd >= 0 else { throw ZipError.notZip }
        var count = Int(read16(bytes, eocd + 10))
        var directorySize = Int(read32(bytes, eocd + 12))
        var directoryOffset = Int(read32(bytes, eocd + 16))
        // ZIP64（4GB 超え・65535 件超え）
        if directoryOffset == 0xFFFF_FFFF || count == 0xFFFF, eocd >= 20 {
            let locator = eocd - 20
            if read32(bytes, locator) == 0x0706_4B50 {
                let recordOffset = Int(read64(bytes, locator + 8))
                let record = [UInt8](data[recordOffset..<min(recordOffset + 56, data.count)])
                guard record.count >= 56, read32(record, 0) == 0x0606_4B50 else { throw ZipError.corrupt("zip64") }
                count = Int(read64(record, 32))
                directorySize = Int(read64(record, 40))
                directoryOffset = Int(read64(record, 48))
            }
        }
        _ = offsetBase
        guard directoryOffset + directorySize <= data.count else { throw ZipError.corrupt("directory") }
        let directory = [UInt8](data[directoryOffset..<(directoryOffset + directorySize)])
        var entries: [Entry] = []
        var cursor = 0
        for _ in 0..<count {
            guard cursor + 46 <= directory.count, read32(directory, cursor) == 0x0201_4B50 else {
                throw ZipError.corrupt("entry")
            }
            let method = read16(directory, cursor + 10)
            let crc = read32(directory, cursor + 16)
            var compressed = Int(read32(directory, cursor + 20))
            var uncompressed = Int(read32(directory, cursor + 24))
            let nameLength = Int(read16(directory, cursor + 28))
            let extraLength = Int(read16(directory, cursor + 30))
            let commentLength = Int(read16(directory, cursor + 32))
            var localOffset = Int(read32(directory, cursor + 42))
            let nameStart = cursor + 46
            guard nameStart + nameLength + extraLength <= directory.count else { throw ZipError.corrupt("name") }
            let name = String(decoding: directory[nameStart..<(nameStart + nameLength)], as: UTF8.self)
            // ZIP64 の拡張フィールド
            var extra = nameStart + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let tag = read16(directory, extra)
                let size = Int(read16(directory, extra + 2))
                if tag == 0x0001 {
                    var field = extra + 4
                    if uncompressed == 0xFFFF_FFFF { uncompressed = Int(read64(directory, field)); field += 8 }
                    if compressed == 0xFFFF_FFFF { compressed = Int(read64(directory, field)); field += 8 }
                    if localOffset == 0xFFFF_FFFF { localOffset = Int(read64(directory, field)) }
                }
                extra += 4 + size
            }
            entries.append(Entry(path: name, method: method, compressedSize: compressed, uncompressedSize: uncompressed,
                                 crc32: crc, localHeaderOffset: localOffset))
            cursor = nameStart + nameLength + extraLength + commentLength
        }
        return entries
    }

    /// 1 ファイルの中身
    func contents(of entry: Entry, inflate: Inflater?) throws -> Data {
        let header = entry.localHeaderOffset
        guard header + 30 <= data.count else { throw ZipError.corrupt("local header") }
        let local = [UInt8](data[header..<(header + 30)])
        guard Self.read32(local, 0) == 0x0403_4B50 else { throw ZipError.corrupt("local header") }
        let start = header + 30 + Int(Self.read16(local, 26)) + Int(Self.read16(local, 28))
        guard start + entry.compressedSize <= data.count else { throw ZipError.corrupt("data") }
        let raw = data[start..<(start + entry.compressedSize)]
        let output: Data
        switch entry.method {
        case 0:
            output = Data(raw)
        case 8:
            guard let inflate else { throw ZipError.unsupportedMethod(8) }
            output = try inflate(Data(raw), entry.uncompressedSize)
        default:
            throw ZipError.unsupportedMethod(entry.method)
        }
        guard output.count == entry.uncompressedSize else { throw ZipError.corrupt(entry.path) }
        guard CRC32.checksum(output) == entry.crc32 else { throw ZipError.checksumMismatch(entry.path) }
        return output
    }

    /// 安全な相対パス（.. や絶対パスを含まない）か
    static func isSafe(_ path: String) -> Bool {
        if path.hasPrefix("/") || path.hasPrefix("\\") { return false }
        let parts = path.split(whereSeparator: { $0 == "/" || $0 == "\\" })
        return !parts.contains("..")
    }

    /// すべて（または prefix 以下だけ）を destination に書き出す。prefix は取り除く。
    /// 戻り値は書き出したファイルの相対パス。
    @discardableResult
    func extract(to destination: URL, stripping prefix: String = "", inflate: Inflater?,
                 skip: (String) -> Bool = { _ in false }) throws -> [String] {
        let fileManager = FileManager.default
        var written: [String] = []
        for entry in entries {
            guard entry.path.hasPrefix(prefix) else { continue }
            let relative = String(entry.path.dropFirst(prefix.count))
            if relative.isEmpty || skip(relative) { continue }
            // macOS が付ける余計なファイル
            if relative.hasPrefix("__MACOSX/") || (relative as NSString).lastPathComponent == ".DS_Store" { continue }
            guard Self.isSafe(relative) else { throw ZipError.unsafePath(entry.path) }
            let target = destination.appendingPathComponent(relative)
            if entry.isDirectory {
                try fileManager.createDirectory(at: target, withIntermediateDirectories: true)
                continue
            }
            try fileManager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try contents(of: entry, inflate: inflate).write(to: target, options: .atomic)
            written.append(relative)
        }
        return written
    }

    static func read16(_ bytes: [UInt8], _ offset: Int) -> UInt16 {
        let low = UInt16(bytes[offset])
        let high = UInt16(bytes[offset + 1])
        return low | (high << 8)
    }

    static func read32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        let low = UInt32(read16(bytes, offset))
        let high = UInt32(read16(bytes, offset + 2))
        return low | (high << 16)
    }

    static func read64(_ bytes: [UInt8], _ offset: Int) -> UInt64 {
        let low = UInt64(read32(bytes, offset))
        let high = UInt64(read32(bytes, offset + 4))
        return low | (high << 32)
    }
}

/// CRC-32（ZIP と同じ多項式）
enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { value -> UInt32 in
        var c = UInt32(value)
        for _ in 0..<8 {
            if c & 1 == 1 {
                c = 0xEDB8_8320 ^ (c >> 1)
            } else {
                c = c >> 1
            }
        }
        return c
    }

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        table.withUnsafeBufferPointer { table in
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                for byte in raw {
                    let index = Int((crc ^ UInt32(byte)) & 0xFF)
                    crc = table[index] ^ (crc >> 8)
                }
            }
        }
        return crc ^ 0xFFFF_FFFF
    }
}

/// ZIP を書く（無圧縮）。テストや、プロジェクトの書き出しに使う。
struct ZipWriter {
    private var body = Data()
    private var directory = Data()
    private var count = 0

    mutating func add(path: String, data: Data) {
        let name = Array(path.utf8)
        let crc = CRC32.checksum(data)
        let offset = UInt32(body.count)
        var local = Data()
        Self.append32(&local, 0x0403_4B50)
        Self.append16(&local, 20)            // 必要なバージョン
        Self.append16(&local, 0x0800)        // UTF-8 のファイル名
        Self.append16(&local, 0)             // 無圧縮
        Self.append16(&local, 0)             // 時刻
        Self.append16(&local, 0x21)          // 日付（1980-01-01）
        Self.append32(&local, crc)
        Self.append32(&local, UInt32(data.count))
        Self.append32(&local, UInt32(data.count))
        Self.append16(&local, UInt16(name.count))
        Self.append16(&local, 0)
        local.append(contentsOf: name)
        body.append(local)
        body.append(data)

        var entry = Data()
        Self.append32(&entry, 0x0201_4B50)
        Self.append16(&entry, 20)
        Self.append16(&entry, 20)
        Self.append16(&entry, 0x0800)
        Self.append16(&entry, 0)
        Self.append16(&entry, 0)
        Self.append16(&entry, 0x21)
        Self.append32(&entry, crc)
        Self.append32(&entry, UInt32(data.count))
        Self.append32(&entry, UInt32(data.count))
        Self.append16(&entry, UInt16(name.count))
        Self.append16(&entry, 0)
        Self.append16(&entry, 0)
        Self.append16(&entry, 0)
        Self.append16(&entry, 0)
        Self.append32(&entry, path.hasSuffix("/") ? 0x10 : 0)
        Self.append32(&entry, offset)
        entry.append(contentsOf: name)
        directory.append(entry)
        count += 1
    }

    func finish() -> Data {
        var output = body
        let directoryOffset = UInt32(output.count)
        output.append(directory)
        var end = Data()
        Self.append32(&end, 0x0605_4B50)
        Self.append16(&end, 0)
        Self.append16(&end, 0)
        Self.append16(&end, UInt16(count))
        Self.append16(&end, UInt16(count))
        Self.append32(&end, UInt32(directory.count))
        Self.append32(&end, directoryOffset)
        Self.append16(&end, 0)
        output.append(end)
        return output
    }

    private static func append16(_ data: inout Data, _ value: UInt16) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8(value >> 8))
    }

    private static func append32(_ data: inout Data, _ value: UInt32) {
        append16(&data, UInt16(value & 0xFFFF))
        append16(&data, UInt16(value >> 16))
    }
}
