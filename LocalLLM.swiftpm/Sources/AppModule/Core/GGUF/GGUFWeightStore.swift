import Foundation

/// ファイルをメモリに対応づけて読む（読んだ分だけ OS が読み込み、足りなくなれば OS が手放せる。アプリのメモリ使用量には数えられにくい）。
final class MappedFile {
    let pointer: UnsafeRawPointer
    let count: Int

    init(url: URL) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw CocoaError(.fileReadNoSuchFile) }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_size > 0 else { throw CocoaError(.fileReadCorruptFile) }
        let size = Int(info.st_size)
        guard let address = mmap(nil, size, PROT_READ, MAP_SHARED, descriptor, 0),
              address != UnsafeMutableRawPointer(bitPattern: -1) else {
            throw CocoaError(.fileReadUnknown)
        }
        pointer = UnsafeRawPointer(address)
        count = size
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: pointer), count)
    }
}

/// GGUF の重みを、このアプリの推論用に変換して保存したファイル（.llw）。
///
/// 行列は「1 行ごとに倍率を持つ 8 ビット整数」にして置く（1 パラメータ 1 バイト）。
/// 4 ビットなどの量子化を毎回ほどくより速く、Float32 のまま置くより 4 倍小さい。
/// 正規化の重みやバイアスなどの小さなものは Float32 のまま置く。
struct GGUFWeightStore {
    static let magic: [UInt8] = [0x4C, 0x4C, 0x57, 0x31]   // "LLW1"
    static let formatVersion: UInt32 = 2
    static let fileExtension = "llw"
    private static let alignment = 64

    struct Entry: Codable, Equatable, Sendable {
        var name: String
        var storage: GGUFMatrix.Storage
        var rows: Int
        var columns: Int
        /// データ部の先頭からの位置
        var offset: Int
        /// 行ごとの倍率（Float32 × rows）の位置。int8Rows のときだけ
        var scalesOffset: Int?
    }

    struct Header: Codable, Equatable, Sendable {
        var config: GGUFModelConfig
        var entries: [Entry]
        /// 変換元のファイルの大きさ（元のファイルが変わっていないかの確認用）
        var sourceBytes: Int
        var dataBytes: Int
    }

    let header: Header
    private let file: MappedFile
    private let dataStart: Int
    private let index: [String: Int]

    var config: GGUFModelConfig { header.config }

    /// 保存ファイルを開く
    init(url: URL) throws {
        let file = try MappedFile(url: url)
        guard file.count >= 16 else { throw GGUFError.invalid("weight store") }
        let bytes = file.pointer.assumingMemoryBound(to: UInt8.self)
        for (offset, byte) in Self.magic.enumerated() where bytes[offset] != byte { throw GGUFError.invalid("weight store magic") }
        let version = file.pointer.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
        guard UInt32(littleEndian: version) == Self.formatVersion else { throw GGUFError.invalid("weight store version") }
        let headerLength = Int(UInt64(littleEndian: file.pointer.loadUnaligned(fromByteOffset: 8, as: UInt64.self)))
        guard headerLength > 0, 16 + headerLength <= file.count else { throw GGUFError.invalid("weight store header") }
        let headerData = Data(bytes: file.pointer + 16, count: headerLength)
        let header = try JSONDecoder().decode(Header.self, from: headerData)
        let dataStart = Self.aligned(16 + headerLength, to: 4096)
        guard dataStart + header.dataBytes <= file.count else { throw GGUFError.truncated(needed: dataStart + header.dataBytes) }
        self.file = file
        self.header = header
        self.dataStart = dataStart
        var index: [String: Int] = [:]
        for (position, entry) in header.entries.enumerated() { index[entry.name] = position }
        self.index = index
    }

    func has(_ name: String) -> Bool { index[name] != nil }

    func matrix(_ name: String) throws -> GGUFMatrix {
        guard let position = index[name] else { throw GGUFError.missingTensor(name) }
        let entry = header.entries[position]
        let base = file.pointer + dataStart
        let scales = entry.scalesOffset.map { (base + $0).assumingMemoryBound(to: Float.self) }
        return GGUFMatrix(rows: entry.rows, columns: entry.columns, storage: entry.storage, data: base + entry.offset, scales: scales)
    }

    /// Float32 のベクトル（正規化の重み・バイアスなど）。無ければ nil
    func vector(_ name: String) -> UnsafePointer<Float>? {
        guard let position = index[name] else { return nil }
        let entry = header.entries[position]
        guard entry.storage == .float32 else { return nil }
        return (file.pointer + dataStart + entry.offset).assumingMemoryBound(to: Float.self)
    }

    func requiredVector(_ name: String) throws -> UnsafePointer<Float> {
        guard let vector = vector(name) else { throw GGUFError.missingTensor(name) }
        return vector
    }

    static func aligned(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }

    /// 変換後のファイルの大きさの見積もり（バイト）
    static func estimatedBytes(file: GGUFFile, config: GGUFModelConfig, compact: Bool) -> Int {
        layout(file: file, config: config, compact: compact).dataBytes + 8192
    }

    /// 重みの置き場所を決める
    static func layout(file: GGUFFile, config: GGUFModelConfig, compact: Bool) -> (entries: [Entry], dataBytes: Int) {
        var entries: [Entry] = []
        var offset = 0
        for name in config.tensorNames(in: file) {
            guard let tensor = file.tensor(name) else { continue }
            let isMatrix = tensor.dimensions.count >= 2
            let storage: GGUFMatrix.Storage = (isMatrix && compact) ? .int8Rows : .float32
            let rows = isMatrix ? tensor.rows : 1
            let columns = isMatrix ? tensor.columns : tensor.elementCount
            offset = aligned(offset, to: alignment)
            var entry = Entry(name: name, storage: storage, rows: rows, columns: columns, offset: offset, scalesOffset: nil)
            offset += rows * columns * (storage == .float32 ? 4 : 1)
            if storage == .int8Rows {
                offset = aligned(offset, to: alignment)
                entry.scalesOffset = offset
                offset += rows * 4
            }
            entries.append(entry)
        }
        return (entries, aligned(offset, to: alignment))
    }
}

/// GGUF を読み、推論用の保存ファイル（.llw）に変換する。
enum GGUFConverter {
    /// - Parameters:
    ///   - compact: true なら行列を 8 ビットにする（ふつうはこちら）。false なら Float32 のまま（大きいが元の値どおり）
    ///   - progress: 0〜1
    ///   - isCancelled: true を返すと CancellationError で止まる
    @discardableResult
    static func convert(gguf source: URL, to destination: URL, compact: Bool = true,
                        progress: ((Double) -> Void)? = nil, isCancelled: (() -> Bool)? = nil) throws -> GGUFModelConfig {
        let mapped = try MappedFile(url: source)
        let file = try GGUFFile.parse(Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: mapped.pointer), count: mapped.count,
                                           deallocator: .none))
        if (file.int("split.count") ?? 1) > 1 {
            throw GGUFError.invalid(L("何個かに分かれた GGUF（split）には対応していません。1 つにまとまったファイルを選んでください。",
                                      "Split GGUF files are not supported. Choose a single-file GGUF."))
        }
        let config = try GGUFModelConfig(file: file)
        guard file.expectedFileSize <= mapped.count else { throw GGUFError.truncated(needed: file.expectedFileSize) }
        let (entries, dataBytes) = GGUFWeightStore.layout(file: file, config: config, compact: compact)
        let header = GGUFWeightStore.Header(config: config, entries: entries, sourceBytes: mapped.count, dataBytes: dataBytes)
        let headerData = try JSONEncoder().encode(header)

        let fileManager = FileManager.default
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".converting-\(UUID().uuidString)")
        fileManager.createFile(atPath: temporary.path, contents: nil)
        let handle = try FileHandle(forWritingTo: temporary)
        var finished = false
        defer {
            try? handle.close()
            if !finished { try? fileManager.removeItem(at: temporary) }
        }

        // 先頭: 目印・版・ヘッダーの長さ・ヘッダー（JSON）
        var prefix = Data(GGUFWeightStore.magic)
        withUnsafeBytes(of: GGUFWeightStore.formatVersion.littleEndian) { prefix.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { prefix.append(contentsOf: $0) }
        prefix.append(headerData)
        let dataStart = GGUFWeightStore.aligned(prefix.count, to: 4096)
        prefix.append(Data(count: dataStart - prefix.count))
        try handle.write(contentsOf: prefix)

        let totalElements = max(1, entries.reduce(0) { $0 + $1.rows * $1.columns })
        var doneElements = 0
        var written = 0
        let writer = ChunkWriter(threads: max(1, min(4, ProcessInfo.processInfo.activeProcessorCount)))
        defer { writer.free() }

        func pad(to offset: Int) throws {
            if offset > written {
                try handle.write(contentsOf: Data(count: offset - written))
                written = offset
            }
        }

        for entry in entries {
            guard let tensor = file.tensor(entry.name), let type = tensor.type, let byteCount = tensor.byteCount else {
                throw GGUFError.missingTensor(entry.name)
            }
            let start = file.fileOffset(of: tensor)
            guard start + byteCount <= mapped.count else { throw GGUFError.truncated(needed: start + byteCount) }
            try pad(to: entry.offset)
            let rowBytes = byteCount / entry.rows
            var scales = [Float](repeating: 0, count: entry.storage == .int8Rows ? entry.rows : 0)
            var row = 0
            while row < entry.rows {
                if isCancelled?() == true { throw CancellationError() }
                let rows = try writer.convert(type: type, source: mapped.pointer + start + row * rowBytes, rowBytes: rowBytes,
                                              rows: entry.rows - row, columns: entry.columns, storage: entry.storage,
                                              scales: &scales, firstRow: row)
                let bytes = rows * entry.columns * (entry.storage == .float32 ? 4 : 1)
                try handle.write(contentsOf: writer.output(byteCount: bytes))
                written += bytes
                row += rows
                doneElements += rows * entry.columns
                progress?(Double(doneElements) / Double(totalElements))
            }
            if let scalesOffset = entry.scalesOffset {
                try pad(to: scalesOffset)
                try scales.withUnsafeBytes { try handle.write(contentsOf: Data($0)) }
                written += scales.count * 4
            }
        }
        try pad(to: dataBytes)
        try handle.close()
        try? fileManager.removeItem(at: destination)
        try fileManager.moveItem(at: temporary, to: destination)
        finished = true
        progress?(1)
        return config
    }

    /// 何行かずつ Float に戻して（必要なら 8 ビットにして）書き出す。行のまとまりごとに並列に計算する。
    private final class ChunkWriter {
        private let threads: Int
        /// 1 回に扱う要素数（スレッド 1 つあたり）
        private let elementsPerThread = 1 << 18
        private let floats: UnsafeMutablePointer<Float>
        private let scaled: UnsafeMutablePointer<Float>
        private let bytes: UnsafeMutablePointer<Int8>
        private let capacity: Int
        private var failure: Error?
        private let lock = NSLock()
        private var outputIsFloat = true

        init(threads: Int) {
            self.threads = threads
            capacity = elementsPerThread * threads
            floats = .allocate(capacity: capacity)
            scaled = .allocate(capacity: capacity)
            bytes = .allocate(capacity: capacity)
        }

        func free() {
            floats.deallocate()
            scaled.deallocate()
            bytes.deallocate()
        }

        /// 最大 rows 行を変換して、変換した行数を返す（結果は output で取り出す）
        func convert(type: GGUFTensorType, source: UnsafeRawPointer, rowBytes: Int, rows: Int, columns: Int,
                     storage: GGUFMatrix.Storage, scales: inout [Float], firstRow: Int) throws -> Int {
            guard columns <= capacity else { throw GGUFError.invalid("row too long") }
            let rowsPerThread = max(1, min(elementsPerThread, capacity / threads) / columns)
            let count = min(rows, rowsPerThread * threads)
            let jobs = (count + rowsPerThread - 1) / rowsPerThread
            outputIsFloat = storage == .float32
            failure = nil
            let rowScales = UnsafeMutablePointer<Float>.allocate(capacity: count)
            defer { rowScales.deallocate() }
            DispatchQueue.concurrentPerform(iterations: jobs) { job in
                let first = job * rowsPerThread
                let last = min(count, first + rowsPerThread)
                do {
                    try GGUFQuant.dequantize(type, source: source + first * rowBytes, count: (last - first) * columns,
                                             into: floats + first * columns)
                    guard storage == .int8Rows else { return }
                    for row in first..<last {
                        rowScales[row] = GGUFKernels.quantizeRow(floats + row * columns, count: columns,
                                                                 into: bytes + row * columns, scratch: scaled + row * columns)
                    }
                } catch {
                    lock.lock()
                    failure = error
                    lock.unlock()
                }
            }
            if let failure { throw failure }
            if storage == .int8Rows {
                for row in 0..<count { scales[firstRow + row] = rowScales[row] }
            }
            return count
        }

        func output(byteCount: Int) -> Data {
            let pointer = outputIsFloat ? UnsafeMutableRawPointer(floats) : UnsafeMutableRawPointer(bytes)
            return Data(bytesNoCopy: pointer, count: byteCount, deallocator: .none)
        }
    }
}
