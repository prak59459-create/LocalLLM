import Foundation

/// GGUF（llama.cpp のモデルファイル）を読むときのエラー。
enum GGUFError: Error, LocalizedError, Equatable {
    case notGGUF
    case unsupportedVersion(Int)
    /// 渡されたデータが途中で終わっている（先頭だけ取ってきたときなど）。needed は最低限必要なバイト数の目安。
    case truncated(needed: Int)
    case invalid(String)
    case unsupportedTensorType(String)
    case unsupportedArchitecture(String)
    case missingTensor(String)
    case missingMetadata(String)

    var errorDescription: String? {
        switch self {
        case .notGGUF:
            return L("GGUF ファイルではありません。", "This is not a GGUF file.")
        case .unsupportedVersion(let version):
            return L("この GGUF のバージョン（\(version)）には対応していません。", "GGUF version \(version) is not supported.")
        case .truncated:
            return L("GGUF ファイルが途中で終わっています（ダウンロードが完了していない可能性があります）。",
                     "The GGUF file ends early (the download may be incomplete).")
        case .invalid(let detail):
            return L("GGUF ファイルの内容が想定外です: \(detail)", "Unexpected GGUF content: \(detail)")
        case .unsupportedTensorType(let name):
            return L("この量子化形式（\(name)）には対応していません。Q4_K_M や Q8_0 のファイルを選んでください。",
                     "The quantization \(name) is not supported. Choose a Q4_K_M or Q8_0 file.")
        case .unsupportedArchitecture(let name):
            return L("このモデルの構造（\(name)）には対応していません。対応: llama / qwen2 / qwen3 / gemma / gemma2 / gemma3 / phi3 など。",
                     "The architecture \(name) is not supported. Supported: llama, qwen2, qwen3, gemma, gemma2, gemma3, phi3 and similar.")
        case .missingTensor(let name):
            return L("GGUF に必要な重み「\(name)」がありません。", "The GGUF file is missing the tensor \"\(name)\".")
        case .missingMetadata(let key):
            return L("GGUF に必要な情報「\(key)」がありません。", "The GGUF file is missing the metadata \"\(key)\".")
        }
    }
}

/// GGUF のメタデータの値。
enum GGUFValue: Equatable, Sendable {
    case uint(UInt64)
    case int(Int64)
    case float(Double)
    case bool(Bool)
    case string(String)
    case strings([String])
    case ints([Int])
    case floats([Float])
    case array([GGUFValue])

    var intValue: Int? {
        switch self {
        case .uint(let value): return value <= UInt64(Int.max) ? Int(value) : nil
        case .int(let value): return Int(value)
        case .float(let value): return value.isFinite && abs(value) < 9e15 ? Int(value) : nil
        case .bool(let value): return value ? 1 : 0
        default: return nil
        }
    }

    var doubleValue: Double? {
        switch self {
        case .uint(let value): return Double(value)
        case .int(let value): return Double(value)
        case .float(let value): return value
        default: return nil
        }
    }

    var boolValue: Bool? {
        switch self {
        case .bool(let value): return value
        case .uint(let value): return value != 0
        case .int(let value): return value != 0
        default: return nil
        }
    }

    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var stringArray: [String]? {
        switch self {
        case .strings(let values): return values
        case .array(let values): return values.isEmpty ? [] : nil
        default: return nil
        }
    }

    var intArray: [Int]? {
        switch self {
        case .ints(let values): return values
        case .floats(let values): return values.map { Int($0) }
        case .array(let values): return values.isEmpty ? [] : nil
        default: return nil
        }
    }

    var floatArray: [Float]? {
        switch self {
        case .floats(let values): return values
        case .ints(let values): return values.map { Float($0) }
        case .array(let values): return values.isEmpty ? [] : nil
        default: return nil
        }
    }

    /// 配列の要素数（配列でなければ nil）
    var count: Int? {
        switch self {
        case .strings(let values): return values.count
        case .ints(let values): return values.count
        case .floats(let values): return values.count
        case .array(let values): return values.count
        default: return nil
        }
    }
}

/// 重みの保存形式（ggml_type）。
enum GGUFTensorType: UInt32, Sendable, CaseIterable {
    case f32 = 0, f16 = 1, q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8, q8_1 = 9
    case q2_K = 10, q3_K = 11, q4_K = 12, q5_K = 13, q6_K = 14, q8_K = 15
    case iq2_XXS = 16, iq2_XS = 17, iq3_XXS = 18, iq1_S = 19, iq4_NL = 20, iq3_S = 21, iq2_S = 22, iq4_XS = 23
    case i8 = 24, i16 = 25, i32 = 26, i64 = 27, f64 = 28, iq1_M = 29, bf16 = 30
    case tq1_0 = 34, tq2_0 = 35

    /// 1 ブロックの要素数とバイト数
    var block: (elements: Int, bytes: Int) {
        switch self {
        case .f32: return (1, 4)
        case .f16, .bf16: return (1, 2)
        case .q4_0: return (32, 18)
        case .q4_1: return (32, 20)
        case .q5_0: return (32, 22)
        case .q5_1: return (32, 24)
        case .q8_0: return (32, 34)
        case .q8_1: return (32, 40)
        case .q2_K: return (256, 84)
        case .q3_K: return (256, 110)
        case .q4_K: return (256, 144)
        case .q5_K: return (256, 176)
        case .q6_K: return (256, 210)
        case .q8_K: return (256, 292)
        case .iq2_XXS: return (256, 66)
        case .iq2_XS: return (256, 74)
        case .iq3_XXS: return (256, 98)
        case .iq1_S: return (256, 50)
        case .iq4_NL: return (32, 18)
        case .iq3_S: return (256, 110)
        case .iq2_S: return (256, 82)
        case .iq4_XS: return (256, 136)
        case .i8: return (1, 1)
        case .i16: return (1, 2)
        case .i32: return (1, 4)
        case .i64, .f64: return (1, 8)
        case .iq1_M: return (256, 56)
        case .tq1_0: return (256, 54)
        case .tq2_0: return (256, 66)
        }
    }

    var name: String {
        switch self {
        case .f32: return "F32"
        case .f16: return "F16"
        case .bf16: return "BF16"
        case .f64: return "F64"
        case .q4_0: return "Q4_0"
        case .q4_1: return "Q4_1"
        case .q5_0: return "Q5_0"
        case .q5_1: return "Q5_1"
        case .q8_0: return "Q8_0"
        case .q8_1: return "Q8_1"
        case .q2_K: return "Q2_K"
        case .q3_K: return "Q3_K"
        case .q4_K: return "Q4_K"
        case .q5_K: return "Q5_K"
        case .q6_K: return "Q6_K"
        case .q8_K: return "Q8_K"
        case .iq2_XXS: return "IQ2_XXS"
        case .iq2_XS: return "IQ2_XS"
        case .iq2_S: return "IQ2_S"
        case .iq3_XXS: return "IQ3_XXS"
        case .iq3_S: return "IQ3_S"
        case .iq1_S: return "IQ1_S"
        case .iq1_M: return "IQ1_M"
        case .iq4_NL: return "IQ4_NL"
        case .iq4_XS: return "IQ4_XS"
        case .i8: return "I8"
        case .i16: return "I16"
        case .i32: return "I32"
        case .i64: return "I64"
        case .tq1_0: return "TQ1_0"
        case .tq2_0: return "TQ2_0"
        }
    }

    /// このアプリで Float に戻せる形式か
    var isSupported: Bool {
        switch self {
        case .iq1_S, .iq1_M, .tq1_0, .tq2_0, .q8_1, .q8_K, .i8, .i16, .i32, .i64, .f64: return false
        default: return true
        }
    }
}

/// 重み 1 つ分の情報。
struct GGUFTensorInfo: Equatable, Sendable {
    var name: String
    /// 各次元の大きさ（ggml の順。dimensions[0] が 1 行の長さ、dimensions[1] が行数）
    var dimensions: [Int]
    var typeCode: UInt32
    /// データ部の先頭からの位置
    var offset: UInt64

    var type: GGUFTensorType? { GGUFTensorType(rawValue: typeCode) }
    var elementCount: Int { dimensions.reduce(1, *) }
    /// 1 行の長さ（線形層なら入力の大きさ）
    var columns: Int { dimensions.first ?? 1 }
    /// 行数（線形層なら出力の大きさ。1 次元なら 1）
    var rows: Int { dimensions.count > 1 ? dimensions.dropFirst().reduce(1, *) : 1 }

    /// ファイル上のバイト数
    var byteCount: Int? {
        guard let type else { return nil }
        let block = type.block
        guard elementCount % block.elements == 0 else { return nil }
        return elementCount / block.elements * block.bytes
    }
}

/// GGUF ファイルの先頭部分（メタデータと重みの一覧）。重みの中身は含まない。
struct GGUFFile: Sendable {
    static let magic: [UInt8] = [0x47, 0x47, 0x55, 0x46]   // "GGUF"

    var version: Int
    var metadata: [String: GGUFValue]
    var tensors: [GGUFTensorInfo]
    /// 重みのデータが始まる位置（ファイルの先頭から）
    var dataOffset: Int
    var alignment: Int

    private var tensorIndex: [String: Int] = [:]

    func tensor(_ name: String) -> GGUFTensorInfo? {
        tensorIndex[name].map { tensors[$0] }
    }

    func has(_ name: String) -> Bool { tensorIndex[name] != nil }

    /// ファイルの先頭からの、重みの位置
    func fileOffset(of tensor: GGUFTensorInfo) -> Int {
        dataOffset + Int(tensor.offset)
    }

    /// 重みのデータの終わり（ファイル全体の大きさの目安）
    var expectedFileSize: Int {
        var end = dataOffset
        for tensor in tensors {
            end = max(end, fileOffset(of: tensor) + (tensor.byteCount ?? 0))
        }
        return end
    }

    // MARK: - よく使うメタデータ

    var architecture: String { metadata["general.architecture"]?.stringValue ?? "" }
    var name: String? { metadata["general.name"]?.stringValue }

    func int(_ key: String) -> Int? { metadata[key]?.intValue }
    func double(_ key: String) -> Double? { metadata[key]?.doubleValue }
    func string(_ key: String) -> String? { metadata[key]?.stringValue }
    func bool(_ key: String) -> Bool? { metadata[key]?.boolValue }

    /// "{architecture}.name" の値
    func archInt(_ suffix: String) -> Int? { int("\(architecture).\(suffix)") }
    func archDouble(_ suffix: String) -> Double? { double("\(architecture).\(suffix)") }

    /// 全体のパラメータ数
    var parameterCount: Int { tensors.reduce(0) { $0 + $1.elementCount } }

    /// いちばん多く使われている量子化形式の名前（"Q4_K" など）
    var dominantQuantization: String {
        var counts: [UInt32: Int] = [:]
        for tensor in tensors where tensor.dimensions.count >= 2 { counts[tensor.typeCode, default: 0] += tensor.elementCount }
        guard let top = counts.max(by: { $0.value < $1.value })?.key else { return "?" }
        return GGUFTensorType(rawValue: top)?.name ?? "type \(top)"
    }

    // MARK: - 読み込み

    /// ファイルを開いて先頭部分を読む（ファイル全体はメモリに読まない）
    static func open(_ url: URL) throws -> GGUFFile {
        let data = try Data(contentsOf: url, options: [.alwaysMapped])
        return try parse(data)
    }

    /// 先頭部分のバイト列から読む。足りなければ GGUFError.truncated を投げる。
    static func parse(_ data: Data) throws -> GGUFFile {
        try data.withUnsafeBytes { raw -> GGUFFile in
            var reader = GGUFReader(bytes: raw)
            return try parse(&reader)
        }
    }

    private static func parse(_ reader: inout GGUFReader) throws -> GGUFFile {
        guard reader.bytes.count >= 4 else { throw GGUFError.truncated(needed: 1 << 16) }
        for (index, byte) in magic.enumerated() where reader.bytes[index] != byte { throw GGUFError.notGGUF }
        reader.offset = 4
        let version = Int(try reader.u32())
        guard version == 2 || version == 3 else { throw GGUFError.unsupportedVersion(version) }
        let tensorCount = try reader.count()
        let metadataCount = try reader.count()
        guard tensorCount < 1_000_000, metadataCount < 1_000_000 else { throw GGUFError.invalid("counts") }

        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(metadataCount)
        for _ in 0..<metadataCount {
            let key = try reader.string()
            let type = try reader.u32()
            metadata[key] = try reader.value(type: type, depth: 0)
        }

        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(tensorCount)
        var index: [String: Int] = [:]
        index.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try reader.string()
            let dimensionCount = Int(try reader.u32())
            guard dimensionCount <= 8 else { throw GGUFError.invalid("tensor dimensions") }
            var dimensions: [Int] = []
            for _ in 0..<dimensionCount { dimensions.append(try reader.count()) }
            let type = try reader.u32()
            let offset = try reader.u64()
            index[name] = tensors.count
            tensors.append(GGUFTensorInfo(name: name, dimensions: dimensions, typeCode: type, offset: offset))
        }
        let alignment = max(1, metadata["general.alignment"]?.intValue ?? 32)
        let dataOffset = (reader.offset + alignment - 1) / alignment * alignment
        var file = GGUFFile(version: version, metadata: metadata, tensors: tensors, dataOffset: dataOffset, alignment: alignment)
        file.tensorIndex = index
        return file
    }
}

/// GGUF のバイト列を先頭から順に読む。
struct GGUFReader {
    let bytes: UnsafeRawBufferPointer
    var offset = 0

    init(bytes: UnsafeRawBufferPointer) {
        self.bytes = bytes
    }

    private mutating func take(_ count: Int) throws -> Int {
        guard count >= 0, offset + count <= bytes.count else {
            throw GGUFError.truncated(needed: max(offset + max(count, 0), bytes.count * 2))
        }
        let start = offset
        offset += count
        return start
    }

    mutating func u8() throws -> UInt8 {
        bytes[try take(1)]
    }

    mutating func u16() throws -> UInt16 {
        UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: try take(2), as: UInt16.self))
    }

    mutating func u32() throws -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: try take(4), as: UInt32.self))
    }

    mutating func u64() throws -> UInt64 {
        UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: try take(8), as: UInt64.self))
    }

    /// 個数・長さ（大きすぎる値は壊れたファイルとして扱う）
    mutating func count() throws -> Int {
        let value = try u64()
        guard value < UInt64(1) << 40 else { throw GGUFError.invalid("length") }
        return Int(value)
    }

    mutating func string() throws -> String {
        let length = try count()
        let start = try take(length)
        let slice = UnsafeRawBufferPointer(rebasing: bytes[start..<(start + length)])
        return String(decoding: slice, as: UTF8.self)
    }

    mutating func value(type: UInt32, depth: Int) throws -> GGUFValue {
        switch type {
        case 0: return .uint(UInt64(try u8()))
        case 1: return .int(Int64(Int8(bitPattern: try u8())))
        case 2: return .uint(UInt64(try u16()))
        case 3: return .int(Int64(Int16(bitPattern: try u16())))
        case 4: return .uint(UInt64(try u32()))
        case 5: return .int(Int64(Int32(bitPattern: try u32())))
        case 6: return .float(Double(Float(bitPattern: try u32())))
        case 7: return .bool(try u8() != 0)
        case 8: return .string(try string())
        case 9:
            guard depth < 4 else { throw GGUFError.invalid("nested arrays") }
            let elementType = try u32()
            let length = try count()
            return try array(elementType: elementType, length: length, depth: depth)
        case 10: return .uint(try u64())
        case 11: return .int(Int64(bitPattern: try u64()))
        case 12: return .float(Double(bitPattern: try u64()))
        default: throw GGUFError.invalid("value type \(type)")
        }
    }

    private mutating func array(elementType: UInt32, length: Int, depth: Int) throws -> GGUFValue {
        switch elementType {
        case 8:
            var values: [String] = []
            values.reserveCapacity(min(length, 1 << 20))
            for _ in 0..<length { values.append(try string()) }
            return .strings(values)
        case 6:
            let start = try take(length * 4)
            var values = [Float](repeating: 0, count: length)
            values.withUnsafeMutableBytes { destination in
                destination.copyMemory(from: UnsafeRawBufferPointer(rebasing: bytes[start..<(start + length * 4)]))
            }
            return .floats(values)
        case 12:
            var values: [Float] = []
            values.reserveCapacity(min(length, 1 << 20))
            for _ in 0..<length { values.append(Float(Double(bitPattern: try u64()))) }
            return .floats(values)
        case 0, 1, 2, 3, 4, 5, 7, 10, 11:
            var values: [Int] = []
            values.reserveCapacity(min(length, 1 << 20))
            for _ in 0..<length {
                guard let number = try value(type: elementType, depth: depth + 1).intValue else {
                    throw GGUFError.invalid("array value")
                }
                values.append(number)
            }
            return .ints(values)
        default:
            var values: [GGUFValue] = []
            for _ in 0..<length { values.append(try value(type: elementType, depth: depth + 1)) }
            return .array(values)
        }
    }
}
