import Foundation

/// Core ML（ML Program）の重みファイル（weights/weight.bin など）の形式。
/// 先頭 64 バイトのヘッダ（個数・版 2）のあと、値ごとに 64 バイトの説明（0xDEADBEEF・型・大きさ・データの位置）と、
/// 64 バイト境界にそろえたデータが並ぶ。MIL の BLOBFILE(offset = …) は説明の位置を指す。
enum MILBlob {
    enum DataType: UInt32, Sendable {
        case float16 = 1
        case float32 = 2
        case uint8 = 3
        case int8 = 4

        var byteSize: Int {
            switch self {
            case .float16: return 2
            case .float32: return 4
            case .uint8, .int8: return 1
            }
        }
    }

    struct Entry: Sendable {
        var dataType: UInt32
        var size: UInt64
        var dataOffset: UInt64
    }

    enum BlobError: Error, LocalizedError {
        case badHeader
        case badEntry(UInt64)
        case unsupportedType(UInt32)

        var errorDescription: String? {
            switch self {
            case .badHeader: return L("重みファイルの形式が想定外です", "Unexpected weight file format")
            case .badEntry(let offset): return L("重みファイルの \(offset) の位置の値が読めません", "Cannot read the weight entry at \(offset)")
            case .unsupportedType(let type): return L("対応していない重みの型です（\(type)）", "Unsupported weight data type (\(type))")
            }
        }
    }

    static let sentinel: UInt32 = 0xDEAD_BEEF

    /// offset（BLOBFILE の offset）にある値の説明を読む
    static func entry(in file: FileHandle, at offset: UInt64) throws -> Entry {
        try file.seek(toOffset: offset)
        guard let header = try file.read(upToCount: 64), header.count == 64 else { throw BlobError.badEntry(offset) }
        let bytes = [UInt8](header)
        guard readUInt32(bytes, 0) == sentinel else { throw BlobError.badEntry(offset) }
        return Entry(dataType: readUInt32(bytes, 4), size: readUInt64(bytes, 8), dataOffset: readUInt64(bytes, 16))
    }

    /// 値を Float の配列として読む（float16 / float32）
    static func readFloats(in file: FileHandle, at offset: UInt64) throws -> [Float] {
        let entry = try entry(in: file, at: offset)
        guard let type = DataType(rawValue: entry.dataType), type == .float16 || type == .float32 else {
            throw BlobError.unsupportedType(entry.dataType)
        }
        try file.seek(toOffset: entry.dataOffset)
        guard let data = try file.read(upToCount: Int(entry.size)), data.count == Int(entry.size) else {
            throw BlobError.badEntry(offset)
        }
        let bytes = [UInt8](data)
        if type == .float32 {
            return stride(from: 0, to: bytes.count, by: 4).map { Float(bitPattern: readUInt32(bytes, $0)) }
        }
        return stride(from: 0, to: bytes.count, by: 2).map { Half.toFloat(UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8) }
    }

    /// 新しい重みファイルを作る。戻り値は値ごとの BLOBFILE の offset。
    static func write(_ blobs: [(type: DataType, values: [Float])], to url: URL) throws -> [UInt64] {
        var output = [UInt8](repeating: 0, count: 64)
        writeUInt32(UInt32(blobs.count), into: &output, at: 0)
        writeUInt32(2, into: &output, at: 4)
        var offsets: [UInt64] = []
        for blob in blobs {
            let metadataOffset = UInt64(output.count)
            let dataOffset = metadataOffset + 64
            var payload: [UInt8] = []
            payload.reserveCapacity(blob.values.count * blob.type.byteSize)
            for value in blob.values {
                switch blob.type {
                case .float16:
                    let bits = Half.fromFloat(value)
                    payload.append(UInt8(bits & 0xFF))
                    payload.append(UInt8(bits >> 8))
                case .float32:
                    let bits = value.bitPattern
                    for shift in stride(from: 0, to: 32, by: 8) { payload.append(UInt8((bits >> UInt32(shift)) & 0xFF)) }
                default:
                    throw BlobError.unsupportedType(blob.type.rawValue)
                }
            }
            var metadata = [UInt8](repeating: 0, count: 64)
            writeUInt32(sentinel, into: &metadata, at: 0)
            writeUInt32(blob.type.rawValue, into: &metadata, at: 4)
            writeUInt64(UInt64(payload.count), into: &metadata, at: 8)
            writeUInt64(dataOffset, into: &metadata, at: 16)
            output.append(contentsOf: metadata)
            output.append(contentsOf: payload)
            let padding = (64 - output.count % 64) % 64
            output.append(contentsOf: [UInt8](repeating: 0, count: padding))
            offsets.append(metadataOffset)
        }
        try Data(output).write(to: url, options: .atomic)
        return offsets
    }

    static func readUInt32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << (8 * UInt32($1)) }
    }

    static func readUInt64(_ bytes: [UInt8], _ offset: Int) -> UInt64 {
        (0..<8).reduce(UInt64(0)) { $0 | UInt64(bytes[offset + $1]) << (8 * UInt64($1)) }
    }

    static func writeUInt32(_ value: UInt32, into bytes: inout [UInt8], at offset: Int) {
        for index in 0..<4 { bytes[offset + index] = UInt8((value >> (8 * UInt32(index))) & 0xFF) }
    }

    static func writeUInt64(_ value: UInt64, into bytes: inout [UInt8], at offset: Int) {
        for index in 0..<8 { bytes[offset + index] = UInt8((value >> (8 * UInt64(index))) & 0xFF) }
    }
}

/// 半精度（IEEE 754 binary16）と Float の変換（Core では Float16 型を使わない）。
enum Half {
    static func toFloat(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = UInt32(bits & 0x3FF)
        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }
            // 非正規化数
            let value = Float(mantissa) / 1024 * powf(2, -14)
            return sign == 0 ? value : -value
        }
        if exponent == 31 {
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }
        return Float(bitPattern: sign | UInt32(exponent - 15 + 127) << 23 | mantissa << 13)
    }

    static func fromFloat(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        var mantissa = bits & 0x7F_FFFF
        if value.isNaN { return sign | 0x7E00 }
        if exponent >= 31 { return sign | 0x7C00 }
        if exponent <= 0 {
            // 非正規化数（または 0）
            if exponent < -10 { return sign }
            mantissa |= 0x80_0000
            let shift = UInt32(14 - exponent)
            var half = mantissa >> shift
            let remainder = mantissa & ((1 << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if remainder > halfway || (remainder == halfway && half & 1 == 1) { half += 1 }
            return sign | UInt16(half)
        }
        var half = UInt32(exponent) << 10 | (mantissa >> 13)
        let remainder = mantissa & 0x1FFF
        if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) { half += 1 }
        return sign | UInt16(truncatingIfNeeded: half)
    }
}
