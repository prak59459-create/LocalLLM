import Foundation

/// protobuf のワイヤー形式を、知らないフィールドも含めてそのまま読み書きする最小限の実装。
/// Core ML の coremldata.bin（ModelDescription）の一部の値だけを書き換えるのに使う。
struct ProtoField: Equatable, Sendable {
    enum Value: Equatable, Sendable {
        case varint(UInt64)
        case fixed64(UInt64)
        case bytes(Data)
        case fixed32(UInt32)
    }

    var number: Int
    var value: Value

    var varint: UInt64? {
        if case .varint(let value) = value { return value }
        return nil
    }

    var bytes: Data? {
        if case .bytes(let data) = value { return data }
        return nil
    }

    var string: String? { bytes.flatMap { String(data: $0, encoding: .utf8) } }
}

enum ProtoWireError: Error, LocalizedError {
    case truncated
    case badWireType(Int)

    var errorDescription: String? {
        switch self {
        case .truncated: return L("protobuf のデータが途中で切れています", "Truncated protobuf data")
        case .badWireType(let type): return L("protobuf の形式が想定外です（\(type)）", "Unexpected protobuf wire type (\(type))")
        }
    }
}

enum ProtoWire {
    static func parse(_ data: Data) throws -> [ProtoField] {
        let bytes = [UInt8](data)
        var index = 0
        var fields: [ProtoField] = []
        while index < bytes.count {
            let key = try readVarint(bytes, &index)
            let number = Int(key >> 3)
            switch Int(key & 7) {
            case 0:
                fields.append(ProtoField(number: number, value: .varint(try readVarint(bytes, &index))))
            case 1:
                guard index + 8 <= bytes.count else { throw ProtoWireError.truncated }
                var value: UInt64 = 0
                for offset in 0..<8 { value |= UInt64(bytes[index + offset]) << (8 * UInt64(offset)) }
                index += 8
                fields.append(ProtoField(number: number, value: .fixed64(value)))
            case 2:
                let length = Int(try readVarint(bytes, &index))
                guard length >= 0, index + length <= bytes.count else { throw ProtoWireError.truncated }
                fields.append(ProtoField(number: number, value: .bytes(Data(bytes[index..<(index + length)]))))
                index += length
            case 5:
                guard index + 4 <= bytes.count else { throw ProtoWireError.truncated }
                var value: UInt32 = 0
                for offset in 0..<4 { value |= UInt32(bytes[index + offset]) << (8 * UInt32(offset)) }
                index += 4
                fields.append(ProtoField(number: number, value: .fixed32(value)))
            case let type:
                throw ProtoWireError.badWireType(type)
            }
        }
        return fields
    }

    static func serialize(_ fields: [ProtoField]) -> Data {
        var output: [UInt8] = []
        for field in fields {
            switch field.value {
            case .varint(let value):
                appendVarint(UInt64(field.number << 3), to: &output)
                appendVarint(value, to: &output)
            case .fixed64(let value):
                appendVarint(UInt64(field.number << 3 | 1), to: &output)
                for offset in 0..<8 { output.append(UInt8(truncatingIfNeeded: value >> (8 * UInt64(offset)))) }
            case .bytes(let data):
                appendVarint(UInt64(field.number << 3 | 2), to: &output)
                appendVarint(UInt64(data.count), to: &output)
                output.append(contentsOf: data)
            case .fixed32(let value):
                appendVarint(UInt64(field.number << 3 | 5), to: &output)
                for offset in 0..<4 { output.append(UInt8(truncatingIfNeeded: value >> (8 * UInt32(offset)))) }
            }
        }
        return Data(output)
    }

    static func readVarint(_ bytes: [UInt8], _ index: inout Int) throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while true {
            guard index < bytes.count, shift < 70 else { throw ProtoWireError.truncated }
            let byte = bytes[index]
            index += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte < 0x80 { return result }
            shift += 7
        }
    }

    static func appendVarint(_ value: UInt64, to output: inout [UInt8]) {
        var value = value
        while value >= 0x80 {
            output.append(UInt8(value & 0x7F) | 0x80)
            value >>= 7
        }
        output.append(UInt8(value))
    }

    /// 詰めて並べた varint（packed repeated int64）を読む
    static func packedVarints(_ data: Data) throws -> [Int64] {
        let bytes = [UInt8](data)
        var index = 0
        var values: [Int64] = []
        while index < bytes.count { values.append(Int64(bitPattern: try readVarint(bytes, &index))) }
        return values
    }

    static func packVarints(_ values: [Int64]) -> Data {
        var output: [UInt8] = []
        for value in values { appendVarint(UInt64(bitPattern: value), to: &output) }
        return Data(output)
    }

    /// 入れ子のメッセージを書き換える（フィールド number の bytes を、変換したものに置き換える）
    static func mapMessages(_ fields: [ProtoField], number: Int,
                            _ transform: ([ProtoField]) throws -> [ProtoField]) rethrows -> [ProtoField] {
        try fields.map { field in
            guard field.number == number, let data = field.bytes,
                  let inner = try? parse(data) else { return field }
            return ProtoField(number: number, value: .bytes(serialize(try transform(inner))))
        }
    }
}
