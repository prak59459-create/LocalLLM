import Foundation

/// GGUF の量子化された重みを Float に戻す（llama.cpp の dequantize_row_* と同じ計算）。
///
/// どの形式も「ブロック」（32 個または 256 個）ごとに独立していて、1 行はブロックの整数倍になっている。
enum GGUFQuant {
    typealias Source = UnsafePointer<UInt8>
    typealias Output = UnsafeMutablePointer<Float>

    /// source から count 個（ブロックの整数倍）を output に書く。
    static func dequantize(_ type: GGUFTensorType, source: UnsafeRawPointer, count: Int, into output: Output) throws {
        let block = type.block
        guard count % block.elements == 0 else { throw GGUFError.invalid("\(type.name) block size") }
        let blocks = count / block.elements
        let bytes = source.assumingMemoryBound(to: UInt8.self)
        switch type {
        case .f32:
            UnsafeMutableRawPointer(output).copyMemory(from: source, byteCount: count * 4)
        case .f16:
            halfToFloat(source, count: count, into: output)
        case .bf16:
            for index in 0..<count {
                let bits = UInt32(bytes[index * 2]) | UInt32(bytes[index * 2 + 1]) << 8
                output[index] = Float(bitPattern: bits << 16)
            }
        case .q4_0: for b in 0..<blocks { q4_0(bytes + b * 18, output + b * 32) }
        case .q4_1: for b in 0..<blocks { q4_1(bytes + b * 20, output + b * 32) }
        case .q5_0: for b in 0..<blocks { q5_0(bytes + b * 22, output + b * 32) }
        case .q5_1: for b in 0..<blocks { q5_1(bytes + b * 24, output + b * 32) }
        case .q8_0: for b in 0..<blocks { q8_0(bytes + b * 34, output + b * 32) }
        case .q2_K: for b in 0..<blocks { q2_K(bytes + b * 84, output + b * 256) }
        case .q3_K: for b in 0..<blocks { q3_K(bytes + b * 110, output + b * 256) }
        case .q4_K: for b in 0..<blocks { q4_K(bytes + b * 144, output + b * 256) }
        case .q5_K: for b in 0..<blocks { q5_K(bytes + b * 176, output + b * 256) }
        case .q6_K: for b in 0..<blocks { q6_K(bytes + b * 210, output + b * 256) }
        case .iq4_NL: for b in 0..<blocks { iq4_NL(bytes + b * 18, output + b * 32) }
        case .iq4_XS: for b in 0..<blocks { iq4_XS(bytes + b * 136, output + b * 256) }
        case .iq2_XXS: for b in 0..<blocks { iq2_XXS(bytes + b * 66, output + b * 256) }
        case .iq2_XS: for b in 0..<blocks { iq2_XS(bytes + b * 74, output + b * 256) }
        case .iq2_S: for b in 0..<blocks { iq2_S(bytes + b * 82, output + b * 256) }
        case .iq3_XXS: for b in 0..<blocks { iq3_XXS(bytes + b * 98, output + b * 256) }
        case .iq3_S: for b in 0..<blocks { iq3_S(bytes + b * 110, output + b * 256) }
        default:
            throw GGUFError.unsupportedTensorType(type.name)
        }
    }

    // MARK: - 半精度

    /// 16 ビット浮動小数（IEEE 754 binary16）→ Float
    @inline(__always)
    static func half(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = UInt32(bits >> 10) & 0x1F
        let mantissa = UInt32(bits & 0x3FF)
        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }
            // 非正規化数
            let value = Float(mantissa) * 5.9604644775390625e-8   // 2^-24
            return sign != 0 ? -value : value
        }
        if exponent == 31 {
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }
        return Float(bitPattern: sign | ((exponent + 112) << 23) | (mantissa << 13))
    }

    @inline(__always)
    private static func half(_ p: Source, _ offset: Int) -> Float {
        half(UInt16(p[offset]) | UInt16(p[offset + 1]) << 8)
    }

    static func halfToFloat(_ source: UnsafeRawPointer, count: Int, into output: Output) {
        let bytes = source.assumingMemoryBound(to: UInt8.self)
        let table = halfTable
        table.withUnsafeBufferPointer { lookup in
            for index in 0..<count {
                let bits = Int(bytes[index * 2]) | Int(bytes[index * 2 + 1]) << 8
                output[index] = lookup[bits]
            }
        }
    }

    /// 65536 通りの半精度の値（1 個ずつ計算するより速い）
    private static let halfTable: [Float] = (0..<65536).map { half(UInt16($0)) }

    // MARK: - 32 個のブロック

    private static func q4_0(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2
        for j in 0..<16 {
            let byte = Int(qs[j])
            y[j] = Float((byte & 0xF) - 8) * d
            y[j + 16] = Float((byte >> 4) - 8) * d
        }
    }

    private static func q4_1(_ p: Source, _ y: Output) {
        let d = half(p, 0), m = half(p, 2)
        let qs = p + 4
        for j in 0..<16 {
            let byte = Int(qs[j])
            y[j] = Float(byte & 0xF) * d + m
            y[j + 16] = Float(byte >> 4) * d + m
        }
    }

    private static func q5_0(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qh = UInt32(p[2]) | UInt32(p[3]) << 8 | UInt32(p[4]) << 16 | UInt32(p[5]) << 24
        let qs = p + 6
        for j in 0..<16 {
            let byte = Int(qs[j])
            let high0 = Int((qh >> UInt32(j)) & 1) << 4
            let high1 = Int((qh >> UInt32(j + 16)) & 1) << 4
            y[j] = Float(((byte & 0xF) | high0) - 16) * d
            y[j + 16] = Float(((byte >> 4) | high1) - 16) * d
        }
    }

    private static func q5_1(_ p: Source, _ y: Output) {
        let d = half(p, 0), m = half(p, 2)
        let qh = UInt32(p[4]) | UInt32(p[5]) << 8 | UInt32(p[6]) << 16 | UInt32(p[7]) << 24
        let qs = p + 8
        for j in 0..<16 {
            let byte = Int(qs[j])
            let high0 = Int((qh >> UInt32(j)) & 1) << 4
            let high1 = Int((qh >> UInt32(j + 16)) & 1) << 4
            y[j] = Float((byte & 0xF) | high0) * d + m
            y[j + 16] = Float((byte >> 4) | high1) * d + m
        }
    }

    private static func q8_0(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2
        for j in 0..<32 { y[j] = Float(Int8(bitPattern: qs[j])) * d }
    }

    private static let iq4Values: [Float] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

    private static func iq4_NL(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2
        iq4Values.withUnsafeBufferPointer { values in
            for j in 0..<16 {
                let byte = Int(qs[j])
                y[j] = d * values[byte & 0xF]
                y[j + 16] = d * values[byte >> 4]
            }
        }
    }

    // MARK: - 256 個のブロック（K 量子化）

    private static func q2_K(_ p: Source, _ y: Output) {
        let scales = p, qs = p + 16
        let d = half(p, 80), dmin = half(p, 82)
        for half in 0..<2 {
            let q = qs + half * 32
            for shiftIndex in 0..<4 {
                let shift = UInt8(shiftIndex * 2)
                for part in 0..<2 {
                    let sub = half * 8 + shiftIndex * 2 + part
                    let scale = scales[sub]
                    let dl = d * Float(scale & 0xF), ml = dmin * Float(scale >> 4)
                    let out = y + sub * 16
                    let base = q + part * 16
                    for l in 0..<16 { out[l] = dl * Float((base[l] >> shift) & 3) - ml }
                }
            }
        }
    }

    private static func q3_K(_ p: Source, _ y: Output) {
        let hmask = p, qs = p + 32, packed = p + 96
        let d = half(p, 108)
        for sub in 0..<16 {
            let low = sub < 8 ? Int(packed[sub] & 0xF) : Int(packed[sub - 8] >> 4)
            let high = Int((packed[8 + sub % 4] >> UInt8(2 * (sub / 4))) & 3)
            let dl = d * Float((low | (high << 4)) - 32)
            let out = y + sub * 16
            for l in 0..<16 {
                let element = sub * 16 + l
                let column = element % 32
                let ql = Int((qs[(element / 128) * 32 + column] >> UInt8(2 * ((element % 128) / 32))) & 3)
                let bit = Int((hmask[column] >> UInt8(element / 32)) & 1)
                out[l] = dl * Float(ql - ((bit ^ 1) << 2))
            }
        }
    }

    /// Q4_K / Q5_K の 6 ビットの scale と min（8 組）
    @inline(__always)
    private static func scaleMin(_ scales: Source, _ sub: Int) -> (Float, Float) {
        if sub < 4 {
            return (Float(scales[sub] & 63), Float(scales[sub + 4] & 63))
        }
        let scale = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4)
        let minimum = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4)
        return (Float(scale), Float(minimum))
    }

    private static func q4_K(_ p: Source, _ y: Output) {
        let d = half(p, 0), dmin = half(p, 2)
        let scales = p + 4, qs = p + 16
        for group in 0..<4 {
            let (s0, m0) = scaleMin(scales, group * 2)
            let (s1, m1) = scaleMin(scales, group * 2 + 1)
            let d0 = d * s0, min0 = dmin * m0, d1 = d * s1, min1 = dmin * m1
            let q = qs + group * 32
            let out = y + group * 64
            for l in 0..<32 {
                let byte = q[l]
                out[l] = d0 * Float(byte & 0xF) - min0
                out[l + 32] = d1 * Float(byte >> 4) - min1
            }
        }
    }

    private static func q5_K(_ p: Source, _ y: Output) {
        let d = half(p, 0), dmin = half(p, 2)
        let scales = p + 4, qh = p + 16, qs = p + 48
        for group in 0..<4 {
            let (s0, m0) = scaleMin(scales, group * 2)
            let (s1, m1) = scaleMin(scales, group * 2 + 1)
            let d0 = d * s0, min0 = dmin * m0, d1 = d * s1, min1 = dmin * m1
            let q = qs + group * 32
            let out = y + group * 64
            let bit0 = UInt8(group * 2), bit1 = UInt8(group * 2 + 1)
            for l in 0..<32 {
                let byte = q[l], high = qh[l]
                out[l] = d0 * Float((byte & 0xF) | (((high >> bit0) & 1) << 4)) - min0
                out[l + 32] = d1 * Float((byte >> 4) | (((high >> bit1) & 1) << 4)) - min1
            }
        }
    }

    private static func q6_K(_ p: Source, _ y: Output) {
        let ql = p, qh = p + 128, scales = p + 192
        let d = half(p, 208)
        for half in 0..<2 {
            let l4 = ql + half * 64, h2 = qh + half * 32
            let out = y + half * 128
            let sc = scales + half * 8
            for l in 0..<32 {
                let scaleIndex = l / 16
                let high = h2[l]
                let q1 = Int((l4[l] & 0xF) | ((high & 3) << 4)) - 32
                let q2 = Int((l4[l + 32] & 0xF) | (((high >> 2) & 3) << 4)) - 32
                let q3 = Int((l4[l] >> 4) | (((high >> 4) & 3) << 4)) - 32
                let q4 = Int((l4[l + 32] >> 4) | (((high >> 6) & 3) << 4)) - 32
                out[l] = d * Float(Int8(bitPattern: sc[scaleIndex])) * Float(q1)
                out[l + 32] = d * Float(Int8(bitPattern: sc[scaleIndex + 2])) * Float(q2)
                out[l + 64] = d * Float(Int8(bitPattern: sc[scaleIndex + 4])) * Float(q3)
                out[l + 96] = d * Float(Int8(bitPattern: sc[scaleIndex + 6])) * Float(q4)
            }
        }
    }

    private static func iq4_XS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let scalesHigh = UInt32(p[2]) | UInt32(p[3]) << 8
        let scalesLow = p + 4, qs = p + 8
        iq4Values.withUnsafeBufferPointer { values in
            for sub in 0..<8 {
                let low = Int((scalesLow[sub / 2] >> UInt8(4 * (sub % 2))) & 0xF)
                let high = Int((scalesHigh >> UInt32(2 * sub)) & 3)
                let dl = d * Float((low | (high << 4)) - 32)
                let q = qs + sub * 16
                let out = y + sub * 32
                for j in 0..<16 {
                    let byte = Int(q[j])
                    out[j] = dl * values[byte & 0xF]
                    out[j + 16] = dl * values[byte >> 4]
                }
            }
        }
    }

    // MARK: - 256 個のブロック（IQ2 / IQ3。表を引く）

    @inline(__always)
    private static func u32(_ p: Source, _ offset: Int) -> UInt32 {
        UInt32(p[offset]) | UInt32(p[offset + 1]) << 8 | UInt32(p[offset + 2]) << 16 | UInt32(p[offset + 3]) << 24
    }

    /// 表の 1 行（count 個）に符号と倍率を掛けて書く。signs のビット firstBit から順に、1 なら負。
    @inline(__always)
    private static func emit(_ grid: UnsafeBufferPointer<Float>, row: Int, count: Int, signs: UInt8, firstBit: Int,
                             scale: Float, _ out: Output) {
        let base = row * count
        for j in 0..<count {
            let negative = (signs >> UInt8(firstBit + j)) & 1 != 0
            out[j] = negative ? -(scale * grid[base + j]) : scale * grid[base + j]
        }
    }

    private static func iq2_XXS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let signTable = GGUFQuantTables.signs
        GGUFQuantTables.iq2xxs.withUnsafeBufferPointer { grid in
            for group in 0..<8 {
                let indices = p + 2 + group * 8
                let word = u32(p, 2 + group * 8 + 4)
                let db = d * (0.5 + Float(word >> 28)) * 0.25
                for k in 0..<4 {
                    let signs = signTable[Int((word >> UInt32(7 * k)) & 127)]
                    emit(grid, row: Int(indices[k]), count: 8, signs: signs, firstBit: 0, scale: db, y + group * 32 + k * 8)
                }
            }
        }
    }

    private static func iq2_XS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2, scales = p + 66
        let signTable = GGUFQuantTables.signs
        GGUFQuantTables.iq2xs.withUnsafeBufferPointer { grid in
            for i in 0..<32 {
                let q = Int(qs[i * 2]) | Int(qs[i * 2 + 1]) << 8
                let scale = (scales[i / 4] >> UInt8(4 * ((i / 2) % 2))) & 0xF
                let db = d * (0.5 + Float(scale)) * 0.25
                emit(grid, row: q & 511, count: 8, signs: signTable[q >> 9], firstBit: 0, scale: db, y + i * 8)
            }
        }
    }

    private static func iq2_S(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2, signs = p + 34, qh = p + 66, scales = p + 74
        GGUFQuantTables.iq2s.withUnsafeBufferPointer { grid in
            for i in 0..<32 {
                let high = Int((qh[i / 4] >> UInt8(2 * (i % 4))) & 3)
                let scale = (scales[i / 4] >> UInt8(4 * ((i / 2) % 2))) & 0xF
                let db = d * (0.5 + Float(scale)) * 0.25
                emit(grid, row: Int(qs[i]) | (high << 8), count: 8, signs: signs[i], firstBit: 0, scale: db, y + i * 8)
            }
        }
    }

    private static func iq3_XXS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2
        let signTable = GGUFQuantTables.signs
        GGUFQuantTables.iq3xxs.withUnsafeBufferPointer { grid in
            for group in 0..<8 {
                let word = u32(p, 66 + group * 4)
                let db = d * (0.5 + Float(word >> 28)) * 0.5
                for k in 0..<4 {
                    let signs = signTable[Int((word >> UInt32(7 * k)) & 127)]
                    let out = y + group * 32 + k * 8
                    emit(grid, row: Int(qs[group * 8 + k * 2]), count: 4, signs: signs, firstBit: 0, scale: db, out)
                    emit(grid, row: Int(qs[group * 8 + k * 2 + 1]), count: 4, signs: signs, firstBit: 4, scale: db, out + 4)
                }
            }
        }
    }

    private static func iq3_S(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2, qh = p + 66, signs = p + 74, scales = p + 106
        GGUFQuantTables.iq3s.withUnsafeBufferPointer { grid in
            for i in 0..<64 {
                let scale = (scales[i / 16] >> UInt8(4 * ((i / 8) % 2))) & 0xF
                let db = d * Float(1 + 2 * Int(scale))
                let high = Int((qh[i / 8] >> UInt8(i % 8)) & 1)
                emit(grid, row: Int(qs[i]) | (high << 8), count: 4, signs: signs[i / 2], firstBit: (i % 2) * 4, scale: db, y + i * 4)
            }
        }
    }
}
