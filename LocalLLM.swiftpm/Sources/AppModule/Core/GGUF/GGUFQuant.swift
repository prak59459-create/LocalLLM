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
                let bits: UInt32 = UInt32(bytes[index * 2]) | (UInt32(bytes[index * 2 + 1]) << 8)
                output[index] = Float(bitPattern: bits << 16)
            }
        default:
            if block.elements == 32 {
                try smallBlocks(type, bytes, blocks: blocks, output)
            } else {
                try largeBlocks(type, bytes, blocks: blocks, output)
            }
        }
    }

    /// 32 個ずつのブロックの形式
    private static func smallBlocks(_ type: GGUFTensorType, _ bytes: Source, blocks: Int, _ output: Output) throws {
        let size = type.block.bytes
        let decode: (Source, Output) -> Void
        switch type {
        case .q4_0: decode = q4_0
        case .q4_1: decode = q4_1
        case .q5_0: decode = q5_0
        case .q5_1: decode = q5_1
        case .q8_0: decode = q8_0
        case .iq4_NL: decode = iq4_NL
        default: throw GGUFError.unsupportedTensorType(type.name)
        }
        for index in 0..<blocks { decode(bytes + index * size, output + index * 32) }
    }

    /// 256 個ずつのブロックの形式
    private static func largeBlocks(_ type: GGUFTensorType, _ bytes: Source, blocks: Int, _ output: Output) throws {
        let size = type.block.bytes
        let decode: (Source, Output) -> Void
        switch type {
        case .q2_K: decode = q2_K
        case .q3_K: decode = q3_K
        case .q4_K: decode = q4_K
        case .q5_K: decode = q5_K
        case .q6_K: decode = q6_K
        case .iq4_XS: decode = iq4_XS
        case .iq2_XXS: decode = iq2_XXS
        case .iq2_XS: decode = iq2_XS
        case .iq2_S: decode = iq2_S
        case .iq3_XXS: decode = iq3_XXS
        case .iq3_S: decode = iq3_S
        default: throw GGUFError.unsupportedTensorType(type.name)
        }
        for index in 0..<blocks { decode(bytes + index * size, output + index * 256) }
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
        let qh: UInt32 = u32(p, 2)
        let qs = p + 6
        for j in 0..<16 {
            let byte = Int(qs[j])
            let high0: Int = Int((qh >> UInt32(j)) & 1) << 4
            let high1: Int = Int((qh >> UInt32(j + 16)) & 1) << 4
            let first: Int = (byte & 0xF) | high0
            let second: Int = (byte >> 4) | high1
            y[j] = Float(first - 16) * d
            y[j + 16] = Float(second - 16) * d
        }
    }

    private static func q5_1(_ p: Source, _ y: Output) {
        let d = half(p, 0), m = half(p, 2)
        let qh: UInt32 = u32(p, 4)
        let qs = p + 8
        for j in 0..<16 {
            let byte = Int(qs[j])
            let high0: Int = Int((qh >> UInt32(j)) & 1) << 4
            let high1: Int = Int((qh >> UInt32(j + 16)) & 1) << 4
            let first: Int = (byte & 0xF) | high0
            let second: Int = (byte >> 4) | high1
            y[j] = Float(first) * d + m
            y[j + 16] = Float(second) * d + m
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
        for sub in 0..<16 {
            // 16 個ずつ。sub = 半分 × 8 + ずらす量 × 2 + 前後
            let halfIndex: Int = sub / 8
            let shift: Int = ((sub % 8) / 2) * 2
            let part: Int = sub % 2
            let scale = Int(scales[sub])
            let dl: Float = d * Float(scale & 0xF)
            let ml: Float = dmin * Float(scale >> 4)
            let base = qs + (halfIndex * 32 + part * 16)
            let out = y + sub * 16
            for l in 0..<16 {
                let q: Int = (Int(base[l]) >> shift) & 3
                out[l] = dl * Float(q) - ml
            }
        }
    }

    private static func q3_K(_ p: Source, _ y: Output) {
        let hmask = p, qs = p + 32, packed = p + 96
        let d = half(p, 108)
        for sub in 0..<16 {
            let low: Int = sub < 8 ? Int(packed[sub]) & 0xF : Int(packed[sub - 8]) >> 4
            let highShift: Int = 2 * (sub / 4)
            let high: Int = (Int(packed[8 + sub % 4]) >> highShift) & 3
            let dl: Float = d * Float((low | (high << 4)) - 32)
            let out = y + sub * 16
            for l in 0..<16 {
                let element: Int = sub * 16 + l
                let column: Int = element % 32
                let byteIndex: Int = (element / 128) * 32 + column
                let shift: Int = 2 * ((element % 128) / 32)
                let ql: Int = (Int(qs[byteIndex]) >> shift) & 3
                let bit: Int = (Int(hmask[column]) >> (element / 32)) & 1
                let q: Int = ql - ((bit ^ 1) << 2)
                out[l] = dl * Float(q)
            }
        }
    }

    /// Q4_K / Q5_K の 6 ビットの scale と min（8 組）
    @inline(__always)
    private static func scaleMin(_ scales: Source, _ sub: Int) -> (Float, Float) {
        if sub < 4 {
            let scale: Int = Int(scales[sub]) & 63
            let minimum: Int = Int(scales[sub + 4]) & 63
            return (Float(scale), Float(minimum))
        }
        let packed = Int(scales[sub + 4])
        let scaleHigh: Int = (Int(scales[sub - 4]) >> 6) << 4
        let minimumHigh: Int = (Int(scales[sub]) >> 6) << 4
        let scale: Int = (packed & 0xF) | scaleHigh
        let minimum: Int = (packed >> 4) | minimumHigh
        return (Float(scale), Float(minimum))
    }

    private static func q4_K(_ p: Source, _ y: Output) {
        let d = half(p, 0), dmin = half(p, 2)
        let scales = p + 4, qs = p + 16
        for group in 0..<4 {
            let (s0, m0) = scaleMin(scales, group * 2)
            let (s1, m1) = scaleMin(scales, group * 2 + 1)
            let d0: Float = d * s0, min0: Float = dmin * m0
            let d1: Float = d * s1, min1: Float = dmin * m1
            let q = qs + group * 32
            let out = y + group * 64
            for l in 0..<32 {
                let byte = Int(q[l])
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
            let d0: Float = d * s0, min0: Float = dmin * m0
            let d1: Float = d * s1, min1: Float = dmin * m1
            let q = qs + group * 32
            let out = y + group * 64
            let bit0: Int = group * 2, bit1: Int = group * 2 + 1
            for l in 0..<32 {
                let byte = Int(q[l]), high = Int(qh[l])
                let top0: Int = ((high >> bit0) & 1) << 4
                let top1: Int = ((high >> bit1) & 1) << 4
                let first: Int = (byte & 0xF) | top0
                let second: Int = (byte >> 4) | top1
                out[l] = d0 * Float(first) - min0
                out[l + 32] = d1 * Float(second) - min1
            }
        }
    }

    private static func q6_K(_ p: Source, _ y: Output) {
        let ql = p, qh = p + 128, scales = p + 192
        let d = half(p, 208)
        for halfIndex in 0..<2 {
            let l4 = ql + halfIndex * 64, h2 = qh + halfIndex * 32
            let out = y + halfIndex * 128
            let sc = scales + halfIndex * 8
            for l in 0..<32 {
                let scaleIndex: Int = l / 16
                let high = Int(h2[l]), lowA = Int(l4[l]), lowB = Int(l4[l + 32])
                let q1: Int = ((lowA & 0xF) | ((high & 3) << 4)) - 32
                let q2: Int = ((lowB & 0xF) | (((high >> 2) & 3) << 4)) - 32
                let q3: Int = ((lowA >> 4) | (((high >> 4) & 3) << 4)) - 32
                let q4: Int = ((lowB >> 4) | (((high >> 6) & 3) << 4)) - 32
                let s1: Float = d * Float(Int8(bitPattern: sc[scaleIndex]))
                let s2: Float = d * Float(Int8(bitPattern: sc[scaleIndex + 2]))
                let s3: Float = d * Float(Int8(bitPattern: sc[scaleIndex + 4]))
                let s4: Float = d * Float(Int8(bitPattern: sc[scaleIndex + 6]))
                out[l] = s1 * Float(q1)
                out[l + 32] = s2 * Float(q2)
                out[l + 64] = s3 * Float(q3)
                out[l + 96] = s4 * Float(q4)
            }
        }
    }

    private static func iq4_XS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let scalesHigh: Int = Int(p[2]) | (Int(p[3]) << 8)
        let scalesLow = p + 4, qs = p + 8
        iq4Values.withUnsafeBufferPointer { values in
            for sub in 0..<8 {
                let lowShift: Int = 4 * (sub % 2)
                let low: Int = (Int(scalesLow[sub / 2]) >> lowShift) & 0xF
                let high: Int = (scalesHigh >> (2 * sub)) & 3
                let dl: Float = d * Float((low | (high << 4)) - 32)
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
        let b0 = UInt32(p[offset]), b1 = UInt32(p[offset + 1])
        let b2 = UInt32(p[offset + 2]), b3 = UInt32(p[offset + 3])
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// 表の 1 行（count 個）に符号と倍率を掛けて書く。signs のビット firstBit から順に、1 なら負。
    @inline(__always)
    private static func emit(_ grid: UnsafeBufferPointer<Float>, row: Int, count: Int, signs: Int, firstBit: Int,
                             scale: Float, _ out: Output) {
        let base = row * count
        for j in 0..<count {
            let value: Float = scale * grid[base + j]
            let negative: Bool = (signs >> (firstBit + j)) & 1 != 0
            out[j] = negative ? -value : value
        }
    }

    private static func iq2_XXS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let signTable = GGUFQuantTables.signs
        GGUFQuantTables.iq2xxs.withUnsafeBufferPointer { grid in
            for group in 0..<8 {
                let indices = p + 2 + group * 8
                let word = Int(u32(p, 2 + group * 8 + 4))
                let db: Float = d * (0.5 + Float(word >> 28)) * 0.25
                for k in 0..<4 {
                    let signs = Int(signTable[(word >> (7 * k)) & 127])
                    emit(grid, row: Int(indices[k]), count: 8, signs: signs, firstBit: 0, scale: db, y + group * 32 + k * 8)
                }
            }
        }
    }

    /// IQ2_XS / IQ2_S の 16 個ごとの倍率（4 ビット）
    @inline(__always)
    private static func iq2Scale(_ scales: Source, _ index: Int, _ d: Float) -> Float {
        let shift: Int = 4 * ((index / 2) % 2)
        let scale: Int = (Int(scales[index / 4]) >> shift) & 0xF
        return d * (0.5 + Float(scale)) * 0.25
    }

    private static func iq2_XS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2, scales = p + 66
        let signTable = GGUFQuantTables.signs
        GGUFQuantTables.iq2xs.withUnsafeBufferPointer { grid in
            for i in 0..<32 {
                let q: Int = Int(qs[i * 2]) | (Int(qs[i * 2 + 1]) << 8)
                let db = iq2Scale(scales, i, d)
                emit(grid, row: q & 511, count: 8, signs: Int(signTable[q >> 9]), firstBit: 0, scale: db, y + i * 8)
            }
        }
    }

    private static func iq2_S(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2, signs = p + 34, qh = p + 66, scales = p + 74
        GGUFQuantTables.iq2s.withUnsafeBufferPointer { grid in
            for i in 0..<32 {
                let highShift: Int = 2 * (i % 4)
                let high: Int = (Int(qh[i / 4]) >> highShift) & 3
                let db = iq2Scale(scales, i, d)
                emit(grid, row: Int(qs[i]) | (high << 8), count: 8, signs: Int(signs[i]), firstBit: 0, scale: db, y + i * 8)
            }
        }
    }

    private static func iq3_XXS(_ p: Source, _ y: Output) {
        let d = half(p, 0)
        let qs = p + 2
        let signTable = GGUFQuantTables.signs
        GGUFQuantTables.iq3xxs.withUnsafeBufferPointer { grid in
            for group in 0..<8 {
                let word = Int(u32(p, 66 + group * 4))
                let db: Float = d * (0.5 + Float(word >> 28)) * 0.5
                for k in 0..<4 {
                    let signs = Int(signTable[(word >> (7 * k)) & 127])
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
                let scaleShift: Int = 4 * ((i / 8) % 2)
                let scale: Int = (Int(scales[i / 16]) >> scaleShift) & 0xF
                let db: Float = d * Float(1 + 2 * scale)
                let high: Int = (Int(qh[i / 8]) >> (i % 8)) & 1
                emit(grid, row: Int(qs[i]) | (high << 8), count: 4, signs: Int(signs[i / 2]), firstBit: (i % 2) * 4, scale: db, y + i * 4)
            }
        }
    }
}
