import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// 4 ビット・5 ビットの重み（32 個ずつの「組」ごとに倍率と下限を持つ）。
///
/// 元の GGUF が 4 ビットなら 4 ビット、5 ビットなら 5 ビットのまま置く（8 ビットにすると 4B のモデルが 4GB になり、
/// 3.5GB ほどの iPad に入りきらないため）。値 = q × 倍率 + 下限（q は 0〜15 か 0〜31）。
///
/// 置き方（行列ごとに 3 つの面に分けて置く。CPU でまとめて計算しやすく、GPU でも続けて読める）:
/// - 下位 4 ビット: 1 バイトに 2 個（偶数番目が下の 4 ビット）。行 × 列 ÷ 2 バイト
/// - 5 ビット目（5 ビットのときだけ）: 1 バイトに 8 個（c 番目は c % 8 ビット目）。行 × 列 ÷ 8 バイト
/// - 倍率と下限: 組ごとに半精度 2 つ（倍率、下限）。行 × (列 ÷ 32) × 4 バイト
enum GGUFGroupQuant {
    static let groupSize = 32

    /// 元の形式から、置き方を決める。4 ビット以下の形式 → 4 ビット、5 ビットの形式 → 5 ビット、それ以外（6・8 ビット、16・32 ビット）→ nil（8 ビット）
    static func storage(for type: GGUFTensorType) -> GGUFMatrix.Storage? {
        switch type {
        case .q4_0, .q4_1, .q4_K, .iq4_NL, .iq4_XS, .q3_K, .q2_K, .iq3_S, .iq3_XXS, .iq2_XXS, .iq2_XS, .iq2_S:
            return .q4Groups
        case .q5_0, .q5_1, .q5_K:
            return .q5Groups
        default:
            return nil
        }
    }

    static func bits(_ storage: GGUFMatrix.Storage) -> Int { storage == .q5Groups ? 5 : 4 }

    static func nibbleBytes(rows: Int, columns: Int) -> Int { rows * columns / 2 }
    static func highBytes(rows: Int, columns: Int) -> Int { rows * columns / 8 }
    static func paramBytes(rows: Int, columns: Int) -> Int { rows * (columns / groupSize) * 4 }

    // MARK: - 半精度への変換

    /// Float → 半精度（最も近い値。丸めは偶数へ）
    static func half(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        var mantissa = bits & 0x7F_FFFF
        if value.isNaN { return sign | 0x7E00 }
        if exponent >= 31 { return sign | 0x7C00 }
        if exponent <= 0 {
            // 非正規化数（小さすぎるものは 0）
            guard exponent >= -10 else { return sign }
            mantissa |= 0x80_0000
            let shift = UInt32(14 - exponent)
            var result = mantissa >> shift
            let remainder = mantissa & ((1 << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if remainder > halfway || (remainder == halfway && (result & 1) != 0) { result += 1 }
            return sign | UInt16(result)
        }
        var result = UInt32(exponent) << 10 | (mantissa >> 13)
        let remainder = mantissa & 0x1FFF
        if remainder > 0x1000 || (remainder == 0x1000 && (result & 1) != 0) { result += 1 }
        return sign | UInt16(result)
    }

    // MARK: - 元の形式から作る

    /// 元の形式のうち、「組の中の値 = a × q + b」の形になっていて、そのまま写せるもの:
    /// (先頭の倍率などのバイト数, 5 ビット目のバイト数)
    private static func exactLayout(_ type: GGUFTensorType) -> (header: Int, high: Int)? {
        switch type {
        case .q4_0: return (2, 0)
        case .q4_1: return (4, 0)
        case .q5_0: return (2, 4)
        case .q5_1: return (4, 4)
        case .q4_K: return (16, 0)
        case .q5_K: return (16, 32)
        default: return nil
        }
    }

    /// 書き出し先（1 行ぶんずつ）
    struct RowTarget {
        var nibbles: UnsafeMutablePointer<UInt8>
        var high: UnsafeMutablePointer<UInt8>?
        var params: UnsafeMutablePointer<UInt16>
    }

    /// 作業用のメモリ（1 行ぶん）
    final class Scratch {
        let values: UnsafeMutablePointer<Float>
        let zeros: UnsafeMutablePointer<Float>
        let ones: UnsafeMutablePointer<Float>
        let block: UnsafeMutablePointer<UInt8>
        let capacity: Int

        init(columns: Int) {
            capacity = max(columns, 256)
            values = .allocate(capacity: capacity)
            zeros = .allocate(capacity: 256)
            ones = .allocate(capacity: 256)
            block = .allocate(capacity: 512)
        }

        deinit {
            values.deallocate()
            zeros.deallocate()
            ones.deallocate()
            block.deallocate()
        }
    }

    /// 元の形式の 1 行（columns 個）を、組の形にして target に書く
    static func packRow(type: GGUFTensorType, source: UnsafeRawPointer, columns: Int, bits: Int,
                        into target: RowTarget, scratch: Scratch) throws {
        guard columns % groupSize == 0, columns <= scratch.capacity else { throw GGUFError.invalid("group row width \(columns)") }
        let levels = Float(bits == 5 ? 31 : 15)
        try GGUFQuant.dequantize(type, source: source, count: columns, into: scratch.values)
        let groups = columns / groupSize
        if let layout = exactLayout(type), bits >= (layout.high > 0 ? 5 : 4) {
            // 量子化の整数を全部 0 にした組と、全部 1 にした組を戻すと、b と a + b がわかる（形式ごとの細かい並びを知らなくてよい）
            let block = type.block
            for first in Swift.stride(from: 0, to: columns, by: block.elements) {
                let bytes = source.advanced(by: first / block.elements * block.bytes)
                try gridOf(type, block: bytes, layout: layout, scratch: scratch)
                for local in Swift.stride(from: 0, to: block.elements, by: groupSize) {
                    let group = (first + local) / groupSize
                    let b = scratch.zeros[local]
                    let a = scratch.ones[local] - b
                    write(group: group, values: scratch.values + first + local, a: a, b: b, levels: levels, bits: bits, into: target)
                }
            }
        } else {
            // ほかの形式（2・3 ビットや IQ）: 組の最小と最大から目盛りを決める
            for group in 0..<groups {
                let values = scratch.values + group * groupSize
                var low = values[0], high = values[0]
                for index in 1..<groupSize {
                    low = min(low, values[index])
                    high = max(high, values[index])
                }
                write(group: group, values: values, a: (high - low) / levels, b: low, levels: levels, bits: bits, into: target)
            }
        }
    }

    /// 1 ブロックの「整数が全部 0」「全部 1」のときの値を scratch.zeros / ones に入れる
    private static func gridOf(_ type: GGUFTensorType, block: UnsafeRawPointer, layout: (header: Int, high: Int),
                               scratch: Scratch) throws {
        let size = type.block.bytes
        let copy = scratch.block
        copy.update(from: block.assumingMemoryBound(to: UInt8.self), count: size)
        (copy + layout.header).update(repeating: 0, count: size - layout.header)
        try GGUFQuant.dequantize(type, source: copy, count: type.block.elements, into: scratch.zeros)
        // 下位 4 ビットが 2 つとも 1、5 ビット目は 0
        (copy + layout.header + layout.high).update(repeating: 0x11, count: size - layout.header - layout.high)
        try GGUFQuant.dequantize(type, source: copy, count: type.block.elements, into: scratch.ones)
    }

    /// 1 つの組を書く（a, b は半精度に丸めてから q を決める）
    private static func write(group: Int, values: UnsafePointer<Float>, a: Float, b: Float, levels: Float, bits: Int,
                              into target: RowTarget) {
        // 半精度に入らない大きさ（本物の重みでは起きない）は、無限大にせず端の値にする
        let limit: Float = 65504
        let scaleBits = half(a.isFinite ? max(-limit, min(limit, a)) : 0)
        let minBits = half(b.isFinite ? max(-limit, min(limit, b)) : 0)
        target.params[group * 2] = scaleBits
        target.params[group * 2 + 1] = minBits
        let scale = GGUFQuant.half(scaleBits), offset = GGUFQuant.half(minBits)
        let inverse: Float = scale != 0 ? 1 / scale : 0
        let first = group * groupSize
        for index in 0..<groupSize {
            let q = Int(max(0, min(levels, ((values[index] - offset) * inverse).rounded())))
            let column = first + index
            let byte = column >> 1
            if column & 1 == 0 {
                target.nibbles[byte] = UInt8(q & 0xF)
            } else {
                target.nibbles[byte] |= UInt8(q & 0xF) << 4
            }
            if bits == 5, let high = target.high {
                if column & 7 == 0 { high[column >> 3] = 0 }
                if q >= 16 { high[column >> 3] |= UInt8(1 << (column & 7)) }
            }
        }
    }

    // MARK: - Float に戻す（1 行）

    /// 組の形の行列の 1 行を Float に戻す（埋め込みの表を引くときなど）
    static func dequantizeRow(_ matrix: GGUFMatrix, row: Int, into output: UnsafeMutablePointer<Float>) {
        let columns = matrix.columns
        let groups = columns / groupSize
        let nibbles = matrix.data.assumingMemoryBound(to: UInt8.self) + row * columns / 2
        let high = matrix.highBits.map { $0.assumingMemoryBound(to: UInt8.self) + row * columns / 8 }
        guard let params = matrix.groupParams?.assumingMemoryBound(to: UInt16.self) else {
            output.update(repeating: 0, count: columns)
            return
        }
        let rowParams = params + row * groups * 2
        for group in 0..<groups {
            let scale = GGUFQuant.half(rowParams[group * 2]), offset = GGUFQuant.half(rowParams[group * 2 + 1])
            for index in 0..<groupSize {
                let column = group * groupSize + index
                var q = Int(nibbles[column >> 1] >> UInt8((column & 1) * 4)) & 0xF
                if let high, (high[column >> 3] >> UInt8(column & 7)) & 1 != 0 { q += 16 }
                output[column] = Float(q) * scale + offset
            }
        }
    }

    // MARK: - 線形層（CPU）

    /// scratch の大きさ（Float の個数）に入る行数
    static func blockRows(columns: Int, capacity: Int) -> Int {
        let groups = columns / groupSize
        return max(1, min(512, (capacity - groups - 64) / (2 * columns + 3 * groups)))
    }

    /// output[n, r] = Σ_c X[n, c] · W[r, c]（range の行だけ。stride は output の 1 行の長さ）
    static func linear(_ matrix: GGUFMatrix, _ range: Range<Int>, input: UnsafePointer<Float>, count: Int,
                       output: UnsafeMutablePointer<Float>, stride: Int, scratch: UnsafeMutablePointer<Float>, capacity: Int) {
        #if canImport(Accelerate)
        accelerated(matrix, range, input: input, count: count, output: output, stride: stride, scratch: scratch, capacity: capacity)
        #else
        let row = scratch
        for r in range {
            dequantizeRow(matrix, row: r, into: row)
            for n in 0..<count {
                let x = input + n * matrix.columns
                var sum: Float = 0
                for c in 0..<matrix.columns { sum += row[c] * x[c] }
                output[n * stride + r] = sum
            }
        }
        #endif
    }

    #if canImport(Accelerate)
    private static let onesVector = [Float](repeating: 1, count: 32)

    /// 整数のまま Float に直し（q）、組ごとに Σ q·x を出してから、倍率と下限を掛けて足す:
    /// y = Σ_組 (倍率 × Σ q·x + 下限 × Σ x)。どの手順も Accelerate でまとめて計算する（最適化なしのビルドでも速い）
    private static func accelerated(_ matrix: GGUFMatrix, _ range: Range<Int>, input: UnsafePointer<Float>, count: Int,
                                    output: UnsafeMutablePointer<Float>, stride: Int,
                                    scratch: UnsafeMutablePointer<Float>, capacity: Int) {
        let columns = matrix.columns
        let groups = columns / groupSize
        let blockRows = blockRows(columns: columns, capacity: capacity)
        let q = scratch
        let work = scratch + blockRows * columns
        let params = work + blockRows * columns
        let dots = params + 2 * blockRows * groups
        let sums = dots + blockRows * groups
        guard let groupParams = matrix.groupParams else { return }
        onesVector.withUnsafeBufferPointer { ones in
            var row = range.lowerBound
            while row < range.upperBound {
                let rows = min(blockRows, range.upperBound - row)
                unpack(matrix, firstRow: row, rows: rows, into: q, work: work)
                // 倍率と下限（半精度 → Float）
                var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: groupParams + row * groups * 4), height: 1,
                                           width: vImagePixelCount(rows * groups * 2), rowBytes: rows * groups * 4)
                var target = vImage_Buffer(data: params, height: 1, width: vImagePixelCount(rows * groups * 2), rowBytes: rows * groups * 8)
                vImageConvert_Planar16FtoPlanarF(&source, &target, vImage_Flags(kvImageNoFlags))
                for n in 0..<count {
                    let x = input + n * columns
                    // 組ごとの Σ x
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(groups), 32, 1, x, 32, ones.baseAddress!, 1, 0, sums, 1)
                    // q ⊙ x を行ごとに → 組ごとに足す
                    for r in 0..<rows {
                        vDSP_vmul(q + r * columns, 1, x, 1, work + r * columns, 1, vDSP_Length(columns))
                    }
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rows * groups), 32, 1, work, 32, ones.baseAddress!, 1, 0, dots, 1)
                    for r in 0..<rows {
                        var scaled: Float = 0, shifted: Float = 0
                        vDSP_dotpr(params + r * groups * 2, 2, dots + r * groups, 1, &scaled, vDSP_Length(groups))
                        vDSP_dotpr(params + r * groups * 2 + 1, 2, sums, 1, &shifted, vDSP_Length(groups))
                        output[n * stride + row + r] = scaled + shifted
                    }
                }
                row += rows
            }
        }
    }

    /// rows 行の q（0〜15 か 0〜31）を Float にする。work は rows × columns 個
    private static func unpack(_ matrix: GGUFMatrix, firstRow: Int, rows: Int, into q: UnsafeMutablePointer<Float>,
                               work: UnsafeMutablePointer<Float>) {
        let columns = matrix.columns
        let total = rows * columns
        let pairs = total / 2
        let packed = matrix.data.assumingMemoryBound(to: UInt8.self) + firstRow * columns / 2
        let low = work, highNibble = work + pairs
        vDSP_vfltu8(packed, 1, low, 1, vDSP_Length(pairs))
        var sixteenth: Float = 1.0 / 16, minusSixteen: Float = -16
        vDSP_vsmul(low, 1, &sixteenth, highNibble, 1, vDSP_Length(pairs))
        var length = Int32(pairs)
        vvfloorf(highNibble, highNibble, &length)
        // 下の 4 ビット = バイト − 16 × 上の 4 ビット
        vDSP_vsma(highNibble, 1, &minusSixteen, low, 1, low, 1, vDSP_Length(pairs))
        cblas_scopy(Int32(pairs), low, 1, q, 2)
        cblas_scopy(Int32(pairs), highNibble, 1, q + 1, 2)
        guard let highBits = matrix.highBits else { return }
        // 5 ビット目: c 番目は (c / 8) バイト目の (c % 8) ビット目。floor で 1 ビットずつ取り出す
        let eighths = total / 8
        let bytes = highBits.assumingMemoryBound(to: UInt8.self) + firstRow * columns / 8
        var current = work, next = work + eighths
        vDSP_vfltu8(bytes, 1, current, 1, vDSP_Length(eighths))
        var oneHalf: Float = 0.5, minusTwo: Float = -2, sixteen: Float = 16
        var count = Int32(eighths)
        for bit in 0..<8 {
            vDSP_vsmul(current, 1, &oneHalf, next, 1, vDSP_Length(eighths))
            vvfloorf(next, next, &count)
            // このビット = 今の値 − 2 × 次の値（current に入れる）
            vDSP_vsma(next, 1, &minusTwo, current, 1, current, 1, vDSP_Length(eighths))
            vDSP_vsma(current, 1, &sixteen, q + bit, 8, q + bit, 8, vDSP_Length(eighths))
            swap(&current, &next)
        }
    }
    #endif
}
