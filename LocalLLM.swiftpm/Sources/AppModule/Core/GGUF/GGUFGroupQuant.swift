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
    static func half(_ value: Float) -> UInt16 { Half.fromFloat(value) }

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

    /// Accelerate での計算の置き場所の、1 行あたりの大きさ（Float の個数）
    static func floatsPerRow(columns: Int, five: Bool) -> Int {
        let half = columns / 2, groups = columns / groupSize
        // バイト・上の 4 ビット・積・x の並び 2 つ（5 ビットは + 4）、倍率と下限 2・倍率・下限・組ごとの和・倍率 × 和
        return (five ? 9 : 5) * half + 6 * groups
    }

    /// scratch の大きさ（Float の個数）に入る行数
    static func blockRows(columns: Int, capacity: Int, five: Bool) -> Int {
        let groups = columns / groupSize
        return max(1, min(512, (capacity - groups - 64) / floatsPerRow(columns: columns, five: five)))
    }

    /// output[n, r] = Σ_c X[n, c] · W[r, c]（range の行だけ。stride は output の 1 行の長さ）
    static func linear(_ matrix: GGUFMatrix, _ range: Range<Int>, input: UnsafePointer<Float>, count: Int,
                       output: UnsafeMutablePointer<Float>, stride: Int, scratch: UnsafeMutablePointer<Float>, capacity: Int) {
        #if canImport(Accelerate)
        accelerated(matrix, range, input: input, count: count, output: output, stride: stride, scratch: scratch, capacity: capacity)
        #else
        simple(matrix, range, input: input, count: count, output: output, stride: stride, row: scratch)
        #endif
    }

    /// 1 行ずつ Float に戻して掛ける（row は columns 個）
    private static func simple(_ matrix: GGUFMatrix, _ range: Range<Int>, input: UnsafePointer<Float>, count: Int,
                               output: UnsafeMutablePointer<Float>, stride: Int, row: UnsafeMutablePointer<Float>) {
        for r in range {
            dequantizeRow(matrix, row: r, into: row)
            for n in 0..<count {
                output[n * stride + r] = MiniMath.dot(row, input + n * matrix.columns, count: matrix.columns)
            }
        }
    }

    #if canImport(Accelerate)
    /// 足し合わせ用の 1 の並び（組の数がこれより多い行列は 1 行ずつの計算にする）
    private static let ones = [Float](repeating: 1, count: 4096)

    /// 1 行ぶんを rows 行に並べる（倍にしながら写すので、呼び出しは log rows 回）
    private static func tile(_ target: UnsafeMutablePointer<Float>, length: Int, rows: Int) {
        var filled = 1
        while filled < rows {
            let count = min(filled, rows - filled)
            (target + filled * length).update(from: target, count: count * length)
            filled += count
        }
    }

    /// 下位 4 ビットを取り出さずに、1 バイトの値 b = 下 + 16 × 上 のまま使う:
    /// Σ q·x = Σ (b · x偶 + 上 · (x奇 − 16 · x偶))。x を行の数だけ並べておけば、まとまり全体を vDSP_vmma 1 回で計算できる。
    /// そのあと y = 倍率 · (組ごとの Σ q·x) + 下限 · (組ごとの Σ x) を sgemv 2 回で。
    /// 行ごとの呼び出しが無い（最適化なしのビルドでは、行ごとの呼び出しのほうが計算より高くつく）
    private static func accelerated(_ matrix: GGUFMatrix, _ range: Range<Int>, input: UnsafePointer<Float>, count: Int,
                                    output: UnsafeMutablePointer<Float>, stride: Int,
                                    scratch: UnsafeMutablePointer<Float>, capacity: Int) {
        guard let groupParams = matrix.groupParams else { return }
        let columns = matrix.columns, halfColumns = columns / 2
        let groups = columns / groupSize
        let five = matrix.highBits != nil
        guard groups <= ones.count, capacity >= groups + 64 + floatsPerRow(columns: columns, five: five) else {
            // とても長い行で置き場所が足りないとき（ふつうは起きない）: 1 行ずつ Float に戻して掛ける
            simple(matrix, range, input: input, count: count, output: output, stride: stride, row: scratch)
            return
        }
        let blockRows = blockRows(columns: columns, capacity: capacity, five: five)
        let block = blockRows * halfColumns, groupBlock = blockRows * groups
        // 置き場所（block = 行 × 列 ÷ 2）: バイトの値・上の 4 ビット・積・x偶の並び・x奇補正の並び
        // （5 ビット: 偶数番目と奇数番目の 5 ビット目 × 16・x奇の並び・積 2）・倍率と下限・倍率・下限・組ごとの和・Σ x
        let bytes = scratch, high = bytes + block, products = high + block
        let evenTiles = products + block, oddTiles = evenTiles + block
        let fifthEven = oddTiles + block, fifthOdd = fifthEven + (five ? block : 0)
        let plainOddTiles = fifthOdd + (five ? block : 0), products2 = plainOddTiles + (five ? block : 0)
        let params = products2 + (five ? block : 0)
        let scales = params + 2 * groupBlock, minimums = scales + groupBlock
        let dots = minimums + groupBlock, scaledDots = dots + groupBlock
        let sums = scaledDots + groupBlock
        var minusSixteen: Float = -16
        ones.withUnsafeBufferPointer { onesBuffer in
            let one = onesBuffer.baseAddress!
            var row = range.lowerBound
            while row < range.upperBound {
                let rows = min(blockRows, range.upperBound - row)
                let pairs = rows * halfColumns, blockGroups = rows * groups
                unpack(matrix, firstRow: row, rows: rows, bytes: bytes, high: high,
                       fifth: five ? (fifthEven, fifthOdd) : nil, work: products)
                // 倍率と下限（半精度 → Float）→ 倍率だけ・下限だけに分ける
                var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: groupParams + row * groups * 4), height: 1,
                                           width: vImagePixelCount(blockGroups * 2), rowBytes: blockGroups * 4)
                var target = vImage_Buffer(data: params, height: 1, width: vImagePixelCount(blockGroups * 2), rowBytes: blockGroups * 8)
                vImageConvert_Planar16FtoPlanarF(&source, &target, vImage_Flags(kvImageNoFlags))
                cblas_scopy(Int32(blockGroups), params, 2, scales, 1)
                cblas_scopy(Int32(blockGroups), params + 1, 2, minimums, 1)
                for n in 0..<count {
                    let x = input + n * columns
                    let y = output + n * stride + row
                    // 組ごとの Σ x
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(groups), 32, 1, x, 32, one, 1, 0, sums, 1)
                    // x偶、x奇 − 16 × x偶（5 ビットなら x奇も）を行の数だけ並べる
                    cblas_scopy(Int32(halfColumns), x, 2, evenTiles, 1)
                    vDSP_vsma(evenTiles, 1, &minusSixteen, x + 1, 2, oddTiles, 1, vDSP_Length(halfColumns))
                    tile(evenTiles, length: halfColumns, rows: rows)
                    tile(oddTiles, length: halfColumns, rows: rows)
                    vDSP_vmma(bytes, 1, evenTiles, 1, high, 1, oddTiles, 1, products, 1, vDSP_Length(pairs))
                    if five {
                        cblas_scopy(Int32(halfColumns), x + 1, 2, plainOddTiles, 1)
                        tile(plainOddTiles, length: halfColumns, rows: rows)
                        vDSP_vmma(fifthEven, 1, evenTiles, 1, fifthOdd, 1, plainOddTiles, 1, products2, 1, vDSP_Length(pairs))
                        vDSP_vadd(products, 1, products2, 1, products, 1, vDSP_Length(pairs))
                    }
                    // 組ごとの Σ q·x（1 つの組は 16 バイト = 32 個）
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(blockGroups), 16, 1, products, 16, one, 1, 0, dots, 1)
                    // y = 下限 · Σ x + (倍率 ⊙ Σ q·x) の行ごとの和
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rows), Int32(groups), 1, minimums, Int32(groups), sums, 1, 0, y, 1)
                    vDSP_vmul(scales, 1, dots, 1, scaledDots, 1, vDSP_Length(blockGroups))
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rows), Int32(groups), 1, scaledDots, Int32(groups), one, 1, 1, y, 1)
                }
                row += rows
            }
        }
    }

    /// rows 行ぶんの、バイトの値（b）と上の 4 ビット（floor(b / 16)）。
    /// 5 ビットなら、5 ビット目 × 16 を、偶数番目（fifth.0）と奇数番目（fifth.1）に分けて置く
    private static func unpack(_ matrix: GGUFMatrix, firstRow: Int, rows: Int, bytes: UnsafeMutablePointer<Float>,
                               high: UnsafeMutablePointer<Float>,
                               fifth: (UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>)?,
                               work: UnsafeMutablePointer<Float>) {
        let columns = matrix.columns
        let total = rows * columns
        let pairs = total / 2
        let packed = matrix.data.assumingMemoryBound(to: UInt8.self) + firstRow * columns / 2
        vDSP_vfltu8(packed, 1, bytes, 1, vDSP_Length(pairs))
        var sixteenth: Float = 1.0 / 16
        vDSP_vsmul(bytes, 1, &sixteenth, high, 1, vDSP_Length(pairs))
        var length = Int32(pairs)
        vvfloorf(high, high, &length)
        guard let fifth, let highBits = matrix.highBits else { return }
        // 5 ビット目: c 番目は (c / 8) バイト目の (c % 8) ビット目。floor で 1 ビットずつ取り出す。
        // c = 8k + j は、j が偶数なら偶数番目の 4k + j/2 番、奇数なら奇数番目の 4k + (j−1)/2 番
        let eighths = total / 8
        let source = highBits.assumingMemoryBound(to: UInt8.self) + firstRow * columns / 8
        var current = work, next = work + eighths
        vDSP_vfltu8(source, 1, current, 1, vDSP_Length(eighths))
        var oneHalf: Float = 0.5, minusTwo: Float = -2, sixteen: Float = 16
        var count = Int32(eighths)
        for bit in 0..<8 {
            vDSP_vsmul(current, 1, &oneHalf, next, 1, vDSP_Length(eighths))
            vvfloorf(next, next, &count)
            // このビット = 今の値 − 2 × 次の値（current に入れる）
            vDSP_vsma(next, 1, &minusTwo, current, 1, current, 1, vDSP_Length(eighths))
            let target = bit % 2 == 0 ? fifth.0 : fifth.1
            vDSP_vsmul(current, 1, &sixteen, target + bit / 2, 4, vDSP_Length(eighths))
            swap(&current, &next)
        }
    }
    #endif
}
