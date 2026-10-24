import Foundation

/// 重みに埋め込まれた回転位置埋め込み（RoPE）の sin / cos の表を、長いコンテキスト用に作り直す。
/// 今の表から列ごとの周波数を推定し、標準の式（θ^(-2i/d)）に当てはまればその正確な値を使う。
enum RopeTable {
    enum Kind: Sendable {
        case sin
        case cos
    }

    enum FitError: Error, LocalizedError {
        case notRope
        case tooSmall

        var errorDescription: String? {
            switch self {
            case .notRope: return L("位置の表が sin / cos の形になっていません", "The position table is not a sin/cos table")
            case .tooSmall: return L("位置の表が小さすぎて推定できません", "The position table is too small to analyze")
            }
        }
    }

    struct Fit: Sendable {
        var kind: Kind
        /// 列ごとの周波数（1 位置あたりの角度）
        var frequencies: [Double]
        /// 標準の式に当てはまったときの θ
        var theta: Double?
        /// 今の表との最大の差
        var maxError: Double
    }

    /// rows × columns の表（行が位置）を調べる。
    static func fit(_ table: [Float], rows: Int, columns: Int) throws -> Fit {
        guard rows >= 8, columns >= 2, table.count >= rows * columns else { throw FitError.tooSmall }
        func value(_ row: Int, _ column: Int) -> Double { Double(table[row * columns + column]) }
        // 0 行目が全部 1 なら cos、全部 0 なら sin
        let first = (0..<columns).map { value(0, $0) }
        let kind: Kind
        if first.allSatisfy({ abs($0 - 1) < 0.01 }) {
            kind = .cos
        } else if first.allSatisfy({ abs($0) < 0.01 }) {
            kind = .sin
        } else {
            throw FitError.notRope
        }

        var frequencies: [Double] = []
        for column in 0..<columns {
            frequencies.append(estimateFrequency(rows: rows, kind: kind) { value($0, column) })
        }
        var theta = snapTheta(frequencies)
        if let candidate = theta {
            // 式の値で今の表を再現できるか確かめる（できなければ推定値のまま使う）
            let exact = standardFrequencies(theta: candidate, columns: columns, layout: layout(of: frequencies, theta: candidate))
            if maxError(table, rows: rows, columns: columns, kind: kind, frequencies: exact) < 4e-3 {
                frequencies = exact
            } else {
                theta = nil
            }
        }
        let error = maxError(table, rows: rows, columns: columns, kind: kind, frequencies: frequencies)
        guard error < 6e-3 else { throw FitError.notRope }
        return Fit(kind: kind, frequencies: frequencies, theta: theta, maxError: error)
    }

    /// 新しい表（rows 行）を作る
    static func generate(_ fit: Fit, rows: Int) -> [Float] {
        let columns = fit.frequencies.count
        var output = [Float](repeating: 0, count: rows * columns)
        for row in 0..<rows {
            let position = Double(row)
            for column in 0..<columns {
                let angle = position * fit.frequencies[column]
                output[row * columns + column] = Float(fit.kind == .cos ? cos(angle) : sin(angle))
            }
        }
        return output
    }

    /// 1 列の周波数を推定する。1 行目から始めて、行を倍々に進めながら角度の候補から一番近いものを選ぶ。
    static func estimateFrequency(rows: Int, kind: Kind, value: (Int) -> Double) -> Double {
        func clamp(_ x: Double) -> Double { max(-1, min(1, x)) }
        // 角度は 1 位置あたり π 未満（RoPE の周波数は 1 以下）
        var frequency = kind == .cos ? acos(clamp(value(1))) : asin(clamp(value(1)))
        var row = 2
        var checkpoints: [Int] = []
        while row < rows {
            checkpoints.append(row)
            row *= 2
        }
        if checkpoints.last != rows - 1 { checkpoints.append(rows - 1) }
        for row in checkpoints {
            let predicted = frequency * Double(row)
            let measured = clamp(value(row))
            let base = kind == .cos ? acos(measured) : asin(measured)
            let turns = (predicted / (2 * Double.pi)).rounded(.down)
            var best = predicted
            var bestDistance = Double.infinity
            for k in [turns - 1, turns, turns + 1] {
                let offset = 2 * Double.pi * k
                let candidates = kind == .cos ? [offset + base, offset - base] : [offset + base, offset + Double.pi - base]
                for candidate in candidates where abs(candidate - predicted) < bestDistance {
                    bestDistance = abs(candidate - predicted)
                    best = candidate
                }
            }
            frequency = best / Double(row)
        }
        return frequency
    }

    /// 列の並び方（前半と後半に同じ周波数を並べる形か、2 つずつ並べる形か）
    enum Layout {
        case halves
        case interleaved
    }

    static func standardFrequencies(theta: Double, columns: Int, layout: Layout) -> [Double] {
        let half = columns / 2
        return (0..<columns).map { column in
            let index = layout == .halves ? column % max(half, 1) : column / 2
            return pow(theta, -2 * Double(index) / Double(columns))
        }
    }

    private static func layout(of frequencies: [Double], theta: Double) -> Layout {
        let halves = standardFrequencies(theta: theta, columns: frequencies.count, layout: .halves)
        let interleaved = standardFrequencies(theta: theta, columns: frequencies.count, layout: .interleaved)
        func distance(_ other: [Double]) -> Double { zip(frequencies, other).map { abs($0 - $1) }.reduce(0, +) }
        return distance(halves) <= distance(interleaved) ? .halves : .interleaved
    }

    /// 周波数の並びから θ を求め、よく使われる値に近ければそれにそろえる
    static func snapTheta(_ frequencies: [Double]) -> Double? {
        let columns = frequencies.count
        guard columns >= 4 else { return nil }
        // 前半と後半で同じなら halves、隣どうしが同じなら interleaved
        let isHalves = zip(frequencies.prefix(columns / 2), frequencies.suffix(columns / 2)).allSatisfy { abs($0 - $1) < 1e-3 }
        let index = isHalves ? 1 : 2
        let step = frequencies[index]
        guard step > 0, step < 1 else { return nil }
        let estimate = pow(step, -Double(columns) / 2)
        let common: [Double] = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 75_000_000]
        if let near = common.first(where: { abs($0 - estimate) / $0 < 0.05 }) { return near }
        return estimate
    }

    static func maxError(_ table: [Float], rows: Int, columns: Int, kind: Kind, frequencies: [Double]) -> Double {
        var worst = 0.0
        for row in 0..<rows {
            for column in 0..<columns {
                let angle = Double(row) * frequencies[column]
                let expected = kind == .cos ? cos(angle) : sin(angle)
                worst = max(worst, abs(expected - Double(table[row * columns + column])))
            }
        }
        return worst
    }
}
