import Foundation

/// 差分の 1 行。
struct DiffLine: Hashable, Sendable {
    enum Kind: Hashable, Sendable {
        case context
        case added
        case removed
    }
    var kind: Kind
    var text: String
    /// 元のファイルの行番号（1 始まり。追加した行は nil）
    var oldNumber: Int?
    /// 新しいファイルの行番号（消した行は nil）
    var newNumber: Int?
}

/// 変わったところとその前後の行のまとまり。
struct DiffHunk: Hashable, Sendable {
    var oldStart: Int
    var oldCount: Int
    var newStart: Int
    var newCount: Int
    var lines: [DiffLine]

    var header: String { "@@ -\(oldStart),\(oldCount) +\(newStart),\(newCount) @@" }
}

/// 行単位の差分（Myers のアルゴリズム）。
enum TextDiff {
    static func lines(_ text: String) -> [String] {
        if text.isEmpty { return [] }
        var lines = TextLines.split(text)
        if lines.last == "" { lines.removeLast() }
        return lines
    }

    /// すべての行に印を付けた差分
    static func diff(old: String, new: String) -> [DiffLine] {
        diff(oldLines: lines(old), newLines: lines(new))
    }

    static func diff(oldLines: [String], newLines: [String]) -> [DiffLine] {
        // 行を番号に置き換えてから比べる（文字列の比較を減らす）
        var ids: [String: Int] = [:]
        func id(_ line: String) -> Int {
            if let existing = ids[line] { return existing }
            let next = ids.count
            ids[line] = next
            return next
        }
        let a = oldLines.map(id)
        let b = newLines.map(id)

        // 先頭と末尾の同じ行は先に除く
        var prefix = 0
        while prefix < a.count && prefix < b.count && a[prefix] == b[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < a.count - prefix && suffix < b.count - prefix
                && a[a.count - 1 - suffix] == b[b.count - 1 - suffix] { suffix += 1 }

        var result: [DiffLine] = []
        for index in 0..<prefix {
            result.append(DiffLine(kind: .context, text: oldLines[index], oldNumber: index + 1, newNumber: index + 1))
        }
        let middleA = Array(a[prefix..<(a.count - suffix)])
        let middleB = Array(b[prefix..<(b.count - suffix)])
        for step in editScript(middleA, middleB) {
            switch step {
            case .keep(let i, let j):
                result.append(DiffLine(kind: .context, text: oldLines[prefix + i], oldNumber: prefix + i + 1, newNumber: prefix + j + 1))
            case .remove(let i):
                result.append(DiffLine(kind: .removed, text: oldLines[prefix + i], oldNumber: prefix + i + 1, newNumber: nil))
            case .insert(let j):
                result.append(DiffLine(kind: .added, text: newLines[prefix + j], oldNumber: nil, newNumber: prefix + j + 1))
            }
        }
        for offset in 0..<suffix {
            let i = a.count - suffix + offset
            let j = b.count - suffix + offset
            result.append(DiffLine(kind: .context, text: oldLines[i], oldNumber: i + 1, newNumber: j + 1))
        }
        return result
    }

    private enum Step {
        case keep(Int, Int)
        case remove(Int)
        case insert(Int)
    }

    /// Myers の O((N+M)D) の差分。違いが大きすぎるときは、全部消して全部足す形にする。
    private static func editScript(_ a: [Int], _ b: [Int]) -> [Step] {
        let n = a.count, m = b.count
        if n == 0 { return (0..<m).map { .insert($0) } }
        if m == 0 { return (0..<n).map { .remove($0) } }
        let maxD = min(n + m, 2000)
        let offset = maxD + 1
        var v = [Int](repeating: 0, count: 2 * maxD + 3)
        // 各段の v のうち使う範囲（k = -d-1 ... d+1）だけを残す（メモリは D² 程度）
        var trace: [[Int]] = []
        var found = false
        outer: for d in 0...maxD {
            trace.append(Array(v[(offset - d - 1)...(offset + d + 1)]))
            var k = -d
            while k <= d {
                var x: Int
                if k == -d || (k != d && v[offset + k - 1] < v[offset + k + 1]) {
                    x = v[offset + k + 1]
                } else {
                    x = v[offset + k - 1] + 1
                }
                var y = x - k
                while x < n && y < m && a[x] == b[y] {
                    x += 1
                    y += 1
                }
                v[offset + k] = x
                if x >= n && y >= m {
                    found = true
                    break outer
                }
                k += 2
            }
        }
        guard found else {
            return (0..<n).map { .remove($0) } + (0..<m).map { .insert($0) }
        }
        // たどり直して手順にする
        var steps: [Step] = []
        var x = n, y = m
        for d in stride(from: trace.count - 1, through: 0, by: -1) {
            let saved = trace[d]
            func value(_ k: Int) -> Int { saved[k + d + 1] }
            let k = x - y
            let previousK: Int
            if k == -d || (k != d && value(k - 1) < value(k + 1)) {
                previousK = k + 1
            } else {
                previousK = k - 1
            }
            let previousX = d == 0 ? 0 : value(previousK)
            let previousY = previousX - previousK
            while x > previousX && y > previousY {
                x -= 1
                y -= 1
                steps.append(.keep(x, y))
            }
            if d > 0 {
                if x == previousX {
                    y -= 1
                    steps.append(.insert(y))
                } else {
                    x -= 1
                    steps.append(.remove(x))
                }
            }
        }
        while x > 0 && y > 0 {
            x -= 1
            y -= 1
            steps.append(.keep(x, y))
        }
        return steps.reversed()
    }

    /// 変わったところだけを、前後 context 行と一緒にまとめる
    static func hunks(old: String, new: String, context: Int = 3) -> [DiffHunk] {
        hunks(from: diff(old: old, new: new), context: context)
    }

    static func hunks(from lines: [DiffLine], context: Int = 3) -> [DiffHunk] {
        let changed = lines.indices.filter { lines[$0].kind != .context }
        guard !changed.isEmpty else { return [] }
        var ranges: [ClosedRange<Int>] = []
        for index in changed {
            let range = max(0, index - context)...min(lines.count - 1, index + context)
            if let last = ranges.last, range.lowerBound <= last.upperBound + 1 {
                ranges[ranges.count - 1] = last.lowerBound...max(last.upperBound, range.upperBound)
            } else {
                ranges.append(range)
            }
        }
        return ranges.map { range in
            let slice = Array(lines[range])
            let oldStart = slice.compactMap(\.oldNumber).first ?? (lines[..<range.lowerBound].compactMap(\.oldNumber).last ?? 0) + 1
            let newStart = slice.compactMap(\.newNumber).first ?? (lines[..<range.lowerBound].compactMap(\.newNumber).last ?? 0) + 1
            return DiffHunk(oldStart: oldStart, oldCount: slice.filter { $0.kind != .added }.count,
                            newStart: newStart, newCount: slice.filter { $0.kind != .removed }.count,
                            lines: slice)
        }
    }

    /// unified diff の文字列（モデルに見せる・書き出す用）
    static func unified(old: String, new: String, path: String, context: Int = 3) -> String {
        let hunks = hunks(old: old, new: new, context: context)
        guard !hunks.isEmpty else { return "" }
        var output = "--- a/\(path)\n+++ b/\(path)\n"
        for hunk in hunks {
            output += hunk.header + "\n"
            for line in hunk.lines {
                switch line.kind {
                case .context: output += " " + line.text + "\n"
                case .added: output += "+" + line.text + "\n"
                case .removed: output += "-" + line.text + "\n"
                }
            }
        }
        return output
    }

    /// 足した行数と消した行数
    static func stats(old: String, new: String) -> (added: Int, removed: Int) {
        let lines = diff(old: old, new: new)
        return (lines.filter { $0.kind == .added }.count, lines.filter { $0.kind == .removed }.count)
    }
}
