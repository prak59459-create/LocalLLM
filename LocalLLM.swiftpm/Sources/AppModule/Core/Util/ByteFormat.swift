import Foundation

/// バイト数の表示（Apple のファイルサイズ表示と同じく 1000 単位）。例: "278 MB", "1.93 GB"
enum ByteFormat {
    static func string(_ bytes: Int64) -> String {
        let value = Double(max(bytes, 0))
        let units = ["B", "KB", "MB", "GB", "TB"]
        var scaled = value
        var unit = 0
        while scaled >= 1000 && unit < units.count - 1 {
            scaled /= 1000
            unit += 1
        }
        if unit == 0 { return "\(Int(scaled)) B" }
        // 有効数字 3 桁くらい
        let digits = scaled >= 100 ? 0 : (scaled >= 10 ? 1 : 2)
        return String(format: "%.\(digits)f", scaled) + " " + units[unit]
    }
}
