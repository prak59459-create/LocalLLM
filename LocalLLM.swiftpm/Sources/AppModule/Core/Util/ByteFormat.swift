import Foundation

/// バイト数の表示（Apple のファイルサイズ表示と同じく 1000 単位）。例: "278 MB", "1.93 GB"
enum ByteFormat {
    private static let units: [String] = ["B", "KB", "MB", "GB", "TB"]

    static func string(_ bytes: Int64) -> String {
        var scaled: Double = Double(max(bytes, 0))
        var unit: Int = 0
        let thousand: Double = 1000
        while scaled >= thousand && unit < units.count - 1 {
            scaled /= thousand
            unit += 1
        }
        if unit == 0 { return "\(Int(scaled)) B" }
        // 有効数字 3 桁くらい
        let format: String
        if scaled >= 100 {
            format = "%.0f"
        } else if scaled >= 10 {
            format = "%.1f"
        } else {
            format = "%.2f"
        }
        let number: String = String(format: format, scaled)
        return "\(number) \(units[unit])"
    }
}
