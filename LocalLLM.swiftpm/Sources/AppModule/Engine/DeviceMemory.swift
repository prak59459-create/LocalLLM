import Foundation
import os

/// 端末とアプリのメモリ状況。
enum DeviceMemory {
    /// 端末の物理メモリ（バイト）。
    static var physicalBytes: Int64 {
        Int64(ProcessInfo.processInfo.physicalMemory)
    }

    /// このアプリが「あと」使えるメモリの目安（バイト）。iOS がアプリを強制終了する上限までの残り。
    static var availableBytes: Int64 {
        Int64(os_proc_available_memory())
    }

    /// このアプリが今使っているメモリ（phys_footprint、バイト）。
    static var footprintBytes: Int64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundPointer, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Int64(info.phys_footprint)
    }

    /// 「アプリが使える上限」の目安 = 使用中 + 残り。
    static var appLimitBytes: Int64 {
        (footprintBytes ?? 0) + availableBytes
    }

    /// 見積もり（必要バイト数）から、読み込めそうかを判定する。
    enum Fit: Sendable {
        case comfortable   // 余裕あり
        case tight         // ぎりぎり（他のアプリを閉じると良い）
        case tooLarge      // おそらく強制終了される

        var label: String {
            switch self {
            case .comfortable: return L("余裕あり", "Fits")
            case .tight: return L("ぎりぎり", "Tight")
            case .tooLarge: return L("メモリ不足の可能性大", "Likely too large")
            }
        }
    }

    /// 必要量を「アプリの上限」と比べる。読み込み済みのモデルがあれば、その分は空く前提で currentModelBytes を足し戻す。
    static func fit(requiredBytes: Int64, currentModelBytes: Int64 = 0) -> Fit {
        let limit = Double(appLimitBytes)
        let available = Double(availableBytes + currentModelBytes)
        let required = Double(requiredBytes)
        if required <= min(available * 0.75, limit * 0.6) { return .comfortable }
        if required <= available * 0.95 { return .tight }
        return .tooLarge
    }
}
