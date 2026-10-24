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

extension DeviceMemory {
    /// 今のメモリの見積もり（測れない環境では nil）
    static var budget: MemoryBudget? {
        let available = availableBytes
        guard available > 0 else { return nil }
        return MemoryBudget(available: available, appLimit: (footprintBytes ?? 0) + available)
    }
}

/// 推論の 1 回ごとにメモリを確かめる。足りなくなったら、iOS に強制終了される前に自分で止める。
/// InferenceEngine（actor）の中だけで使う。
final class MemoryGate {
    /// 語彙数と logits 1 つあたりのバイト数（1 回の出力の大きさの見積もりに使う）
    var vocabSize = 32_000
    var bytesPerLogit = 2
    /// iOS からメモリ不足の警告を受けた（しばらくは控えめに計算する）
    private(set) var warnedAt: Date?

    func noteWarning() { warnedAt = Date() }

    private var recentlyWarned: Bool {
        guard let warnedAt else { return false }
        return Date().timeIntervalSince(warnedAt) < 90
    }

    /// 計算の前に呼ぶ。残りが危ないところまで減っていたら EngineError.lowMemory を投げる。
    func check() throws {
        guard let budget = DeviceMemory.budget else { return }
        if budget.isCritical || (recentlyWarned && budget.isLow) {
            throw EngineError.lowMemory(available: budget.available)
        }
    }

    /// プロンプトを流し込む 1 回あたりのトークン数（メモリが少ないほど小さくする）
    func chunk(_ requested: Int) -> Int {
        guard let budget = DeviceMemory.budget else { return requested }
        var size = budget.prefillChunk(requested: requested, vocabSize: vocabSize, bytesPerLogit: bytesPerLogit)
        if recentlyWarned { size = min(size, 16) }
        return size
    }

    /// 毎回系列全体を入力するモデルで、安全に扱える長さ（測れなければ nil）
    func statelessLimit() -> Int? {
        DeviceMemory.budget?.statelessTokenLimit(vocabSize: vocabSize, bytesPerLogit: bytesPerLogit)
    }

    /// 新しく確保する bytes が、危ないところまで食い込まないか
    func canAllocate(_ bytes: Int64) -> Bool {
        guard let budget = DeviceMemory.budget else { return true }
        return budget.available - bytes > budget.critical
    }
}

