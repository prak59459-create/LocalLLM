import Foundation

/// GGUF のモデルの重みを、メモリ（RAM）にどう置くか。
///
/// 重みはファイルをメモリに対応づけて（mmap）読んでいる。ふつうは OS が「読んだところを残し、足りなくなったら捨てる」。
/// ここでは、それに口を出す:
/// - 先に全部読む: 読み込みのときに、重み全体を読んでおくよう OS に頼む（最初の返事が速い。RAM は多く使う）
/// - 使う分だけ: 使い終わった重みを、すぐ手放すよう OS に頼む。MoE は「最近使った専門家」だけ、
///   普通のモデルは「先頭から決めた数の層」だけを RAM に残し、ほかは毎回ファイルから読む
///   （RAM より大きなモデルも動かせるが、ファイルを読む分だけ遅い）
struct GGUFMemoryOptions: Codable, Equatable, Sendable {
    enum Residency: String, Codable, CaseIterable, Identifiable, Sendable {
        /// OS に任せる（これまでどおり）
        case automatic
        /// 読み込みのときに全部読んでおく
        case preload
        /// 使う分だけ置く
        case activeOnly

        var id: String { rawValue }
    }

    var residency: Residency = .automatic
    /// 1 トークンで使う専門家の数（MoE だけ。0 ならモデルの既定）
    var expertsUsed: Int = 0
    /// 使う分だけ置くとき、層ごとに RAM に残す専門家の数（0 なら、使う数の 2 倍）
    var keptExperts: Int = 0
    /// 使う分だけ置くとき（普通のモデル）、RAM に残す層の数（先頭から。-1 なら空いているメモリから決める）
    var keptLayers: Int = -1

    static let defaultsKey = "gguf.memoryOptions"

    /// 保存してある設定（無ければ既定）
    static var stored: GGUFMemoryOptions {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let options = try? JSONDecoder().decode(GGUFMemoryOptions.self, from: data) else { return GGUFMemoryOptions() }
        return options
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.defaultsKey) }
    }

    /// GPU は重みを先に全部 GPU から読める状態にするので、使う分だけ置くときは CPU で計算する
    var allowsGPU: Bool { residency != .activeOnly }
}

/// メモリに対応づけたファイルの範囲について、OS に使い方を伝える（madvise）
enum GGUFPages {
    typealias Region = (start: UnsafeRawPointer, count: Int)

    static var pageSize: Int { max(1, Int(getpagesize())) }

    /// 近いうちに使う（先に読んでおいてもらう）。ページの境目まで広げる
    static func willNeed(_ regions: [Region]) {
        for region in regions { advise(region, MADV_WILLNEED, inward: false) }
    }

    /// しばらく使わない（RAM から手放してもらう。次に使うときはファイルから読み直す）。
    /// ほかの重みと同じページは手放さないよう、ページの境目の内側だけ
    static func dontNeed(_ regions: [Region]) {
        for region in regions { advise(region, MADV_DONTNEED, inward: true) }
    }

    /// 飛び飛びに読む（先読みしない。使わない専門家まで読み込まれないように）
    static func random(_ regions: [Region]) {
        for region in regions { advise(region, MADV_RANDOM, inward: false) }
    }

    /// ふつうの読み方に戻す
    static func normal(_ regions: [Region]) {
        for region in regions { advise(region, MADV_NORMAL, inward: false) }
    }

    private static func advise(_ region: Region, _ advice: Int32, inward: Bool) {
        let page = pageSize
        let start = Int(bitPattern: region.start), end = start + region.count
        let first = inward ? (start + page - 1) / page * page : start / page * page
        let last = inward ? end / page * page : (end + page - 1) / page * page
        guard last > first, let pointer = UnsafeMutableRawPointer(bitPattern: first) else { return }
        _ = madvise(pointer, last - first, advice)
    }
}

extension GGUFMatrix {
    /// 重みが置いてあるメモリの範囲（面ごと）
    var regions: [GGUFPages.Region] {
        let rowBytes: Int
        switch storage {
        case .float32: rowBytes = columns * MemoryLayout<Float>.size
        case .int8Rows: rowBytes = columns
        case .raw: rowBytes = rawRowBytes
        case .q4Groups, .q5Groups: rowBytes = columns / 2
        }
        var result: [GGUFPages.Region] = [(data, rows * rowBytes)]
        if let scales { result.append((UnsafeRawPointer(scales), rows * MemoryLayout<Float>.size)) }
        if let highBits { result.append((highBits, rows * columns / 8)) }
        if let groupParams { result.append((groupParams, rows * (columns / GGUFGroupQuant.groupSize) * 4)) }
        return result
    }

    /// 重みのバイト数
    var byteCount: Int { regions.reduce(0) { $0 + $1.count } }
}

/// 「使う分だけ置く」の実際の管理。GGUFModel の計算の合間に呼ばれる（同じスレッドから）
final class GGUFResidency {
    private let options: GGUFMemoryOptions
    /// 層ごとの専門家の重みの範囲 [層][専門家]（MoE の層だけ）
    private let expertRegions: [[[GGUFPages.Region]]]
    /// 層ごとの重みの範囲（専門家を除く）
    private let layerRegions: [[GGUFPages.Region]]
    /// RAM に残す層（普通のモデル）
    private let keptLayers: Int
    /// 層ごとに RAM に残す専門家の数
    private let keptExperts: Int
    /// 層ごとの、最近使った順の専門家（先頭が一番最近）
    private var recent: [[Int]]
    private let isMixture: Bool

    /// - Parameter availableBytes: 空いているメモリ（keptLayers が -1 のとき、残す層の数を決めるのに使う）
    init(store: GGUFWeightStore, options: GGUFMemoryOptions, availableBytes: Int64?) {
        let config = store.config
        self.options = options
        isMixture = config.moe != nil
        var experts: [[[GGUFPages.Region]]] = []
        var layers: [[GGUFPages.Region]] = []
        for index in 0..<config.layerCount {
            let prefix = "blk.\(index)."
            let names = store.header.entries.map(\.name).filter { $0.hasPrefix(prefix) }
            let expertNames = Set(["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"].map { prefix + $0 })
            layers.append(names.filter { !expertNames.contains($0) }.compactMap { try? store.matrix($0) }.flatMap(\.regions))
            if let moe = config.moe, config.layers[index].moe,
               let gate = try? store.matrix(prefix + "ffn_gate_exps.weight"),
               let up = try? store.matrix(prefix + "ffn_up_exps.weight"),
               let down = try? store.matrix(prefix + "ffn_down_exps.weight") {
                let ff = config.layers[index].feedForwardSize, d = config.embeddingSize
                experts.append((0..<moe.expertCount).map { e in
                    gate.slice((e * ff)..<((e + 1) * ff)).regions + up.slice((e * ff)..<((e + 1) * ff)).regions
                        + down.slice((e * d)..<((e + 1) * d)).regions
                })
            } else {
                experts.append([])
            }
        }
        expertRegions = experts
        layerRegions = layers
        recent = experts.map { _ in [] }
        let used = config.moe.map { options.expertsUsed > 0 ? options.expertsUsed : $0.expertsUsed } ?? 0
        keptExperts = max(1, options.keptExperts > 0 ? options.keptExperts : used * 2)
        if options.keptLayers >= 0 {
            keptLayers = min(options.keptLayers, config.layerCount)
        } else {
            // 空いているメモリの半分に収まるだけの層を残す
            let perLayer = max(1, layers.map { $0.reduce(0) { $0 + $1.count } }.max() ?? 1)
            let budget = Int((availableBytes ?? 0) / 2)
            keptLayers = min(config.layerCount, max(0, budget / perLayer))
        }
    }

    /// 層ごとの専門家の重みのバイト数（画面に出す目安）
    var expertBytes: Int { expertRegions.first { !$0.isEmpty }?.first?.reduce(0) { $0 + $1.count } ?? 0 }

    /// 読み込んだすぐあと
    func prepare() {
        switch options.residency {
        case .automatic:
            return
        case .preload:
            GGUFPages.willNeed(layerRegions.flatMap { $0 } + expertRegions.flatMap { $0.flatMap { $0 } })
        case .activeOnly:
            // 先読みで使わない専門家・層まで読まれないようにして、いったん全部手放す
            let experts = expertRegions.flatMap { $0.flatMap { $0 } }
            GGUFPages.random(experts)
            GGUFPages.dontNeed(experts)
            if !isMixture {
                let streamed = layerRegions.dropFirst(keptLayers).flatMap { $0 }
                GGUFPages.dontNeed(streamed)
            }
        }
    }

    /// 専門家 1 つの計算が終わった。最近使った専門家だけを残し、それより古いものを手放す
    func expertFinished(layer: Int, expert: Int) {
        guard options.residency == .activeOnly, layer < recent.count, !expertRegions[layer].isEmpty else { return }
        var list = recent[layer]
        list.removeAll { $0 == expert }
        list.insert(expert, at: 0)
        while list.count > keptExperts {
            let dropped = list.removeLast()
            GGUFPages.dontNeed(expertRegions[layer][dropped])
        }
        recent[layer] = list
    }

    /// 層 1 つの計算が終わった（普通のモデル）。残す層でなければ手放す
    func layerFinished(_ layer: Int) {
        guard options.residency == .activeOnly, !isMixture, layer >= keptLayers, layer < layerRegions.count else { return }
        GGUFPages.dontNeed(layerRegions[layer])
    }

    /// 今 RAM に残している数（画面に出す）
    var summary: String {
        switch options.residency {
        case .automatic, .preload:
            return ""
        case .activeOnly:
            return isMixture ? L("専門家は層ごとに最近の \(keptExperts) 個だけ RAM に置く", "Keeping the \(keptExperts) most recent experts per layer in RAM")
                             : L("先頭の \(keptLayers) 層だけ RAM に置き、ほかは毎回読む", "Keeping the first \(keptLayers) layers in RAM, streaming the rest")
        }
    }
}
