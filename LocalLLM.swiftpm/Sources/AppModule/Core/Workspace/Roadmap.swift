import Foundation

/// ロードマップの丸 1 つ（機能・アイデア・目標など）。
struct RoadmapNode: Codable, Hashable, Identifiable, Sendable {
    enum Status: String, Codable, CaseIterable, Identifiable, Sendable {
        case idea
        case planned
        case inProgress
        case done
        case blocked

        var id: String { rawValue }

        var label: String {
            switch self {
            case .idea: return L("アイデア", "Idea")
            case .planned: return L("予定", "Planned")
            case .inProgress: return L("作業中", "In progress")
            case .done: return L("完了", "Done")
            case .blocked: return L("保留", "Blocked")
            }
        }

        /// モデルが書いた状態の名前を読む（"in_progress" "完了" なども）
        static func parse(_ text: String) -> Status? {
            let key = text.lowercased().replacingOccurrences(of: "_", with: "").replacingOccurrences(of: "-", with: "")
                .replacingOccurrences(of: " ", with: "")
            switch key {
            case "idea", "アイデア", "案": return .idea
            case "planned", "todo", "plan", "予定", "未着手": return .planned
            case "inprogress", "doing", "wip", "started", "作業中", "実装中": return .inProgress
            case "done", "complete", "completed", "finished", "implemented", "完了", "実装済み": return .done
            case "blocked", "onhold", "paused", "保留": return .blocked
            default: return nil
            }
        }
    }

    enum Kind: String, Codable, CaseIterable, Identifiable, Sendable {
        case goal
        case feature
        case idea
        case bug
        case note

        var id: String { rawValue }

        var label: String {
            switch self {
            case .goal: return L("目標", "Goal")
            case .feature: return L("機能", "Feature")
            case .idea: return L("アイデア", "Idea")
            case .bug: return L("不具合", "Bug")
            case .note: return L("メモ", "Note")
            }
        }

        static func parse(_ text: String) -> Kind? {
            switch text.lowercased() {
            case "goal", "milestone", "目標": return .goal
            case "feature", "task", "機能": return .feature
            case "idea", "アイデア", "案": return .idea
            case "bug", "fix", "不具合", "バグ": return .bug
            case "note", "memo", "メモ": return .note
            default: return nil
            }
        }
    }

    var id: UUID = UUID()
    /// モデルや人が呼ぶための短い番号（#3 など）
    var number: Int
    var title: String
    var detail: String = ""
    var status: Status = .idea
    var kind: Kind = .feature
    /// ボード上の位置（中心）
    var x: Double = 0
    var y: Double = 0
    /// 関係するファイル
    var files: [String] = []
    var createdAt: Date = Date()
    var updatedAt: Date = Date()

    var reference: String { "#\(number)" }
}

/// 丸どうしのつながり。from から to が生えている（to は from から派生した、または from のあとに作る）。
struct RoadmapEdge: Codable, Hashable, Identifiable, Sendable {
    var id: UUID = UUID()
    var from: UUID
    var to: UUID
}

/// プロジェクトのロードマップ（フリーボード）。.cdws/roadmap.json に保存する。
struct Roadmap: Codable, Sendable {
    var nodes: [RoadmapNode] = []
    var edges: [RoadmapEdge] = []
    var nextNumber: Int = 1

    /// 丸の大きさの目安（重ならないように置くため）
    static let spacing: Double = 170

    func node(_ id: UUID) -> RoadmapNode? { nodes.first { $0.id == id } }

    /// "#3" "3" やタイトルから丸を探す
    func node(reference: String) -> RoadmapNode? {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        let digits = trimmed.hasPrefix("#") ? String(trimmed.dropFirst()) : trimmed
        if let number = Int(digits) { return nodes.first { $0.number == number } }
        if let id = UUID(uuidString: trimmed) { return node(id) }
        let lower = trimmed.lowercased()
        return nodes.first { $0.title.lowercased() == lower } ?? nodes.first { $0.title.lowercased().contains(lower) && !lower.isEmpty }
    }

    func children(of id: UUID) -> [RoadmapNode] {
        let ids = Set(edges.filter { $0.from == id }.map(\.to))
        return nodes.filter { ids.contains($0.id) }
    }

    func parents(of id: UUID) -> [RoadmapNode] {
        let ids = Set(edges.filter { $0.to == id }.map(\.from))
        return nodes.filter { ids.contains($0.id) }
    }

    /// 丸を足す。parent があればその近くに置いて線でつなぐ。
    @discardableResult
    mutating func add(title: String, detail: String = "", kind: RoadmapNode.Kind = .feature,
                      status: RoadmapNode.Status = .idea, parent: UUID? = nil,
                      at position: (x: Double, y: Double)? = nil) -> RoadmapNode {
        let parentNode = parent.flatMap { id in nodes.first { $0.id == id } }
        let place = position ?? freePosition(near: parentNode)
        let added = RoadmapNode(number: nextNumber, title: title, detail: detail, status: status, kind: kind,
                                x: place.x, y: place.y)
        nextNumber += 1
        nodes.append(added)
        if let parentNode {
            edges.append(RoadmapEdge(from: parentNode.id, to: added.id))
        }
        return added
    }

    mutating func update(_ id: UUID, _ change: (inout RoadmapNode) -> Void) {
        guard let index = nodes.firstIndex(where: { $0.id == id }) else { return }
        change(&nodes[index])
        nodes[index].updatedAt = Date()
    }

    mutating func move(_ id: UUID, to point: (x: Double, y: Double)) {
        guard let index = nodes.firstIndex(where: { $0.id == id }) else { return }
        nodes[index].x = point.x
        nodes[index].y = point.y
    }

    /// つなぐ（同じ線や、自分自身へは足さない）
    mutating func connect(_ from: UUID, _ to: UUID) {
        guard from != to, node(from) != nil, node(to) != nil,
              !edges.contains(where: { ($0.from == from && $0.to == to) || ($0.from == to && $0.to == from) }) else { return }
        edges.append(RoadmapEdge(from: from, to: to))
    }

    mutating func disconnect(_ edgeID: UUID) {
        edges.removeAll { $0.id == edgeID }
    }

    mutating func remove(_ id: UUID) {
        nodes.removeAll { $0.id == id }
        edges.removeAll { $0.from == id || $0.to == id }
    }

    /// 親の周り（右側から順に）で、ほかの丸と重ならない場所。親が無ければ全体の右下の空き。
    func freePosition(near parent: RoadmapNode?) -> (x: Double, y: Double) {
        func isFree(_ x: Double, _ y: Double) -> Bool {
            !nodes.contains { hypot($0.x - x, $0.y - y) < Self.spacing * 0.8 }
        }
        if let parent {
            let siblings = children(of: parent.id).count
            for ring in 1...6 {
                let radius = Self.spacing * Double(ring)
                let count = 8 * ring
                for step in 0..<count {
                    // 右・右下・右上…と広げていく
                    let order = step % 2 == 0 ? step / 2 : -(step + 1) / 2
                    let angle = Double(order + siblings) * (2 * Double.pi / Double(count))
                    let x = parent.x + radius * cos(angle)
                    let y = parent.y + radius * sin(angle)
                    if isFree(x, y) { return (x, y) }
                }
            }
            return (parent.x + Self.spacing, parent.y + Self.spacing)
        }
        guard !nodes.isEmpty else { return (0, 0) }
        let maxX = nodes.map(\.x).max() ?? 0
        let minY = nodes.map(\.y).min() ?? 0
        var y = minY
        while !isFree(maxX + Self.spacing, y) { y += Self.spacing }
        return (maxX + Self.spacing, y)
    }

    /// 木の形に並べ直す（根を左に、子を右へ。兄弟は縦に並べる）。
    mutating func autoLayout() {
        guard !nodes.isEmpty else { return }
        let childIDs = Set(edges.map(\.to))
        var roots = nodes.filter { !childIDs.contains($0.id) }
        if roots.isEmpty { roots = [nodes[0]] }
        var placed = Set<UUID>()
        var nextRow = 0.0
        func place(_ id: UUID, depth: Int) -> Double {
            placed.insert(id)
            let kids = children(of: id).filter { !placed.contains($0.id) }.sorted { $0.number < $1.number }
            let y: Double
            if kids.isEmpty {
                y = nextRow
                nextRow += 1
            } else {
                let rows = kids.map { place($0.id, depth: depth + 1) }
                y = ((rows.first ?? 0) + (rows.last ?? 0)) / 2
            }
            move(id, to: (Double(depth) * Self.spacing * 1.3, y * Self.spacing * 0.8))
            return y
        }
        for root in roots.sorted(by: { $0.number < $1.number }) where !placed.contains(root.id) {
            _ = place(root.id, depth: 0)
            nextRow += 0.5
        }
        // つながっていない輪の中のものなど
        for node in nodes where !placed.contains(node.id) {
            _ = place(node.id, depth: 0)
        }
    }

    /// モデルに渡す要約（番号・状態・タイトル・親）
    func summary(maxNodes: Int = 40, detailCharacters: Int = 80) -> String {
        guard !nodes.isEmpty else { return L("（まだ何もありません）", "(empty)") }
        var lines: [String] = []
        for node in nodes.sorted(by: { $0.number < $1.number }).prefix(maxNodes) {
            var line = "\(node.reference) [\(node.status.rawValue)] \(node.title)"
            let parents = parents(of: node.id).map(\.reference)
            if !parents.isEmpty { line += " ← " + parents.joined(separator: ",") }
            if !node.detail.isEmpty {
                let detail = node.detail.replacingOccurrences(of: "\n", with: " ")
                line += " — " + (detail.count > detailCharacters ? String(detail.prefix(detailCharacters)) + "…" : detail)
            }
            lines.append(line)
        }
        if nodes.count > maxNodes { lines.append("… (+\(nodes.count - maxNodes))") }
        return lines.joined(separator: "\n")
    }

    /// 状態ごとの数
    var progress: (done: Int, total: Int) {
        let counted = nodes.filter { $0.kind != .note }
        return (counted.filter { $0.status == .done }.count, counted.count)
    }
}
