import SwiftUI

/// ロードマップのフリーボード。機能やアイデアが丸で並び、線でつながる。
/// 丸はドラッグで動かせて、背景のドラッグで移動、ピンチで拡大縮小、ダブルタップで丸を足せる。
struct RoadmapBoardView: View {
    @Bindable var session: WorkspaceSession
    @Environment(ChatController.self) private var chat
    @Environment(AppSettings.self) private var settings
    @Environment(ConnectorStore.self) private var connectors

    @State private var offset: CGSize = .zero
    @State private var panDelta: CGSize = .zero
    @State private var scale: CGFloat = 1
    @State private var zoomDelta: CGFloat = 1
    @State private var dragNode: UUID?
    @State private var dragTranslation: CGSize = .zero
    @State private var connectFrom: UUID?
    @State private var planning = false
    @State private var planGoal = ""
    @State private var boardSize: CGSize = .zero

    private var currentScale: CGFloat { max(0.25, min(3, scale * zoomDelta)) }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                BoardGrid(scale: currentScale, offset: totalOffset)
                RoadmapEdges(roadmap: session.roadmap, position: { position(of: $0, in: geometry.size) },
                             radius: { RoadmapStyle.diameter(for: $0) * currentScale / 2 })
                ForEach(session.roadmap.nodes) { node in
                    nodeView(node, size: geometry.size)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .contentShape(Rectangle())
            .gesture(panGesture.simultaneously(with: zoomGesture))
            .onTapGesture(count: 2, coordinateSpace: .local) { location in
                addNode(at: location, size: geometry.size)
            }
            .onTapGesture { session.selectedNodeID = nil; connectFrom = nil }
            .onAppear { boardSize = geometry.size }
            .onChange(of: geometry.size) { _, size in boardSize = size }
        }
        .clipped()
        .background(Color(uiColor: .systemGroupedBackground))
        .overlay(alignment: .topLeading) { boardToolbar }
        .overlay(alignment: .bottomLeading) { RoadmapLegend().padding(12) }
        .overlay(alignment: .top) { connectHint }
        .overlay(alignment: .trailing) { inspector }
        .animation(.spring(duration: 0.35), value: session.roadmap.nodes.count)
        .alert(L("AI に計画を立ててもらう", "Let the AI plan"), isPresented: $planning) {
            TextField(L("作りたいもの・目標", "What do you want to build?"), text: $planGoal)
            Button(L("キャンセル", "Cancel"), role: .cancel) {}
            Button(L("計画する", "Plan")) {
                ask(CodingAgentPrompt.planRoadmap(goal: planGoal))
                planGoal = ""
            }
        } message: {
            Text(L("エージェントが機能を分けて、ロードマップに丸として並べます。", "The agent breaks the goal into features and adds them to the roadmap."))
        }
    }

    // MARK: - 座標

    private var totalOffset: CGSize {
        CGSize(width: offset.width + panDelta.width, height: offset.height + panDelta.height)
    }

    private func position(of node: RoadmapNode, in size: CGSize) -> CGPoint {
        var x = CGFloat(node.x) * currentScale + size.width / 2 + totalOffset.width
        var y = CGFloat(node.y) * currentScale + size.height / 2 + totalOffset.height
        if dragNode == node.id {
            x += dragTranslation.width
            y += dragTranslation.height
        }
        return CGPoint(x: x, y: y)
    }

    private func worldPoint(_ point: CGPoint, in size: CGSize) -> (x: Double, y: Double) {
        (Double((point.x - size.width / 2 - totalOffset.width) / currentScale),
         Double((point.y - size.height / 2 - totalOffset.height) / currentScale))
    }

    // MARK: - 丸

    private func nodeView(_ node: RoadmapNode, size: CGSize) -> some View {
        RoadmapBubble(node: node,
                      isSelected: session.selectedNodeID == node.id,
                      isConnectSource: connectFrom == node.id)
            .scaleEffect(currentScale)
            .position(position(of: node, in: size))
            .gesture(nodeDrag(node))
            .onTapGesture { tap(node) }
            .transition(.scale.combined(with: .opacity))
    }

    private func nodeDrag(_ node: RoadmapNode) -> some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { value in
                dragNode = node.id
                dragTranslation = value.translation
            }
            .onEnded { value in
                let x = node.x + Double(value.translation.width / currentScale)
                let y = node.y + Double(value.translation.height / currentScale)
                session.moveNode(node.id, to: (x, y))
                dragNode = nil
                dragTranslation = .zero
            }
    }

    private func tap(_ node: RoadmapNode) {
        if let source = connectFrom {
            if source != node.id { session.connect(source, node.id) }
            connectFrom = nil
            return
        }
        session.selectedNodeID = session.selectedNodeID == node.id ? nil : node.id
    }

    private func addNode(at location: CGPoint, size: CGSize) {
        session.addNode(title: L("新しいアイデア", "New idea"), parent: nil, at: worldPoint(location, in: size))
    }

    // MARK: - ジェスチャー

    private var panGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { value in panDelta = value.translation }
            .onEnded { value in
                offset.width += value.translation.width
                offset.height += value.translation.height
                panDelta = .zero
            }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in zoomDelta = value.magnification }
            .onEnded { value in
                scale = max(0.25, min(3, scale * value.magnification))
                zoomDelta = 1
            }
    }

    /// すべての丸が見えるように拡大率と位置を合わせる
    private func fitAll() {
        let nodes = session.roadmap.nodes
        guard !nodes.isEmpty, boardSize.width > 0 else { return }
        let xs = nodes.map { CGFloat($0.x) }, ys = nodes.map { CGFloat($0.y) }
        let minX = (xs.min() ?? 0) - 90, maxX = (xs.max() ?? 0) + 90
        let minY = (ys.min() ?? 0) - 90, maxY = (ys.max() ?? 0) + 90
        let fit = min(boardSize.width / max(maxX - minX, 1), boardSize.height / max(maxY - minY, 1))
        withAnimation(.easeInOut(duration: 0.3)) {
            scale = max(0.25, min(1.5, fit * 0.9))
            offset = CGSize(width: -(minX + maxX) / 2 * scale, height: -(minY + maxY) / 2 * scale)
        }
    }

    private func ask(_ prompt: String) {
        session.sendToAgent(prompt, chat: chat, settings: settings, connectors: connectors)
        session.requestAgentPanel()
    }

    // MARK: - 上に重ねるもの

    private var boardToolbar: some View {
        HStack(spacing: 10) {
            Button {
                let center = CGPoint(x: boardSize.width / 2, y: boardSize.height / 2)
                session.addNode(title: L("新しい機能", "New feature"), parent: session.selectedNodeID,
                                at: session.selectedNodeID == nil ? worldPoint(center, in: boardSize) : nil)
            } label: {
                Label(L("丸を足す", "Add"), systemImage: "plus.circle.fill")
            }
            Button { withAnimation(.spring) { session.autoLayout() }; fitAll() } label: {
                Label(L("並べる", "Arrange"), systemImage: "wand.and.stars")
            }
            Button(action: fitAll) {
                Label(L("全体", "Fit"), systemImage: "arrow.up.left.and.arrow.down.right")
            }
            Button { planning = true } label: {
                Label(L("AI に計画", "AI plan"), systemImage: "sparkles")
            }
            .disabled(session.isAgentRunning)
            RoadmapProgress(roadmap: session.roadmap)
        }
        .font(.callout)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: Capsule())
        .padding(12)
    }

    @ViewBuilder
    private var connectHint: some View {
        if connectFrom != nil {
            Label(L("つなぐ先の丸をタップ", "Tap the circle to connect to"), systemImage: "arrow.triangle.branch")
                .font(.callout)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.thinMaterial, in: Capsule())
                .padding(.top, 64)
        }
    }

    @ViewBuilder
    private var inspector: some View {
        if let node = session.selectedNode {
            RoadmapInspector(session: session, node: node,
                             onConnect: { connectFrom = node.id },
                             onAsk: ask)
                .frame(width: 300)
                .padding(12)
                .transition(.move(edge: .trailing).combined(with: .opacity))
        }
    }
}

/// 状態と種類ごとの見た目。
enum RoadmapStyle {
    static func color(for status: RoadmapNode.Status) -> Color {
        switch status {
        case .idea: return .yellow
        case .planned: return .blue
        case .inProgress: return .orange
        case .done: return .green
        case .blocked: return .gray
        }
    }

    static func diameter(for node: RoadmapNode) -> CGFloat {
        switch node.kind {
        case .goal: return 132
        case .feature: return 104
        case .idea: return 88
        case .bug: return 84
        case .note: return 76
        }
    }

    static func symbol(for kind: RoadmapNode.Kind) -> String {
        switch kind {
        case .goal: return "flag.fill"
        case .feature: return "puzzlepiece.fill"
        case .idea: return "lightbulb.fill"
        case .bug: return "ladybug.fill"
        case .note: return "note.text"
        }
    }
}

/// 丸 1 つ。型チェックを速くするため、色や線は先に計算しておく。
private struct RoadmapBubble: View {
    let node: RoadmapNode
    let isSelected: Bool
    let isConnectSource: Bool

    private var color: Color { RoadmapStyle.color(for: node.status) }
    private var diameter: CGFloat { RoadmapStyle.diameter(for: node) }
    private var isIdea: Bool { node.status == .idea }
    private var ringColor: Color { isSelected || isConnectSource ? Color.accentColor : Color.white.opacity(0.7) }
    private var ringStyle: StrokeStyle {
        let dash: [CGFloat] = isIdea ? [5, 4] : []
        return StrokeStyle(lineWidth: isSelected ? 4 : 1.5, dash: dash)
    }
    private var textColor: Color { isIdea || node.status == .done ? Color.black.opacity(0.8) : Color.white }
    private var badgeSymbol: String { node.status == .done ? "checkmark.circle.fill" : RoadmapStyle.symbol(for: node.kind) }
    private var shadowColor: Color { color.opacity(isSelected ? 0.7 : 0.35) }

    var body: some View {
        ZStack {
            Circle()
                .fill(color.gradient)
                .opacity(isIdea ? 0.75 : 0.95)
            Circle()
                .strokeBorder(ringColor, style: ringStyle)
            label
        }
        .frame(width: diameter, height: diameter)
        .shadow(color: shadowColor, radius: isSelected ? 12 : 5)
        .overlay { pulse }
    }

    private var label: some View {
        VStack(spacing: 3) {
            HStack(spacing: 3) {
                Image(systemName: badgeSymbol)
                Text(node.reference)
            }
            .font(.system(size: 10, weight: .bold))
            .opacity(0.85)
            Text(node.title)
                .font(.system(size: node.kind == .goal ? 14 : 12, weight: .semibold))
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .minimumScaleFactor(0.7)
        }
        .foregroundStyle(textColor)
        .padding(12)
    }

    @ViewBuilder
    private var pulse: some View {
        if node.status == .inProgress {
            Circle()
                .stroke(Color.orange.opacity(0.5), lineWidth: 2)
                .phaseAnimator([false, true]) { view, phase in
                    view.opacity(phase ? 0.1 : 0.9).scaleEffect(phase ? 1.25 : 1.08)
                } animation: { _ in .easeInOut(duration: 1.2) }
        }
    }
}

/// 丸どうしの線（親から子へ、曲線と矢印）。
private struct RoadmapEdges: View {
    let roadmap: Roadmap
    let position: (RoadmapNode) -> CGPoint
    let radius: (RoadmapNode) -> CGFloat

    var body: some View {
        Canvas { context, _ in
            for edge in roadmap.edges {
                guard let from = roadmap.node(edge.from), let to = roadmap.node(edge.to) else { continue }
                draw(from: from, to: to, in: &context)
            }
        }
        .allowsHitTesting(false)
    }

    private func draw(from: RoadmapNode, to: RoadmapNode, in context: inout GraphicsContext) {
        let start = position(from), end = position(to)
        let dx = end.x - start.x, dy = end.y - start.y
        let length = max(sqrt(dx * dx + dy * dy), 1)
        let ux = dx / length, uy = dy / length
        // 丸の縁から縁へ
        let a = CGPoint(x: start.x + ux * radius(from), y: start.y + uy * radius(from))
        let b = CGPoint(x: end.x - ux * (radius(to) + 4), y: end.y - uy * (radius(to) + 4))
        let control = CGPoint(x: (a.x + b.x) / 2 - uy * length * 0.12, y: (a.y + b.y) / 2 + ux * length * 0.12)
        var path = Path()
        path.move(to: a)
        path.addQuadCurve(to: b, control: control)
        let color = to.status == .done ? Color.green : Color.secondary
        context.stroke(path, with: .color(color.opacity(0.7)), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
        // 矢印
        let tx = b.x - control.x, ty = b.y - control.y
        let tLength = max(sqrt(tx * tx + ty * ty), 1)
        let vx = tx / tLength, vy = ty / tLength
        var arrow = Path()
        arrow.move(to: b)
        arrow.addLine(to: CGPoint(x: b.x - vx * 11 - vy * 6, y: b.y - vy * 11 + vx * 6))
        arrow.addLine(to: CGPoint(x: b.x - vx * 11 + vy * 6, y: b.y - vy * 11 - vx * 6))
        arrow.closeSubpath()
        context.fill(arrow, with: .color(color.opacity(0.8)))
    }
}

/// 背景の点。
private struct BoardGrid: View {
    let scale: CGFloat
    let offset: CGSize

    var body: some View {
        Canvas { context, size in
            let step = max(18, 40 * scale)
            let startX = (size.width / 2 + offset.width).truncatingRemainder(dividingBy: step)
            let startY = (size.height / 2 + offset.height).truncatingRemainder(dividingBy: step)
            var path = Path()
            var x = startX
            while x < size.width {
                var y = startY
                while y < size.height {
                    path.addEllipse(in: CGRect(x: x - 1, y: y - 1, width: 2, height: 2))
                    y += step
                }
                x += step
            }
            context.fill(path, with: .color(.secondary.opacity(0.25)))
        }
        .allowsHitTesting(false)
    }
}

private struct RoadmapLegend: View {
    var body: some View {
        HStack(spacing: 10) {
            ForEach(RoadmapNode.Status.allCases) { status in
                HStack(spacing: 4) {
                    Circle().fill(RoadmapStyle.color(for: status)).frame(width: 10, height: 10)
                    Text(status.label)
                }
            }
        }
        .font(.caption2)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.regularMaterial, in: Capsule())
    }
}

private struct RoadmapProgress: View {
    let roadmap: Roadmap

    var body: some View {
        let progress = roadmap.progress
        HStack(spacing: 6) {
            ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                .frame(width: 70)
            Text("\(progress.done)/\(progress.total)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }
}

/// 選んだ丸の詳細と操作。
private struct RoadmapInspector: View {
    @Bindable var session: WorkspaceSession
    let node: RoadmapNode
    var onConnect: () -> Void
    var onAsk: (String) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(node.reference).font(.headline.monospacedDigit()).foregroundStyle(.secondary)
                    Spacer()
                    Button { session.selectedNodeID = nil } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                }
                TextField(L("タイトル", "Title"), text: binding(\.title), axis: .vertical)
                    .font(.title3.weight(.semibold))
                pickers
                TextField(L("説明・メモ", "Description / notes"), text: binding(\.detail), axis: .vertical)
                    .lineLimit(3...10)
                    .textFieldStyle(.roundedBorder)
                RoadmapNodeLinks(session: session, node: node)
                actions
            }
            .padding(16)
        }
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18))
        .shadow(radius: 10)
    }

    private func binding(_ keyPath: WritableKeyPath<RoadmapNode, String>) -> Binding<String> {
        Binding(get: { session.roadmap.node(node.id)?[keyPath: keyPath] ?? "" },
                set: { value in session.updateNode(node.id) { $0[keyPath: keyPath] = value } })
    }

    private var pickers: some View {
        HStack {
            Picker(L("状態", "Status"), selection: Binding(get: { node.status },
                                                          set: { value in session.updateNode(node.id) { $0.status = value } })) {
                ForEach(RoadmapNode.Status.allCases) { Text($0.label).tag($0) }
            }
            Picker(L("種類", "Kind"), selection: Binding(get: { node.kind },
                                                        set: { value in session.updateNode(node.id) { $0.kind = value } })) {
                ForEach(RoadmapNode.Kind.allCases) { Label($0.label, systemImage: RoadmapStyle.symbol(for: $0)).tag($0) }
            }
        }
        .pickerStyle(.menu)
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                session.addNode(title: L("新しいアイデア", "New idea"), parent: node.id)
            } label: {
                Label(L("ここから案を生やす", "Branch a new idea"), systemImage: "arrow.triangle.branch")
            }
            Button { onAsk(CodingAgentPrompt.brainstorm(from: node)) } label: {
                Label(L("AI にアイデアを出してもらう", "Ask the AI for ideas"), systemImage: "lightbulb")
            }
            .disabled(session.isAgentRunning)
            Button {
                session.updateNode(node.id) { $0.status = .inProgress }
                onAsk(CodingAgentPrompt.implement(node))
            } label: {
                Label(L("AI に実装してもらう", "Let the AI implement it"), systemImage: "hammer.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(session.isAgentRunning)
            Button(action: onConnect) {
                Label(L("ほかの丸とつなぐ", "Connect to another"), systemImage: "link")
            }
            Button(role: .destructive) { session.removeNode(node.id) } label: {
                Label(L("この丸を消す", "Delete"), systemImage: "trash")
            }
        }
        .buttonStyle(.bordered)
        .font(.callout)
    }
}

/// つながっている丸と、関係するファイル。
private struct RoadmapNodeLinks: View {
    @Bindable var session: WorkspaceSession
    let node: RoadmapNode

    var body: some View {
        let parents = session.roadmap.parents(of: node.id)
        let children = session.roadmap.children(of: node.id)
        VStack(alignment: .leading, spacing: 6) {
            if !parents.isEmpty {
                linkRow(L("元", "From"), parents)
            }
            if !children.isEmpty {
                linkRow(L("派生", "Branches"), children)
            }
            ForEach(node.files, id: \.self) { file in
                Button {
                    session.open(file)
                    session.pane = .editor
                } label: {
                    Label(file, systemImage: "doc.text").font(.caption.monospaced())
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func linkRow(_ title: String, _ nodes: [RoadmapNode]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            ForEach(nodes) { other in
                Button { session.selectedNodeID = other.id } label: {
                    HStack(spacing: 6) {
                        Circle().fill(RoadmapStyle.color(for: other.status)).frame(width: 8, height: 8)
                        Text("\(other.reference) \(other.title)").lineLimit(1)
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
            }
        }
    }
}
