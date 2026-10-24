import SwiftUI

/// コーディングワークスペース（CDWS）の見た目: 暗い IDE 風の色と、共通の部品。
enum CDWSTheme {
    // 背景（奥から手前へ）
    static let canvas = Color(red: 0.055, green: 0.067, blue: 0.090)
    static let panel = Color(red: 0.086, green: 0.106, blue: 0.133)
    static let raised = Color(red: 0.118, green: 0.141, blue: 0.180)
    static let stroke = Color.white.opacity(0.08)
    static let strongStroke = Color.white.opacity(0.16)

    // 文字
    static let text = Color(red: 0.90, green: 0.93, blue: 0.97)
    static let secondaryText = Color(red: 0.56, green: 0.61, blue: 0.69)

    // アクセント
    static let violet = Color(red: 0.49, green: 0.42, blue: 1.0)
    static let cyan = Color(red: 0.16, green: 0.83, blue: 0.93)
    static let pink = Color(red: 0.96, green: 0.38, blue: 0.75)
    static let green = Color(red: 0.25, green: 0.73, blue: 0.31)
    static let orange = Color(red: 0.94, green: 0.53, blue: 0.24)
    static let red = Color(red: 0.97, green: 0.32, blue: 0.29)

    static var accent: LinearGradient {
        LinearGradient(colors: [violet, cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    static var hero: LinearGradient {
        LinearGradient(colors: [violet.opacity(0.55), pink.opacity(0.30), cyan.opacity(0.35)],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    static let codeFont = Font.system(.callout, design: .monospaced)
}

/// パネルの背景（角丸・細い枠）
struct CDWSPanelStyle: ViewModifier {
    var radius: CGFloat = 14
    var fill: Color = CDWSTheme.panel

    func body(content: Content) -> some View {
        content
            .background(fill, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).strokeBorder(CDWSTheme.stroke))
    }
}

extension View {
    func cdwsPanel(radius: CGFloat = 14, fill: Color = CDWSTheme.panel) -> some View {
        modifier(CDWSPanelStyle(radius: radius, fill: fill))
    }
}

/// グラデーションの角丸の中にアイコン
struct CDWSIconTile: View {
    let symbol: String
    var size: CGFloat = 40

    var body: some View {
        RoundedRectangle(cornerRadius: size * 0.28, style: .continuous)
            .fill(CDWSTheme.accent)
            .frame(width: size, height: size)
            .overlay(Image(systemName: symbol).font(.system(size: size * 0.42, weight: .semibold)).foregroundStyle(.white))
            .shadow(color: CDWSTheme.violet.opacity(0.35), radius: 8, y: 3)
    }
}

/// 動いている間、枠がゆっくり回って光る
struct CDWSGlow: ViewModifier {
    var isActive: Bool
    var radius: CGFloat = 14
    @State private var angle: Double = 0

    func body(content: Content) -> some View {
        content.overlay {
            if isActive {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(glowGradient, lineWidth: 1.5)
                    .onAppear {
                        withAnimation(.linear(duration: 3).repeatForever(autoreverses: false)) { angle = 360 }
                    }
            }
        }
    }

    private var glowGradient: AngularGradient {
        AngularGradient(colors: [CDWSTheme.violet, CDWSTheme.cyan, CDWSTheme.pink, CDWSTheme.violet],
                        center: .center, angle: .degrees(angle))
    }
}

extension View {
    func cdwsGlow(_ isActive: Bool, radius: CGFloat = 14) -> some View {
        modifier(CDWSGlow(isActive: isActive, radius: radius))
    }
}

/// 小さな丸い札（状態などの表示）
struct CDWSChip: View {
    let text: String
    var symbol: String?
    var color: Color = CDWSTheme.secondaryText

    var body: some View {
        HStack(spacing: 4) {
            if let symbol { Image(systemName: symbol).font(.caption2.weight(.bold)) }
            Text(text).font(.caption2.weight(.semibold)).lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .foregroundStyle(color)
        .background(color.opacity(0.14), in: Capsule())
    }
}

/// 横長のメーター（使った量 / 全体）
struct CDWSMeter: View {
    let label: String
    let fraction: Double
    let detail: String

    private var color: Color {
        fraction > 0.85 ? CDWSTheme.red : (fraction > 0.65 ? CDWSTheme.orange : CDWSTheme.cyan)
    }

    var body: some View {
        HStack(spacing: 6) {
            Text(label).foregroundStyle(CDWSTheme.secondaryText)
            Capsule()
                .fill(CDWSTheme.raised)
                .frame(width: 54, height: 5)
                .overlay(alignment: .leading) {
                    Capsule().fill(color).frame(width: 54 * min(max(fraction, 0.02), 1), height: 5)
                }
            Text(detail).monospacedDigit().foregroundStyle(CDWSTheme.text)
        }
        .font(.caption2)
    }
}

/// 上部の切り替え（エディタ・プレビュー・ロードマップ・変更履歴）。選んだものに光る丸が動いていく。
struct CDWSPaneSwitcher: View {
    @Binding var selection: WorkspacePane
    @Namespace private var namespace

    var body: some View {
        HStack(spacing: 4) {
            ForEach(WorkspacePane.allCases) { pane in
                CDWSPaneButton(pane: pane, isSelected: pane == selection, namespace: namespace) {
                    withAnimation(.spring(response: 0.32, dampingFraction: 0.82)) { selection = pane }
                }
            }
        }
        .padding(4)
        .background(CDWSTheme.panel, in: Capsule())
        .overlay(Capsule().strokeBorder(CDWSTheme.stroke))
    }
}

private struct CDWSPaneButton: View {
    let pane: WorkspacePane
    let isSelected: Bool
    let namespace: Namespace.ID
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(pane.label, systemImage: pane.symbolName)
                .font(.footnote.weight(.semibold))
                .labelStyle(.titleAndIcon)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .foregroundStyle(isSelected ? Color.white : CDWSTheme.secondaryText)
                .background { selectedBackground }
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var selectedBackground: some View {
        if isSelected {
            Capsule().fill(CDWSTheme.accent).matchedGeometryEffect(id: "pane", in: namespace)
        }
    }
}

/// 点を並べた背景（ロードマップの盤面）
struct CDWSDotGrid: View {
    var spacing: CGFloat = 26

    var body: some View {
        Canvas { context, size in
            context.fill(dots(in: size), with: .color(.white.opacity(0.07)))
        }
        .background(CDWSTheme.canvas)
    }

    private func dots(in size: CGSize) -> Path {
        var path = Path()
        var x: CGFloat = spacing / 2
        while x < size.width {
            var y: CGFloat = spacing / 2
            while y < size.height {
                path.addEllipse(in: CGRect(x: x, y: y, width: 1.6, height: 1.6))
                y += spacing
            }
            x += spacing
        }
        return path
    }
}
