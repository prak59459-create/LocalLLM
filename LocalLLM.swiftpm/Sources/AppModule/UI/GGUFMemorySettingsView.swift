import SwiftUI

extension GGUFMemoryOptions.Residency {
    var label: String {
        switch self {
        case .automatic: return L("自動（OS に任せる）", "Automatic (let the OS decide)")
        case .preload: return L("先に全部読む", "Preload everything")
        case .activeOnly: return L("使う分だけ置く", "Keep only what is in use")
        }
    }

    var explanation: String {
        switch self {
        case .automatic:
            return L("読んだ重みは RAM に残り、足りなくなると OS が捨てます。ふだんはこれで十分です。",
                     "Weights stay in RAM once read; the OS drops them when memory runs low. Fine for most cases.")
        case .preload:
            return L("読み込みのときに重み全体を先に読みます。最初の返事が速くなりますが、RAM を多く使います。",
                     "Reads all weights while loading. The first reply is faster, but more RAM is used.")
        case .activeOnly:
            return L("使い終わった重みをすぐ手放します。MoE は最近使った専門家だけ、普通のモデルは先頭の層だけを RAM に残し、ほかは毎回ファイルから読みます。RAM より大きなモデルも動かせますが、ファイルを読む分だけ遅くなります（CPU で計算します）。",
                     "Drops weights right after use. MoE models keep only recently used experts, other models keep only the first layers in RAM and read the rest from storage each time. Lets models larger than RAM run, but slower (computed on the CPU).")
        }
    }
}

/// 設定の「GGUF のメモリ（上級）」への入り口
struct GGUFMemorySettingsLink: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        Section {
            NavigationLink {
                GGUFMemorySettingsView()
            } label: {
                LabeledContent {
                    Text(settings.ggufMemory.residency.label)
                } label: {
                    Label(L("GGUF のメモリ（上級）", "GGUF memory (advanced)"), systemImage: "memorychip")
                }
            }
        }
    }
}

/// GGUF のモデルの重みを、メモリにどう置くかの設定
struct GGUFMemorySettingsView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(ModelLibrary.self) private var library
    @Environment(ChatController.self) private var chat

    var body: some View {
        Form {
            ResidencySection()
            MixtureSection()
            DenseSection()
            if let id = chat.loadedInfo?.modelID {
                Section {
                    Button(L("今のモデルを読み込み直して反映", "Reload current model to apply")) {
                        chat.loadModel(id: id, library: library, settings: settings)
                    }
                }
            }
        }
        .navigationTitle(L("GGUF のメモリ", "GGUF memory"))
    }
}

private struct ResidencySection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            Picker(L("重みの置き方", "Weights in memory"), selection: $settings.ggufMemory.residency) {
                ForEach(GGUFMemoryOptions.Residency.allCases) { option in
                    Text(option.label).tag(option)
                }
            }
        } header: {
            Text(L("置き方", "Residency"))
        } footer: {
            Text(settings.ggufMemory.residency.explanation + L(" 変更は次の読み込みから有効です。", " Changes apply on the next load."))
        }
    }
}

private struct MixtureSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            Picker(L("1 トークンで使う専門家", "Experts per token"), selection: $settings.ggufMemory.expertsUsed) {
                Text(L("モデルの既定", "Model default")).tag(0)
                ForEach(1...16, id: \.self) { count in Text("\(count)").tag(count) }
            }
            Picker(L("RAM に残す専門家（層ごと）", "Experts kept in RAM (per layer)"), selection: $settings.ggufMemory.keptExperts) {
                Text(L("自動（使う数の 2 倍）", "Automatic (2× experts used)")).tag(0)
                ForEach([1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64], id: \.self) { count in Text("\(count)").tag(count) }
            }
            .disabled(settings.ggufMemory.residency != .activeOnly)
        } header: {
            Text(L("MoE（専門家の混合）", "Mixture of experts"))
        } footer: {
            Text(L("MoE のモデルは、トークンごとに一部の専門家（FFN）だけを使います。使う専門家を減らすと速く・読み込む量も減りますが、答えの質は下がります。「RAM に残す専門家」は「使う分だけ置く」のときに使います。",
                   "MoE models use only a few experts (FFNs) per token. Fewer experts is faster and reads less, but lowers quality. \"Experts kept in RAM\" applies with \"Keep only what is in use\"."))
        }
    }
}

private struct DenseSection: View {
    @Environment(AppSettings.self) private var settings

    var body: some View {
        @Bindable var settings = settings
        Section {
            Picker(L("RAM に残す層", "Layers kept in RAM"), selection: $settings.ggufMemory.keptLayers) {
                Text(L("自動（空きメモリから）", "Automatic (from free memory)")).tag(-1)
                ForEach([0, 2, 4, 8, 12, 16, 24, 32, 48, 64], id: \.self) { count in Text("\(count)").tag(count) }
            }
            .disabled(settings.ggufMemory.residency != .activeOnly)
        } header: {
            Text(L("普通のモデル", "Dense models"))
        } footer: {
            Text(L("「使う分だけ置く」のとき、先頭からこの数の層を RAM に残し、残りの層は毎回ファイルから読みます。多いほど速く、少ないほど RAM を使いません。",
                   "With \"Keep only what is in use\", this many layers from the start stay in RAM; the rest are read from storage every time. More is faster, fewer uses less RAM."))
        }
    }
}
