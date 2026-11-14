import SwiftUI
import UIKit

/// 画像生成の画面（文章から画像を作る）。全画面で表示する。
struct ImageStudioView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ImageStudio.self) private var studio
    @Environment(ImageModelStore.self) private var store

    var body: some View {
        NavigationStack {
            Form {
                ImageModelSection()
                if !store.completedModels.isEmpty {
                    ImagePromptSection()
                    ImageOptionsSection()
                    ImageGenerateSection()
                    ImageResultSection()
                    ImageGallerySection()
                }
            }
            .navigationTitle(L("画像生成", "Image Generation"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close")) { dismiss() }
                }
            }
        }
    }
}

/// 使うモデルの選択と、ダウンロード。
private struct ImageModelSection: View {
    @Environment(ImageStudio.self) private var studio
    @Environment(ImageModelStore.self) private var store
    @Environment(AppSettings.self) private var settings
    @State private var customRepo = ""

    var body: some View {
        Section {
            ForEach(ImageModelSpec.catalog) { spec in
                ImageModelRow(repoID: spec.repoID, title: spec.title, note: spec.note, approximateBytes: spec.approximateBytes)
            }
            ForEach(otherModels) { model in
                ImageModelRow(repoID: model.repoID, title: model.title, note: model.repoID, approximateBytes: model.totalBytes)
            }
            customRow
        } header: {
            Text(L("モデル", "Model"))
        } footer: {
            Text(L("Core ML 版の Stable Diffusion（Apple の ml-stable-diffusion で変換したもの）に対応しています。初めて使うときは、端末向けの準備に数分かかります。生成中は、メモリを空けるために文章のモデルを解放します。",
                   "Core ML builds of Stable Diffusion (converted with Apple's ml-stable-diffusion) are supported. The first run takes a few minutes to prepare for the device. The chat model is unloaded while generating to free memory."))
        }
    }

    /// 一覧に無いリポジトリから入れたモデル
    private var otherModels: [InstalledImageModel] {
        let known = Set(ImageModelSpec.catalog.map(\.repoID))
        return store.models.filter { !known.contains($0.repoID) }
    }

    private var customRow: some View {
        HStack {
            TextField(L("ほかのリポジトリ（owner/name）", "Another repository (owner/name)"), text: $customRepo)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button(L("入手", "Get")) {
                let repo = customRepo.trimmingCharacters(in: .whitespaces)
                store.startDownload(repoID: repo, title: repo.split(separator: "/").last.map(String.init) ?? repo,
                                    client: settings.makeHFClient())
                customRepo = ""
            }
            .disabled(!customRepo.contains("/"))
        }
    }
}

private struct ImageModelRow: View {
    @Environment(ImageStudio.self) private var studio
    @Environment(ImageModelStore.self) private var store
    @Environment(AppSettings.self) private var settings

    let repoID: String
    let title: String
    let note: String
    let approximateBytes: Int64

    private var installed: InstalledImageModel? { store.model(repoID: repoID) }
    private var isSelected: Bool {
        guard installed?.isComplete == true else { return false }
        return studio.selectedRepoID == repoID || (studio.selectedRepoID == nil && store.completedModels.first?.repoID == repoID)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    Text(note).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text(ByteFormat.string(installed?.totalBytes ?? approximateBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            if let state = store.downloads[repoID] {
                DownloadStatusView(state: state)
            }
            ImageModelButtons(repoID: repoID, title: title, isSelected: isSelected)
        }
        .padding(.vertical, 2)
    }
}

private struct ImageModelButtons: View {
    @Environment(ImageStudio.self) private var studio
    @Environment(ImageModelStore.self) private var store
    @Environment(AppSettings.self) private var settings

    let repoID: String
    let title: String
    let isSelected: Bool

    private var installed: InstalledImageModel? { store.model(repoID: repoID) }
    private var isDownloading: Bool { store.downloads[repoID]?.phase == .downloading }

    var body: some View {
        HStack {
            if installed?.isComplete == true {
                Button {
                    studio.selectedRepoID = repoID
                } label: {
                    Label(isSelected ? L("使用中", "Selected") : L("使う", "Use"), systemImage: isSelected ? "checkmark.circle.fill" : "circle")
                }
                .buttonStyle(.bordered)
                .disabled(isSelected)
            } else if isDownloading {
                Button {
                    store.pauseDownload(repoID: repoID)
                } label: {
                    Label(L("一時停止", "Pause"), systemImage: "pause.fill")
                }
                .buttonStyle(.bordered)
            } else {
                Button {
                    store.startDownload(repoID: repoID, title: title, client: settings.makeHFClient())
                } label: {
                    Label(installed == nil ? L("ダウンロード", "Download") : L("再開", "Resume"), systemImage: "arrow.down.circle.fill")
                }
                .buttonStyle(.borderedProminent)
            }
            Spacer()
            if installed != nil {
                Button(role: .destructive) {
                    store.delete(repoID: repoID)
                } label: {
                    Label(L("削除", "Delete"), systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .disabled(studio.isGenerating)
            }
        }
    }
}

/// プロンプトの入力。
private struct ImagePromptSection: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        Section {
            TextField(L("描いてほしいもの（英語）", "What to draw (English)"), text: $studio.prompt, axis: .vertical)
                .lineLimit(3...8)
            TextField(L("描いてほしくないもの（任意）", "What to avoid (optional)"), text: $studio.negativePrompt, axis: .vertical)
                .lineLimit(1...4)
            ImageTranslateButton()
        } header: {
            Text(L("プロンプト", "Prompt"))
        } footer: {
            Text(L("このモデルは英語で学習しているので、英語で書くときれいに出ます（例: a watercolor painting of a cat sitting on a windowsill, soft light）。日本語のままでは、うまく伝わりません。",
                   "The model was trained on English, so write in English (for example: a watercolor painting of a cat sitting on a windowsill, soft light)."))
        }
    }
}

/// 日本語などで書いたプロンプトを、読み込んである文章のモデルで英語にするボタン。
private struct ImageTranslateButton: View {
    @Environment(ImageStudio.self) private var studio
    @Environment(ChatController.self) private var chat

    var body: some View {
        if chat.loadedInfo != nil {
            Button {
                studio.translatePrompt(chat: chat)
            } label: {
                if studio.isTranslating {
                    HStack {
                        ProgressView()
                        Text(L("英語のプロンプトにしています…", "Rewriting as an English prompt…"))
                    }
                } else {
                    Label(L("文章のモデルで英語のプロンプトにする", "Rewrite as an English prompt with the chat model"),
                          systemImage: "character.bubble")
                }
            }
            .disabled(studio.isTranslating || studio.isGenerating || chat.isGenerating
                      || studio.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }
}

/// 回数・強さ・種。
private struct ImageOptionsSection: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        Section {
            Stepper(value: $studio.stepCount, in: 4...50) {
                LabeledContent(L("回数", "Steps"), value: "\(studio.stepCount)")
            }
            VStack(alignment: .leading) {
                LabeledContent(L("プロンプトに従う強さ", "Guidance"), value: String(format: "%.1f", studio.guidanceScale))
                Slider(value: $studio.guidanceScale, in: 1...15, step: 0.5)
            }
            Toggle(L("毎回ちがう絵にする（種をランダムに）", "Different picture each time (random seed)"), isOn: $studio.randomSeed)
            if !studio.randomSeed {
                ImageSeedField()
            }
        } header: {
            Text(L("設定", "Options"))
        } footer: {
            Text(L("回数を増やすと細かくなりますが、その分だけ時間がかかります（20 回前後がおすすめ）。同じ種・同じプロンプトなら、同じ絵になります。",
                   "More steps add detail but take longer (about 20 is a good default). The same seed and prompt give the same picture."))
        }
    }
}

private struct ImageSeedField: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        LabeledContent(L("種", "Seed")) {
            TextField("42", text: Binding(
                get: { String(studio.seed) },
                set: { studio.seed = UInt64($0.filter(\.isNumber)) ?? 0 }
            ))
            .keyboardType(.numberPad)
            .multilineTextAlignment(.trailing)
        }
    }
}

/// 生成ボタンと進み具合。
private struct ImageGenerateSection: View {
    @Environment(ImageStudio.self) private var studio
    @Environment(ImageModelStore.self) private var store
    @Environment(ChatController.self) private var chat

    var body: some View {
        Section {
            if studio.isGenerating {
                ImageProgressView()
                Button(role: .destructive) {
                    studio.cancel()
                } label: {
                    Label(L("止める", "Stop"), systemImage: "stop.fill")
                }
            } else {
                Button {
                    studio.generate(store: store, chat: chat)
                } label: {
                    Label(L("画像を作る", "Generate"), systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!studio.canGenerate)
            }
            if let message = studio.errorMessage {
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
    }
}

private struct ImageProgressView: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let step = studio.step, step.total > 0 {
                ProgressView(value: Double(step.done), total: Double(step.total))
            } else {
                ProgressView()
            }
            if let status = studio.status {
                Text(status).font(.footnote).foregroundStyle(.secondary)
            }
        }
    }
}

/// できた画像と、その操作。
private struct ImageResultSection: View {
    @Environment(ImageStudio.self) private var studio
    @State private var savedMessage: String?

    var body: some View {
        if let image = studio.current, let picture = studio.uiImage(for: image) {
            Section {
                Image(uiImage: picture)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
                ImageDetailText(image: image)
                ImageActionRow(image: image, picture: picture, savedMessage: $savedMessage)
                if let savedMessage {
                    Text(savedMessage).font(.footnote).foregroundStyle(.secondary)
                }
            }
        }
    }
}

private struct ImageDetailText: View {
    let image: GeneratedImage

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(image.prompt).font(.footnote)
            Text(L("\(image.modelTitle)・\(image.stepCount) 回・強さ \(String(format: "%.1f", image.guidanceScale))・種 \(String(image.seed))・\(Int(image.seconds)) 秒",
                   "\(image.modelTitle) · \(image.stepCount) steps · guidance \(String(format: "%.1f", image.guidanceScale)) · seed \(String(image.seed)) · \(Int(image.seconds)) s"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .textSelection(.enabled)
    }
}

private struct ImageActionRow: View {
    @Environment(ImageStudio.self) private var studio
    let image: GeneratedImage
    let picture: UIImage
    @Binding var savedMessage: String?

    var body: some View {
        HStack {
            ShareLink(item: studio.url(for: image)) {
                Label(L("共有", "Share"), systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
            Button {
                UIImageWriteToSavedPhotosAlbum(picture, nil, nil, nil)
                savedMessage = L("写真に保存しました。", "Saved to Photos.")
            } label: {
                Label(L("写真に保存", "Save to Photos"), systemImage: "photo.badge.plus")
            }
            .buttonStyle(.bordered)
            Button {
                studio.reuseSettings(of: image)
            } label: {
                Label(L("この設定を使う", "Reuse settings"), systemImage: "arrow.uturn.backward")
            }
            .buttonStyle(.bordered)
            Spacer()
            Button(role: .destructive) {
                studio.delete(image)
                savedMessage = nil
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.bordered)
        }
        .labelStyle(.iconOnly)
    }
}

/// これまでに作った画像。
private struct ImageGallerySection: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        if studio.images.count > 1 {
            Section(L("これまでに作った画像", "Gallery")) {
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: 8) {
                        ForEach(studio.images) { image in
                            ImageThumbnail(image: image)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }
}

private struct ImageThumbnail: View {
    @Environment(ImageStudio.self) private var studio
    let image: GeneratedImage

    var body: some View {
        Button {
            studio.currentID = image.id
        } label: {
            thumbnail
                .frame(width: 88, height: 88)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(studio.currentID == image.id ? Color.accentColor : Color.clear, lineWidth: 3))
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var thumbnail: some View {
        if let picture = studio.uiImage(for: image) {
            Image(uiImage: picture).resizable().scaledToFill()
        } else {
            Color.secondary.opacity(0.2)
        }
    }
}
