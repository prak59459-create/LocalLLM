import AVKit
import SwiftUI
import UIKit

/// 画像を作るか、動画を作るか。
struct StudioModeSection: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        Section {
            Picker(L("作るもの", "Create"), selection: $studio.mode) {
                Label(L("画像", "Image"), systemImage: "photo").tag(StudioMode.image)
                Label(L("動画", "Video"), systemImage: "film").tag(StudioMode.video)
            }
            .pickerStyle(.segmented)
            .disabled(studio.isGenerating)
        } footer: {
            if studio.mode == .video {
                Text(L("画像モデルで少しずつ違うコマを何枚も描き、その間をなめらかにつないで短い動画（MP4）にします。コマの数だけ画像を作るので、画像 1 枚の数倍〜十数倍の時間がかかります。",
                       "Draws several slightly different frames with the image model and joins them smoothly into a short MP4. It takes several times longer than one image."))
            }
        }
    }
}

/// 動画の作り方。
struct VideoOptionsSection: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        Section {
            Picker(L("動き", "Motion"), selection: $studio.videoSettings.motion) {
                ForEach(VideoMotion.allCases) { motion in
                    Text(motion.label).tag(motion)
                }
            }
            Stepper(value: $studio.videoSettings.keyframes, in: 2...32) {
                LabeledContent(L("描くコマ", "Keyframes"), value: "\(studio.videoSettings.keyframes)")
            }
            Stepper(value: $studio.videoSettings.inBetween, in: 0...6) {
                LabeledContent(L("間に挟むコマ", "In-between frames"), value: "\(studio.videoSettings.inBetween)")
            }
            Stepper(value: $studio.videoSettings.fps, in: 2...30) {
                LabeledContent(L("1 秒あたりのコマ", "Frames per second"), value: "\(studio.videoSettings.fps)")
            }
            if studio.videoSettings.motion == .camera {
                VideoCameraControls()
            }
            LabeledContent(L("できる動画", "Result"),
                           value: L("\(studio.videoSettings.frameCount) コマ・\(String(format: "%.1f", studio.videoSettings.duration)) 秒",
                                    "\(studio.videoSettings.frameCount) frames · \(String(format: "%.1f", studio.videoSettings.duration)) s"))
        } header: {
            Text(L("動画", "Video"))
        } footer: {
            Text(studio.videoSettings.motion == .camera
                 ? L("カメラ: 1 コマ目を描いたあと、前のコマを少し動かしてから描き直します。描き直す強さを上げると変化が大きく、下げるとなめらかになります。",
                     "Camera: after the first frame, each frame moves the previous one a little and redraws it. A higher strength changes more; a lower one is smoother.")
                 : L("変化: 最初のプロンプトから最後のプロンプトへ、少しずつ変わっていくコマを描きます（最後のプロンプトが空なら、同じプロンプトのまま絵が揺れ動きます）。",
                     "Morph: frames change gradually from the first prompt to the last (with no last prompt, the same prompt drifts)."))
        }
    }
}

private struct VideoCameraControls: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        @Bindable var studio = studio
        VideoSlider(title: L("ズーム（1 コマごと）", "Zoom per frame"), value: $studio.videoSettings.zoom, range: 0.9...1.15,
                    format: "×%.2f")
        VideoSlider(title: L("横に移動", "Pan horizontally"), value: $studio.videoSettings.panX, range: -0.1...0.1, format: "%.2f")
        VideoSlider(title: L("縦に移動", "Pan vertically"), value: $studio.videoSettings.panY, range: -0.1...0.1, format: "%.2f")
        VideoSlider(title: L("回転（度）", "Rotation (degrees)"), value: $studio.videoSettings.rotation, range: -10...10, format: "%.1f°")
        VideoSlider(title: L("描き直す強さ", "Redraw strength"), value: $studio.videoSettings.strength, range: 0.2...0.8, format: "%.2f")
    }
}

private struct VideoSlider: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String

    var body: some View {
        VStack(alignment: .leading) {
            LabeledContent(title, value: String(format: format, value))
            Slider(value: $value, in: range)
        }
    }
}

/// 動画を作っている途中の、できたキーフレーム。
struct VideoProgressPreview: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        if studio.isGenerating, let preview = studio.videoPreview {
            Section(L("できたコマ", "Latest frame")) {
                Image(uiImage: preview)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                if let frames = studio.keyframeProgress {
                    Text(L("\(frames.done)/\(frames.total) コマ", "\(frames.done)/\(frames.total) frames"))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// できた動画と、その操作。
struct VideoResultSection: View {
    @Environment(ImageStudio.self) private var studio
    @State private var savedMessage: String?

    var body: some View {
        if let video = studio.currentVideo {
            Section {
                LoopingVideoPlayer(url: studio.url(for: video))
                    .aspectRatio(1, contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
                VideoDetailText(video: video)
                VideoActionRow(video: video, savedMessage: $savedMessage)
                if let savedMessage {
                    Text(savedMessage).font(.footnote).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// 繰り返し再生する動画
private struct LoopingVideoPlayer: View {
    let url: URL
    @State private var player: AVQueuePlayer?
    @State private var looper: AVPlayerLooper?

    var body: some View {
        VideoPlayer(player: player)
            .onAppear(perform: start)
            .onChange(of: url) { _, _ in start() }
            .onDisappear { player?.pause() }
    }

    private func start() {
        let queue = AVQueuePlayer()
        looper = AVPlayerLooper(player: queue, templateItem: AVPlayerItem(url: url))
        player = queue
        queue.play()
    }
}

private struct VideoDetailText: View {
    let video: GeneratedVideo

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(video.prompt).font(.footnote)
            if !video.endPrompt.isEmpty {
                Text("→ " + video.endPrompt).font(.footnote)
            }
            Text(L("\(video.modelTitle)・\(video.settings.motion.label)・\(video.settings.frameCount) コマ・種 \(String(video.seed))・\(Int(video.seconds)) 秒",
                   "\(video.modelTitle) · \(video.settings.motion.label) · \(video.settings.frameCount) frames · seed \(String(video.seed)) · \(Int(video.seconds)) s"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .textSelection(.enabled)
    }
}

private struct VideoActionRow: View {
    @Environment(ImageStudio.self) private var studio
    let video: GeneratedVideo
    @Binding var savedMessage: String?

    var body: some View {
        HStack {
            ShareLink(item: studio.url(for: video)) {
                Label(L("共有", "Share"), systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
            Button {
                let path = studio.url(for: video).path
                if UIVideoAtPathIsCompatibleWithSavedPhotosAlbum(path) {
                    UISaveVideoAtPathToSavedPhotosAlbum(path, nil, nil, nil)
                    savedMessage = L("写真に保存しました。", "Saved to Photos.")
                } else {
                    savedMessage = L("この動画は写真に保存できません。", "This video cannot be saved to Photos.")
                }
            } label: {
                Label(L("写真に保存", "Save to Photos"), systemImage: "photo.badge.plus")
            }
            .buttonStyle(.bordered)
            Button {
                studio.reuseSettings(of: video)
            } label: {
                Label(L("この設定を使う", "Reuse settings"), systemImage: "arrow.uturn.backward")
            }
            .buttonStyle(.bordered)
            Spacer()
            Button(role: .destructive) {
                studio.delete(video)
                savedMessage = nil
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.bordered)
        }
        .labelStyle(.iconOnly)
    }
}

/// これまでに作った動画。
struct VideoGallerySection: View {
    @Environment(ImageStudio.self) private var studio

    var body: some View {
        if studio.videos.count > 1 {
            Section(L("これまでに作った動画", "Videos")) {
                ForEach(studio.videos) { video in
                    Button {
                        studio.currentVideoID = video.id
                    } label: {
                        HStack {
                            Image(systemName: studio.currentVideoID == video.id ? "play.circle.fill" : "film")
                            VStack(alignment: .leading) {
                                Text(video.prompt).lineLimit(1)
                                Text(video.createdAt, style: .date).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }
}
