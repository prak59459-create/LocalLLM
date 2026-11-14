import SwiftUI
import UniformTypeIdentifiers

/// 入力欄のマイクのボタン。タップで音声入力を始める・止める。長押し（メニュー）で音声の設定。
struct VoiceButton: View {
    @Environment(VoiceController.self) private var voice
    /// 聞き取った文章を入れる先（入力欄）
    @Binding var text: String
    /// 声だけの会話のとき、話し終わったら送る
    var onSend: () -> Void
    @State private var showingTranscription = false

    var body: some View {
        Menu {
            VoiceMenuItems(showingTranscription: $showingTranscription)
        } label: {
            Image(systemName: voice.isListening ? "mic.fill" : "mic")
                .font(.system(size: 19))
                .foregroundStyle(voice.isListening ? Color.red : Color.secondary)
                .frame(width: 34, height: 40)
                .symbolEffect(.pulse, isActive: voice.isListening)
        } primaryAction: {
            toggle()
        }
        .accessibilityLabel(voice.isListening ? L("音声入力を止める", "Stop dictation") : L("音声入力", "Dictate"))
        .sheet(isPresented: $showingTranscription) {
            TranscriptionView(onInsert: { text = text.isEmpty ? $0 : text + "\n" + $0 })
        }
    }

    private func toggle() {
        if voice.isListening {
            voice.stopListening()
            return
        }
        let prefix = text.isEmpty ? "" : text + " "
        Task {
            await voice.startListening(onUpdate: { heard in
                text = prefix + heard
            }, onFinish: { heard in
                if !heard.isEmpty { text = prefix + heard }
                if voice.handsFree, !heard.isEmpty { onSend() }
            })
        }
    }
}

private struct VoiceMenuItems: View {
    @Environment(VoiceController.self) private var voice
    @Binding var showingTranscription: Bool

    var body: some View {
        @Bindable var voice = voice
        Toggle(isOn: $voice.autoSpeak) {
            Label(L("返答を読み上げる", "Read replies aloud"), systemImage: "speaker.wave.2")
        }
        Toggle(isOn: $voice.handsFree) {
            Label(L("声だけで会話（話し終わったら自動で送る）", "Voice chat (send when you stop talking)"), systemImage: "waveform")
        }
        Picker(selection: $voice.language) {
            ForEach(VoiceLanguage.allCases) { language in
                Text(language.label).tag(language)
            }
        } label: {
            Label(L("聞き取る言語", "Dictation language"), systemImage: "globe")
        }
        Picker(selection: $voice.speechRate) {
            Text(L("ゆっくり", "Slow")).tag(0.8)
            Text(L("標準", "Normal")).tag(1.0)
            Text(L("速い", "Fast")).tag(1.15)
            Text(L("とても速い", "Very fast")).tag(1.3)
        } label: {
            Label(L("読み上げの速さ", "Speech rate"), systemImage: "speedometer")
        }
        Divider()
        if voice.isSpeaking {
            Button {
                voice.stopSpeaking()
            } label: {
                Label(L("読み上げを止める", "Stop speaking"), systemImage: "speaker.slash")
            }
        }
        Button {
            showingTranscription = true
        } label: {
            Label(L("音声ファイルを文字にする…", "Transcribe an audio file…"), systemImage: "doc.text.below.ecg")
        }
    }
}

/// 返答が終わったら読み上げ、声だけの会話なら続けて聞き取る。チャット画面に付ける。
struct VoiceReplyModifier: ViewModifier {
    @Environment(VoiceController.self) private var voice
    @Environment(ChatController.self) private var chat
    /// いま終わった返答の本文を返す
    var latestReply: () -> String?
    /// もう一度聞き取りを始める（声だけの会話）
    var listenAgain: () -> Void

    func body(content: Content) -> some View {
        content
            .onChange(of: chat.isGenerating) { wasGenerating, isGenerating in
                guard wasGenerating, !isGenerating else { return }
                replyFinished()
            }
            .onDisappear {
                voice.stopListening()
                voice.stopSpeaking()
            }
            .overlay(alignment: .top) {
                if let message = voice.errorMessage {
                    NoticeBar(text: message) { voice.errorMessage = nil }
                }
            }
    }

    private func replyFinished() {
        guard voice.autoSpeak || voice.handsFree, let reply = latestReply(), !reply.isEmpty else { return }
        voice.speak(reply) {
            if voice.handsFree { listenAgain() }
        }
    }
}

/// メッセージのメニューに入れる「読み上げる」。
struct SpeakMessageButton: View {
    @Environment(VoiceController.self) private var voice
    let text: String

    var body: some View {
        Button {
            if voice.isSpeaking {
                voice.stopSpeaking()
            } else {
                voice.speak(text)
            }
        } label: {
            Label(voice.isSpeaking ? L("読み上げを止める", "Stop speaking") : L("読み上げる", "Read aloud"),
                  systemImage: voice.isSpeaking ? "speaker.slash" : "speaker.wave.2")
        }
    }
}

/// 音声ファイルを文字にする画面。
struct TranscriptionView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(VoiceController.self) private var voice
    /// 文字にした結果を入力欄に入れる
    var onInsert: (String) -> Void

    @State private var isImporting = false
    @State private var isWorking = false
    @State private var fileName: String?
    @State private var result = ""
    @State private var errorMessage: String?
    @State private var language: VoiceLanguage = .automatic

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker(L("言語", "Language"), selection: $language) {
                        ForEach(VoiceLanguage.allCases) { language in
                            Text(language.label).tag(language)
                        }
                    }
                    Button {
                        isImporting = true
                    } label: {
                        Label(L("音声ファイルを選ぶ…", "Choose an audio file…"), systemImage: "waveform.badge.plus")
                    }
                    .disabled(isWorking)
                } footer: {
                    Text(L("m4a・mp3・wav などの音声ファイルを、iPad の音声認識で文字にします。端末の中だけで認識できる言語では、音声は外に送られません。",
                           "Turns audio files (m4a, mp3, wav and others) into text with the iPad's speech recognition. For languages recognized on device, the audio never leaves the iPad."))
                }
                TranscriptionResultSection(isWorking: isWorking, fileName: fileName, result: result, errorMessage: errorMessage,
                                           onInsert: insert)
            }
            .navigationTitle(L("音声を文字にする", "Transcribe Audio"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("閉じる", "Close")) { dismiss() }
                }
            }
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.audio, .movie]) { picked in
                if case .success(let url) = picked { transcribe(url) }
            }
        }
    }

    private func insert() {
        onInsert(result)
        dismiss()
    }

    private func transcribe(_ url: URL) {
        isWorking = true
        errorMessage = nil
        result = ""
        fileName = url.lastPathComponent
        Task {
            do {
                result = try await voice.transcribe(file: url, language: language)
            } catch {
                errorMessage = error.localizedDescription
            }
            isWorking = false
        }
    }
}

private struct TranscriptionResultSection: View {
    let isWorking: Bool
    let fileName: String?
    let result: String
    let errorMessage: String?
    var onInsert: () -> Void

    var body: some View {
        if isWorking {
            Section {
                HStack {
                    ProgressView()
                    Text(L("\(fileName ?? "") を文字にしています…", "Transcribing \(fileName ?? "")…"))
                        .foregroundStyle(.secondary)
                }
            }
        } else if let errorMessage {
            Section {
                Label(errorMessage, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
        } else if !result.isEmpty {
            Section(fileName ?? "") {
                Text(result).textSelection(.enabled)
                Button(action: onInsert) {
                    Label(L("入力欄に入れる", "Insert into the message"), systemImage: "text.insert")
                }
                ShareLink(item: result) {
                    Label(L("共有・コピー", "Share / Copy"), systemImage: "square.and.arrow.up")
                }
            }
        }
    }
}
