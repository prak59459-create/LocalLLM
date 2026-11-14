import AVFoundation
import Foundation
import Observation
import Speech

enum VoiceError: Error, LocalizedError {
    case speechDenied
    case microphoneDenied
    case recognizerUnavailable(String)
    case noInput

    var errorDescription: String? {
        switch self {
        case .speechDenied:
            return L("音声認識が許可されていません。「設定」アプリで、このアプリの「音声認識」をオンにしてください。",
                     "Speech recognition is not allowed. Turn on Speech Recognition for this app in the Settings app.")
        case .microphoneDenied:
            return L("マイクが許可されていません。「設定」アプリで、このアプリの「マイク」をオンにしてください。",
                     "The microphone is not allowed. Turn on Microphone for this app in the Settings app.")
        case .recognizerUnavailable(let language):
            return L("この言語（\(language)）の音声認識は、今は使えません。", "Speech recognition for \(language) is not available right now.")
        case .noInput:
            return L("マイクから音が取れませんでした。", "No audio could be captured from the microphone.")
        }
    }
}

/// 音声入力の言語。
enum VoiceLanguage: String, CaseIterable, Identifiable, Sendable {
    case automatic, japanese, english

    var id: String { rawValue }

    var label: String {
        switch self {
        case .automatic: return L("アプリの言語に合わせる", "Follow the app language")
        case .japanese: return "日本語"
        case .english: return "English"
        }
    }

    var localeIdentifier: String {
        switch self {
        case .automatic: return L("ja-JP", "en-US")
        case .japanese: return "ja-JP"
        case .english: return "en-US"
        }
    }
}

/// 音声入力（聞き取り）と読み上げ。iOS の音声認識・音声合成を使うので、追加のモデルは要らない。
@MainActor
@Observable
final class VoiceController: NSObject, AVSpeechSynthesizerDelegate {
    // MARK: 設定（保存する）

    /// 返答を自動で読み上げる
    var autoSpeak: Bool {
        didSet { UserDefaults.standard.set(autoSpeak, forKey: "voice.autoSpeak") }
    }
    /// 声だけで会話する（話し終わったら自動で送り、返答を読み上げたら、また聞き取る）
    var handsFree: Bool {
        didSet { UserDefaults.standard.set(handsFree, forKey: "voice.handsFree") }
    }
    var language: VoiceLanguage {
        didSet { UserDefaults.standard.set(language.rawValue, forKey: "voice.language") }
    }
    /// 読み上げの速さ（1 が標準）
    var speechRate: Double {
        didSet { UserDefaults.standard.set(speechRate, forKey: "voice.rate") }
    }

    // MARK: 状態

    private(set) var isListening = false
    private(set) var isSpeaking = false
    /// 聞き取った文章（途中経過）
    private(set) var transcript = ""
    var errorMessage: String?

    @ObservationIgnored private let audioEngine = AVAudioEngine()
    @ObservationIgnored private var request: SFSpeechAudioBufferRecognitionRequest?
    @ObservationIgnored private var recognitionTask: SFSpeechRecognitionTask?
    @ObservationIgnored private var silenceTask: Task<Void, Never>?
    @ObservationIgnored private var lastHeard = Date()
    @ObservationIgnored private var onUpdate: ((String) -> Void)?
    @ObservationIgnored private var onFinish: ((String) -> Void)?
    @ObservationIgnored private let synthesizer = AVSpeechSynthesizer()
    @ObservationIgnored private var onSpeechFinished: (() -> Void)?
    /// 聞き取りの回ごとの番号（古い回の結果が、新しい回に混ざらないように）
    @ObservationIgnored private var session = 0

    override init() {
        let defaults = UserDefaults.standard
        autoSpeak = defaults.bool(forKey: "voice.autoSpeak")
        handsFree = defaults.bool(forKey: "voice.handsFree")
        language = VoiceLanguage(rawValue: defaults.string(forKey: "voice.language") ?? "") ?? .automatic
        let rate = defaults.double(forKey: "voice.rate")
        speechRate = rate > 0 ? rate : 1
        super.init()
        synthesizer.delegate = self
    }

    // MARK: - 聞き取り

    /// 聞き取りを始める。
    /// - Parameters:
    ///   - onUpdate: 途中経過（そこまでに聞き取った文章の全体）
    ///   - onFinish: 聞き取りが終わったときの文章（自分で止めたとき・黙ったとき・認識が終わったとき）
    func startListening(onUpdate: @escaping (String) -> Void, onFinish: @escaping (String) -> Void) async {
        guard !isListening else { return }
        stopSpeaking()
        errorMessage = nil
        do {
            try await authorize()
            let identifier = language.localeIdentifier
            guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: identifier)), recognizer.isAvailable else {
                throw VoiceError.recognizerUnavailable(identifier)
            }
            try begin(recognizer: recognizer, onUpdate: onUpdate, onFinish: onFinish)
        } catch {
            errorMessage = error.localizedDescription
            cleanUp()
        }
    }

    private func authorize() async throws {
        let speech = await Self.requestSpeechAuthorization()
        guard speech == .authorized else { throw VoiceError.speechDenied }
        guard await Self.requestMicrophone() else { throw VoiceError.microphoneDenied }
    }

    nonisolated private static func requestSpeechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in continuation.resume(returning: status) }
        }
    }

    nonisolated private static func requestMicrophone() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in continuation.resume(returning: granted) }
        }
    }

    private func begin(recognizer: SFSpeechRecognizer, onUpdate: @escaping (String) -> Void,
                       onFinish: @escaping (String) -> Void) throws {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .duckOthers, .allowBluetooth])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        // 端末の中だけで認識できる言語なら、音声を外に送らない
        if recognizer.supportsOnDeviceRecognition { request.requiresOnDeviceRecognition = true }
        self.request = request
        self.onUpdate = onUpdate
        self.onFinish = onFinish
        transcript = ""
        session += 1
        let current = session

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw VoiceError.noInput }
        Self.installTap(on: input, format: format, request: request)
        audioEngine.prepare()
        try audioEngine.start()

        recognitionTask = Self.recognize(recognizer, request) { [weak self] text, isFinal, failed in
            Task { @MainActor [weak self] in
                self?.handle(text: text, isFinal: isFinal, failed: failed, session: current)
            }
        }
        isListening = true
        lastHeard = Date()
        watchSilence(session: current)
    }

    /// マイクの音を認識に流す（音声のスレッドで呼ばれるので、メインアクターの外で作る）
    nonisolated private static func installTap(on node: AVAudioInputNode, format: AVAudioFormat,
                                               request: SFSpeechAudioBufferRecognitionRequest) {
        node.removeTap(onBus: 0)
        node.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }
    }

    nonisolated private static func recognize(_ recognizer: SFSpeechRecognizer, _ request: SFSpeechRecognitionRequest,
                                              handler: @escaping @Sendable (String?, Bool, Bool) -> Void) -> SFSpeechRecognitionTask {
        recognizer.recognitionTask(with: request) { result, error in
            handler(result?.bestTranscription.formattedString, result?.isFinal ?? false, error != nil)
        }
    }

    private func handle(text: String?, isFinal: Bool, failed: Bool, session current: Int) {
        guard current == session, isListening else { return }
        if let text, !text.isEmpty, text != transcript {
            transcript = text
            lastHeard = Date()
            onUpdate?(text)
        }
        if isFinal || failed { finishListening() }
    }

    /// 声だけの会話では、話し終わって少し黙ったら、そこで区切る
    private func watchSilence(session current: Int) {
        silenceTask?.cancel()
        silenceTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 300_000_000)
                guard let self, self.isListening, self.session == current else { return }
                let quiet = Date().timeIntervalSince(self.lastHeard)
                if self.handsFree, !self.transcript.isEmpty, quiet > 1.8 {
                    self.finishListening()
                    return
                }
                // 何も聞こえないまま長く経ったら止める
                if self.transcript.isEmpty, quiet > 20 {
                    self.finishListening()
                    return
                }
            }
        }
    }

    /// 聞き取りを止めて、そこまでの文章を渡す
    func stopListening() {
        guard isListening else { return }
        finishListening()
    }

    private func finishListening() {
        let text = transcript
        let finish = onFinish
        cleanUp()
        finish?(text)
    }

    private func cleanUp() {
        silenceTask?.cancel()
        silenceTask = nil
        if audioEngine.isRunning { audioEngine.stop() }
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        recognitionTask?.cancel()
        request = nil
        recognitionTask = nil
        onUpdate = nil
        onFinish = nil
        isListening = false
        session += 1
        if !isSpeaking {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    // MARK: - 読み上げ

    /// 文章を読み上げる（Markdown の記号やコードは省く）。前の読み上げは止める。
    func speak(_ markdown: String, onFinished: (() -> Void)? = nil) {
        let languageCode = SpeechText.language(of: markdown)
        let text = SpeechText.spoken(from: markdown, language: languageCode)
        stopSpeaking()
        guard !text.isEmpty else {
            onFinished?()
            return
        }
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: languageCode)
        let rate = Float(speechRate) * AVSpeechUtteranceDefaultSpeechRate
        utterance.rate = min(AVSpeechUtteranceMaximumSpeechRate, max(AVSpeechUtteranceMinimumSpeechRate, rate))
        onSpeechFinished = onFinished
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stopSpeaking() {
        guard isSpeaking else { return }
        onSpeechFinished = nil
        isSpeaking = false
        synthesizer.stopSpeaking(at: .immediate)
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.speechEnded() }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.speechEnded() }
    }

    private func speechEnded() {
        guard isSpeaking, !synthesizer.isSpeaking else { return }
        isSpeaking = false
        let finished = onSpeechFinished
        onSpeechFinished = nil
        if !isListening {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
        finished?()
    }

    // MARK: - 音声ファイル

    /// 音声ファイルを文字にする
    func transcribe(file url: URL, language: VoiceLanguage) async throws -> String {
        let speech = await Self.requestSpeechAuthorization()
        guard speech == .authorized else { throw VoiceError.speechDenied }
        let identifier = language.localeIdentifier
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: identifier)), recognizer.isAvailable else {
            throw VoiceError.recognizerUnavailable(identifier)
        }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        return try await Self.transcribe(url, recognizer: recognizer)
    }

    nonisolated private static func transcribe(_ url: URL, recognizer: SFSpeechRecognizer) async throws -> String {
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.shouldReportPartialResults = false
        request.addsPunctuation = true
        if recognizer.supportsOnDeviceRecognition { request.requiresOnDeviceRecognition = true }
        let box = ResumeOnce()
        return try await withCheckedThrowingContinuation { continuation in
            _ = recognizer.recognitionTask(with: request) { result, error in
                if let error {
                    if box.claim() { continuation.resume(throwing: error) }
                } else if let result, result.isFinal {
                    if box.claim() { continuation.resume(returning: result.bestTranscription.formattedString) }
                }
            }
        }
    }
}

/// 続きを 1 回だけ再開するための印
private final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if done { return false }
        done = true
        return true
    }
}
