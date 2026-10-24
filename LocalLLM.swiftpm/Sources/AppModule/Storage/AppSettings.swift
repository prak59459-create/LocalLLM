import CoreML
import Foundation
import Observation
import Security

/// 計算に使うチップの選び方（MLComputeUnits）。
enum ComputeUnitsOption: String, CaseIterable, Codable, Identifiable, Sendable {
    case automatic
    case cpuAndGPU
    case all
    case cpuAndNeuralEngine
    case cpuOnly

    var id: String { rawValue }

    var computeUnits: MLComputeUnits {
        switch self {
        case .automatic: return .cpuAndGPU
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuOnly: return .cpuOnly
        }
    }

    var label: String {
        switch self {
        case .automatic: return L("自動（推奨）", "Automatic (recommended)")
        case .all: return L("すべて（Neural Engine を含む）", "All (incl. Neural Engine)")
        case .cpuAndGPU: return L("CPU + GPU", "CPU + GPU")
        case .cpuAndNeuralEngine: return L("CPU + Neural Engine", "CPU + Neural Engine")
        case .cpuOnly: return L("CPU のみ", "CPU only")
        }
    }
}

/// 思考モード（Qwen3 などの <think> を使うモデル向け）。
enum ThinkingMode: String, CaseIterable, Codable, Identifiable, Sendable {
    case modelDefault
    case on
    case off

    var id: String { rawValue }

    /// テンプレート変数 enable_thinking に渡す値（nil なら渡さない）
    var enableThinking: Bool? {
        switch self {
        case .modelDefault: return nil
        case .on: return true
        case .off: return false
        }
    }

    var label: String {
        switch self {
        case .modelDefault: return L("モデルの既定", "Model default")
        case .on: return L("オン", "On")
        case .off: return L("オフ", "Off")
        }
    }
}

/// アプリの設定（UserDefaults に保存）。HF トークンだけはキーチェーンに保存する。
@MainActor
@Observable
final class AppSettings {
    @ObservationIgnored private let defaults = UserDefaults.standard

    var generation: GenerationParameters {
        didSet { saveCodable(generation, forKey: Keys.generation) }
    }
    var systemPrompt: String {
        didSet { defaults.set(systemPrompt, forKey: Keys.systemPrompt) }
    }
    var thinkingMode: ThinkingMode {
        didSet { defaults.set(thinkingMode.rawValue, forKey: Keys.thinkingMode) }
    }
    var computeUnits: ComputeUnitsOption {
        didSet { defaults.set(computeUnits.rawValue, forKey: Keys.computeUnits) }
    }
    var prefillChunkSize: Int {
        didSet { defaults.set(prefillChunkSize, forKey: Keys.prefillChunkSize) }
    }
    var showStats: Bool {
        didSet { defaults.set(showStats, forKey: Keys.showStats) }
    }
    var language: AppLanguage {
        didSet { defaults.set(language.rawValue, forKey: AppLanguage.defaultsKey) }
    }
    /// 最後に使ったモデル（InstalledModel.id）
    var activeModelID: String? {
        didSet { defaults.set(activeModelID, forKey: Keys.activeModelID) }
    }
    /// 起動時に前回のモデルを自動で読み込む
    var autoLoadLastModel: Bool {
        didSet { defaults.set(autoLoadLastModel, forKey: Keys.autoLoadLastModel) }
    }
    /// モデルにツール（Web 検索や MCP）を使わせる
    var toolsEnabled: Bool {
        didSet { defaults.set(toolsEnabled, forKey: Keys.toolsEnabled) }
    }
    /// 1 回に使うコンテキストの上限（0 ならモデルの最大まで）。メモリと速度を節約したいときに小さくする。
    var contextLimit: Int {
        didSet { defaults.set(contextLimit, forKey: Keys.contextLimit) }
    }
    /// Hugging Face のアクセストークン（ゲート付きモデル用。任意）
    var huggingFaceToken: String {
        didSet { Keychain.set(huggingFaceToken, for: Keys.huggingFaceToken) }
    }

    static let defaultSystemPrompt = "あなたは親切で正確なアシスタントです。質問された言語で、簡潔に分かりやすく答えてください。"

    init() {
        let defaults = UserDefaults.standard
        generation = Self.loadCodable(GenerationParameters.self, forKey: Keys.generation, from: defaults) ?? .default
        systemPrompt = defaults.string(forKey: Keys.systemPrompt) ?? Self.defaultSystemPrompt
        thinkingMode = defaults.string(forKey: Keys.thinkingMode).flatMap(ThinkingMode.init(rawValue:)) ?? .modelDefault
        // 既定は CPU + GPU。可変長入力の LLM は Neural Engine で動かないことが多く、
        // .all だと初回読み込み時の Neural Engine 向け変換で時間がかかったり失敗したりするため。
        computeUnits = defaults.string(forKey: Keys.computeUnits).flatMap(ComputeUnitsOption.init(rawValue:)) ?? .automatic
        let chunk = defaults.integer(forKey: Keys.prefillChunkSize)
        prefillChunkSize = chunk > 0 ? chunk : 128
        showStats = defaults.object(forKey: Keys.showStats) as? Bool ?? true
        language = AppLanguage.stored
        activeModelID = defaults.string(forKey: Keys.activeModelID)
        autoLoadLastModel = defaults.object(forKey: Keys.autoLoadLastModel) as? Bool ?? true
        huggingFaceToken = Keychain.get(Keys.huggingFaceToken) ?? ""
        toolsEnabled = defaults.object(forKey: Keys.toolsEnabled) as? Bool ?? true
        contextLimit = max(0, defaults.integer(forKey: Keys.contextLimit))
    }

    /// 生成パラメータを既定値に戻す。
    func resetGeneration() {
        generation = .default
    }

    /// HF API クライアント（トークンがあれば付ける）
    func makeHFClient() -> HFClient {
        let token = huggingFaceToken.trimmingCharacters(in: .whitespacesAndNewlines)
        return HFClient(token: token.isEmpty ? nil : token)
    }

    private enum Keys {
        static let generation = "generationParameters"
        static let systemPrompt = "systemPrompt"
        static let thinkingMode = "thinkingMode"
        static let computeUnits = "computeUnits"
        static let prefillChunkSize = "prefillChunkSize"
        static let showStats = "showStats"
        static let activeModelID = "activeModelID"
        static let autoLoadLastModel = "autoLoadLastModel"
        static let huggingFaceToken = "huggingFaceToken"
        static let toolsEnabled = "toolsEnabled"
        static let contextLimit = "contextLimit"
    }

    private func saveCodable<T: Encodable>(_ value: T, forKey key: String) {
        if let data = try? JSONEncoder().encode(value) { defaults.set(data, forKey: key) }
    }

    private static func loadCodable<T: Decodable>(_ type: T.Type, forKey key: String, from defaults: UserDefaults) -> T? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}

/// キーチェーンへの簡単な読み書き（文字列のみ）。
enum Keychain {
    private static let service = "LocalLLM"

    static func set(_ value: String, for account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        guard !value.isEmpty, let data = value.data(using: .utf8) else { return }
        var attributes = query
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(attributes as CFDictionary, nil)
    }

    static func get(_ account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
