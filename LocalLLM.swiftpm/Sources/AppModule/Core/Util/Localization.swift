import Foundation

/// アプリの表示言語。設定画面で切り替えられる。
enum AppLanguage: String, CaseIterable, Codable, Sendable {
    case system
    case japanese
    case english

    /// UserDefaults に保存するキー。
    static let defaultsKey = "appLanguage"

    /// 現在の設定値（保存されていなければ system）。
    static var stored: AppLanguage {
        guard let raw = UserDefaults.standard.string(forKey: defaultsKey),
              let value = AppLanguage(rawValue: raw) else { return .system }
        return value
    }

    /// 実際に使う言語（system の場合は端末の優先言語から決める）。
    static var effective: AppLanguage {
        switch stored {
        case .japanese: return .japanese
        case .english: return .english
        case .system:
            let preferred = Locale.preferredLanguages.first ?? "ja"
            return preferred.hasPrefix("ja") ? .japanese : .english
        }
    }
}

/// 日本語と英語の文字列を並べて書き、現在の言語に合わせて片方を返す。
///
///     Text(L("新しいチャット", "New Chat"))
func L(_ japanese: String, _ english: String) -> String {
    AppLanguage.effective == .english ? english : japanese
}
