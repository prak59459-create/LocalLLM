import Foundation

/// テスト用フィクスチャ（Tests/Fixtures）の場所。
/// 環境変数 LOCALLLM_FIXTURES があればそれを使う。シンボリックリンク経由でも見つかるように実体のパスに直す。
enum TestFixtures {
    static var root: URL {
        if let path = ProcessInfo.processInfo.environment["LOCALLLM_FIXTURES"] {
            return URL(fileURLWithPath: path)
        }
        // このファイル: Tests/LocalLLMCoreTests/<フォルダ>/TestFixtures.swift
        return URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }
}
