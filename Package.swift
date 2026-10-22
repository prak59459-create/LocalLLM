// swift-tools-version: 5.9

// 開発用パッケージ（Mac / Linux 用）。
// アプリ本体は LocalLLM.swiftpm（Swift Playgrounds で開く）です。
// ここでは Core フォルダ（Apple 専用フレームワークに依存しない部分）だけを
// ライブラリとしてビルドし、`swift test` で単体テストを実行します。

import PackageDescription

let package = Package(
    name: "LocalLLMCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "LocalLLMCore", targets: ["LocalLLMCore"])
    ],
    targets: [
        .target(
            name: "LocalLLMCore",
            path: "LocalLLM.swiftpm/Sources/AppModule/Core"
        ),
        .testTarget(
            name: "LocalLLMCoreTests",
            dependencies: ["LocalLLMCore"],
            path: "Tests/LocalLLMCoreTests"
        )
    ]
)
