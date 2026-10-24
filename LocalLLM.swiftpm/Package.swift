// swift-tools-version: 5.9

// Swift Playgrounds (iPad) で開く App プロジェクトの定義ファイルです。
// Playgrounds のアプリ設定画面で項目を変更すると、このファイルは自動で書き換えられます。

import PackageDescription
import AppleProductTypes

let package = Package(
    name: "LocalLLM",
    platforms: [
        .iOS("18.0")
    ],
    products: [
        .iOSApplication(
            name: "LocalLLM",
            targets: ["AppModule"],
            bundleIdentifier: "com.example.LocalLLM",
            teamIdentifier: "",
            displayVersion: "0.5",
            bundleVersion: "5",
            appIcon: .placeholder(icon: .chatMessage),
            accentColor: .presetColor(.indigo),
            supportedDeviceFamilies: [
                .pad,
                .phone
            ],
            supportedInterfaceOrientations: [
                .portrait,
                .landscapeRight,
                .landscapeLeft,
                .portraitUpsideDown(.when(deviceFamilies: [.pad]))
            ],
            appCategory: .productivity
        )
    ],
    targets: [
        .executableTarget(
            name: "AppModule",
            path: "Sources/AppModule"
        )
    ]
)
