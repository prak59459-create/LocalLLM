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
            displayVersion: "0.6",
            bundleVersion: "6",
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
            capabilities: [
                .microphone(purposeString: "音声で入力するために、マイクを使います。 / Used for voice input."),
                .speechRecognition(purposeString: "話した内容を文字にするために、音声認識を使います。 / Used to turn your speech into text."),
                .photoLibraryAdd(purposeString: "作った画像を「写真」に保存するために使います。 / Used to save generated images to Photos.")
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
