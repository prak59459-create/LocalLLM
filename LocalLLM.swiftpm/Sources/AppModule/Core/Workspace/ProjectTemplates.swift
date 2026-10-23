import Foundation

/// 新しいプロジェクトのひな形。
struct ProjectTemplate: Identifiable, Sendable {
    var id: String
    var name: String
    var summary: String
    var symbolName: String
    /// フォルダ名の末尾に付ける拡張子（Swift Playgrounds のアプリは ".swiftpm"）
    var folderSuffix: String = ""
    /// 最初に開くファイル
    var mainFile: String?
    /// パス → 中身。{{NAME}} はプロジェクト名、{{IDENT}} は識別子に使える名前に置き換える。
    var files: [String: String]
    /// ロードマップの最初の丸
    var roadmap: [(title: String, detail: String, kind: RoadmapNode.Kind, status: RoadmapNode.Status)]

    func render(_ text: String, name: String) -> String {
        text.replacingOccurrences(of: "{{NAME}}", with: name)
            .replacingOccurrences(of: "{{IDENT}}", with: ProjectTemplates.identifier(from: name))
    }
}

enum ProjectTemplates {
    static var all: [ProjectTemplate] { [web, swiftApp, javascript, notes, empty] }

    static func template(id: String) -> ProjectTemplate? { all.first { $0.id == id } }

    /// フォルダ名に使える名前（使えない文字は _ に）
    static func folderName(for name: String, template: ProjectTemplate) -> String {
        let forbidden = CharacterSet(charactersIn: "/\\:?%*|\"<>")
        var cleaned = String(name.unicodeScalars.map { forbidden.contains($0) ? "_" : Character($0) })
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty { cleaned = "Project" }
        if !template.folderSuffix.isEmpty && !cleaned.hasSuffix(template.folderSuffix) { cleaned += template.folderSuffix }
        return cleaned
    }

    /// Swift の型名などに使える名前
    static func identifier(from name: String) -> String {
        let base = name.replacingOccurrences(of: ".swiftpm", with: "")
        var result = ""
        var upperNext = true
        for scalar in base.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) && scalar.isASCII {
                let character = String(scalar)
                result += upperNext ? character.uppercased() : character
                upperNext = false
            } else {
                upperNext = true
            }
        }
        if result.isEmpty { return "MyApp" }
        if result.first?.isNumber == true { result = "My" + result }
        return result
    }

    /// ひな形のファイルとロードマップを作る。
    static func create(_ template: ProjectTemplate, name: String, in fileSystem: WorkspaceFileSystem) throws -> Roadmap {
        for (path, content) in template.files {
            try fileSystem.write(template.render(path, name: name), template.render(content, name: name))
        }
        var roadmap = Roadmap()
        let goal = roadmap.add(title: name, detail: template.summary, kind: .goal, status: .inProgress, at: (0, 0))
        for item in template.roadmap {
            roadmap.add(title: item.title, detail: item.detail, kind: item.kind, status: item.status, parent: goal.id)
        }
        return roadmap
    }

    // MARK: - ひな形

    static var web: ProjectTemplate { ProjectTemplate(
        id: "web",
        name: L("Web アプリ（HTML / CSS / JavaScript）", "Web app (HTML / CSS / JavaScript)"),
        summary: L("ブラウザで動くアプリ。プレビューですぐに動かせます。", "Runs in the browser. Try it right away in the preview."),
        symbolName: "globe",
        mainFile: "index.html",
        files: [
            "index.html": """
            <!doctype html>
            <html lang="ja">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>{{NAME}}</title>
              <link rel="stylesheet" href="style.css">
            </head>
            <body>
              <main>
                <h1>{{NAME}}</h1>
                <p id="message">Hello!</p>
                <button id="button">Click me</button>
              </main>
              <script src="app.js"></script>
            </body>
            </html>

            """,
            "style.css": """
            :root { color-scheme: light dark; font-family: -apple-system, system-ui, sans-serif; }
            body { margin: 0; display: grid; place-items: center; min-height: 100vh; }
            main { text-align: center; padding: 24px; }
            button { font-size: 1rem; padding: 10px 20px; border-radius: 10px; border: none; background: #5b5bd6; color: white; }

            """,
            "app.js": """
            const button = document.getElementById("button");
            const message = document.getElementById("message");
            let count = 0;

            button.addEventListener("click", () => {
              count += 1;
              message.textContent = `Clicked ${count} times`;
              console.log("clicked", count);
            });

            """,
            "README.md": "# {{NAME}}\n\n",
        ],
        roadmap: [
            (L("画面のレイアウト", "Page layout"), "index.html / style.css", .feature, .done),
            (L("ボタンの動き", "Button behavior"), "app.js", .feature, .done),
            (L("次に作る機能", "Next feature"), "", .idea, .idea),
        ]
    ) }

    static var swiftApp: ProjectTemplate { ProjectTemplate(
        id: "swiftpm",
        name: L("iPad アプリ（Swift Playgrounds）", "iPad app (Swift Playgrounds)"),
        summary: L("SwiftUI のアプリ。フォルダを Swift Playgrounds で開くとビルドして動かせます。",
                   "A SwiftUI app. Open the folder in Swift Playgrounds to build and run it."),
        symbolName: "swift",
        folderSuffix: ".swiftpm",
        mainFile: "Sources/AppModule/ContentView.swift",
        files: [
            "Package.swift": """
            // swift-tools-version: 5.9

            import PackageDescription
            import AppleProductTypes

            let package = Package(
                name: "{{IDENT}}",
                platforms: [
                    .iOS("17.0")
                ],
                products: [
                    .iOSApplication(
                        name: "{{IDENT}}",
                        targets: ["AppModule"],
                        bundleIdentifier: "com.example.{{IDENT}}",
                        teamIdentifier: "",
                        displayVersion: "1.0",
                        bundleVersion: "1",
                        appIcon: .placeholder(icon: .sparkle),
                        accentColor: .presetColor(.blue),
                        supportedDeviceFamilies: [
                            .pad,
                            .phone
                        ],
                        supportedInterfaceOrientations: [
                            .portrait,
                            .landscapeRight,
                            .landscapeLeft,
                            .portraitUpsideDown(.when(deviceFamilies: [.pad]))
                        ]
                    )
                ],
                targets: [
                    .executableTarget(
                        name: "AppModule",
                        path: "Sources/AppModule"
                    )
                ]
            )

            """,
            "Sources/AppModule/App.swift": """
            import SwiftUI

            @main
            struct {{IDENT}}App: App {
                var body: some Scene {
                    WindowGroup {
                        ContentView()
                    }
                }
            }

            """,
            "Sources/AppModule/ContentView.swift": """
            import SwiftUI

            struct ContentView: View {
                @State private var count = 0

                var body: some View {
                    VStack(spacing: 16) {
                        Text("{{NAME}}")
                            .font(.largeTitle.bold())
                        Text("\\(count)")
                            .font(.system(size: 48, weight: .semibold, design: .rounded))
                        Button("+1") { count += 1 }
                            .buttonStyle(.borderedProminent)
                    }
                    .padding()
                }
            }

            """,
        ],
        roadmap: [
            (L("最初の画面", "First screen"), "ContentView.swift", .feature, .done),
            (L("データの保存", "Saving data"), "", .idea, .idea),
            (L("設定画面", "Settings screen"), "", .idea, .idea),
        ]
    ) }

    static var javascript: ProjectTemplate { ProjectTemplate(
        id: "js",
        name: L("JavaScript のプログラム", "JavaScript program"),
        summary: L("main.js を iPad の中で実行して、結果をコンソールで見られます。", "Run main.js on the iPad and see the output in the console."),
        symbolName: "curlybraces",
        mainFile: "main.js",
        files: [
            "main.js": """
            // ▶︎ ボタン（または run_javascript ツール）で実行できます
            function fizzbuzz(n) {
              const out = [];
              for (let i = 1; i <= n; i++) {
                out.push(i % 15 === 0 ? "FizzBuzz" : i % 3 === 0 ? "Fizz" : i % 5 === 0 ? "Buzz" : String(i));
              }
              return out;
            }

            console.log(fizzbuzz(15).join(", "));

            """,
            "README.md": "# {{NAME}}\n\n",
        ],
        roadmap: [
            (L("最初のプログラム", "First program"), "main.js", .feature, .done),
        ]
    ) }

    static var notes: ProjectTemplate { ProjectTemplate(
        id: "notes",
        name: L("ドキュメント・メモ（Markdown）", "Docs & notes (Markdown)"),
        summary: L("文章や設計メモをまとめるプロジェクト。", "A project for writing and design notes."),
        symbolName: "doc.richtext",
        mainFile: "README.md",
        files: [
            "README.md": "# {{NAME}}\n\n## 目的\n\n## メモ\n\n",
        ],
        roadmap: []
    ) }

    static var empty: ProjectTemplate { ProjectTemplate(
        id: "empty",
        name: L("空のプロジェクト", "Empty project"),
        summary: L("何も入っていないフォルダから始めます。", "Start from an empty folder."),
        symbolName: "folder",
        mainFile: nil,
        files: [:],
        roadmap: []
    ) }
}
