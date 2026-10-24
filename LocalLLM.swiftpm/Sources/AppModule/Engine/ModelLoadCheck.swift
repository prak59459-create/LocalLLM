import CoreML
import Foundation

/// 書き換えたモデルが Core ML で読み込めるかを確かめる（メインスレッドの外で呼ぶ）。
enum ModelLoadCheck {
    /// CPU で読み込んでみる（multifunction なら関数ごとに）。読み込みで MIL の型や形の食い違いが分かる。
    static func verify(_ url: URL) throws {
        let functions = (try? ContextResizer.analyze(modelDirectory: url).functions) ?? []
        if functions.count <= 1 {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuOnly
            _ = try MLModel(contentsOf: url, configuration: configuration)
            return
        }
        for function in functions {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuOnly
            configuration.functionName = function
            _ = try MLModel(contentsOf: url, configuration: configuration)
        }
    }
}
