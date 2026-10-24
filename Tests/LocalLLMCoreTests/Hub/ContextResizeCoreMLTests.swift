#if canImport(CoreML)
import CoreML
import Foundation
import XCTest
@testable import LocalLLMCore

/// 本物のモデルのコンテキスト長を変えて、Core ML で読み込み・推論できることを確かめる（macOS の CI 用）。
/// LOCALLLM_COREML_MODELS にモデルを置いたフォルダを指定したときだけ動く:
///   <フォルダ>/qwen.mlmodelc   … finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit
///   <フォルダ>/anemll.mlmodelc … anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4 の FFN チャンク
@available(macOS 15.0, *)
final class ContextResizeCoreMLTests: XCTestCase {
    private var modelsFolder: URL {
        get throws {
            guard let path = ProcessInfo.processInfo.environment["LOCALLLM_COREML_MODELS"] else {
                throw XCTSkip("set LOCALLLM_COREML_MODELS to run Core ML end-to-end tests")
            }
            return URL(fileURLWithPath: path)
        }
    }

    /// モデルのフォルダを、重みはハードリンクで複製する
    private func copyModel(_ source: URL) throws -> URL {
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent("e2e-\(UUID().uuidString).mlmodelc")
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(at: source, includingPropertiesForKeys: [.isDirectoryKey]) else {
            throw XCTSkip("model missing: \(source.path)")
        }
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
        for case let url as URL in enumerator {
            let relative = String(url.path.dropFirst(source.path.count + 1))
            let target = destination.appendingPathComponent(relative)
            if (try url.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
                try fileManager.createDirectory(at: target, withIntermediateDirectories: true)
            } else if url.lastPathComponent == "weight.bin" {
                try fileManager.linkItem(at: url, to: target)
            } else {
                try fileManager.copyItem(at: url, to: target)
            }
        }
        return destination
    }

    private func floatArray(_ shape: [Int], fill: (Int) -> Float) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { pointer, _ in
            for index in 0..<pointer.count { pointer[index] = Float16(fill(index)) }
        }
        return array
    }

    private func intArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { pointer, _ in
            for (index, value) in values.enumerated() { pointer[index] = value }
        }
        return array
    }

    /// [1, 1, rows, columns] の因果マスク（past の後ろに rows 個）
    private func mask(rows: Int, columns: Int, past: Int) throws -> MLMultiArray {
        try floatArray([1, 1, rows, columns]) { index in
            let row = index / columns, column = index % columns
            return column <= past + row ? 0 : -65504
        }
    }

    private func values(_ array: MLMultiArray) -> [Float] {
        var output: [Float] = []
        array.withUnsafeBufferPointer(ofType: Float16.self) { pointer in output = pointer.map { Float($0) } }
        return output
    }

    private func cpuConfiguration(function: String? = nil) -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        if let function { configuration.functionName = function }
        return configuration
    }

    /// 可変長のステートフルモデル（Qwen2.5 0.5B）: 2048 を超えた位置でも推論でき、短い入力では元と同じ結果になる
    func testFlexibleStatefulModelBeyondOriginalContext() throws {
        let source = try modelsFolder.appendingPathComponent("qwen.mlmodelc")
        let resized = try copyModel(source)
        let backup = resized.deletingLastPathComponent().appendingPathComponent("backup-\(UUID().uuidString)")
        try ContextResizer.resize(modelDirectory: resized, to: 4096, backup: backup)

        let original = try MLModel(contentsOf: source, configuration: cpuConfiguration())
        let model = try MLModel(contentsOf: resized, configuration: cpuConfiguration())
        XCTAssertEqual(model.modelDescription.stateDescriptionsByName["key_cache"]?.stateConstraint?.bufferShape,
                       [24, 1, 2, 4096, 64])

        // 短い入力: 元のモデルと同じ logits
        let tokens: [Int32] = [9707, 11, 1879, 0, 21927, 13, 2585, 525]
        func run(_ model: MLModel) throws -> [Float] {
            let state = model.makeState()
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": try intArray(tokens, shape: [1, tokens.count]),
                "causal_mask": try mask(rows: tokens.count, columns: tokens.count, past: 0),
            ])
            let output = try model.prediction(from: input, using: state)
            return values(try XCTUnwrap(output.featureValue(for: "logits")?.multiArrayValue))
        }
        let a = try run(original)
        let b = try run(model)
        XCTAssertEqual(a.count, b.count)
        let difference = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(difference, 0.05)

        // 2048 を超えるまで入れて、最後の位置の logits が有限であること
        let state = model.makeState()
        var past = 0
        let chunk = 512
        var last: [Float] = []
        while past < 2560 {
            let ids = (0..<chunk).map { Int32(100 + ($0 + past) % 1000) }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": try intArray(ids, shape: [1, chunk]),
                "causal_mask": try mask(rows: chunk, columns: past + chunk, past: past),
            ])
            let output = try model.prediction(from: input, using: state)
            last = values(try XCTUnwrap(output.featureValue(for: "logits")?.multiArrayValue))
            past += chunk
        }
        XCTAssertFalse(last.isEmpty)
        XCTAssertTrue(last.allSatisfy(\.isFinite))
    }

    /// 固定長の ANEMLL チャンク: RoPE の表を作り直して、元の長さ（512）と表の長さ（1024）を超えた位置で推論できる
    func testAnemllChunkBeyondOriginalContext() throws {
        let source = try modelsFolder.appendingPathComponent("anemll.mlmodelc")
        let resized = try copyModel(source)
        let backup = resized.deletingLastPathComponent().appendingPathComponent("backup-\(UUID().uuidString)")
        try ContextResizer.resize(modelDirectory: resized, to: 2048, backup: backup)

        let hidden = 1024
        let batch = 64
        func prefill(_ url: URL, context: Int, position: Int) throws -> [Float] {
            let model = try MLModel(contentsOf: url, configuration: cpuConfiguration(function: "prefill"))
            let state = model.makeState()
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": try floatArray([1, batch, hidden]) { Float(($0 * 7919) % 200 - 100) / 400 },
                "position_ids": try intArray((0..<batch).map { Int32(position + $0) }, shape: [batch]),
                "causal_mask": try mask(rows: batch, columns: context, past: position),
                "current_pos": try intArray([Int32(position)], shape: [1]),
            ])
            let output = try model.prediction(from: input, using: state)
            return values(try XCTUnwrap(output.featureValue(for: "output_hidden_states")?.multiArrayValue))
        }
        // 位置 0 では元のモデルと同じ
        let a = try prefill(source, context: 512, position: 0)
        let b = try prefill(resized, context: 2048, position: 0)
        let difference = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(difference, 0.05)
        // 元の表の長さ（1024）を超えた位置
        let far = try prefill(resized, context: 2048, position: 1536)
        XCTAssertTrue(far.allSatisfy(\.isFinite))

        // infer 関数も読み込める
        let infer = try MLModel(contentsOf: resized, configuration: cpuConfiguration(function: "infer"))
        let state = infer.makeState()
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": try floatArray([1, 1, hidden]) { Float($0 % 13) / 50 },
            "position_ids": try intArray([2000], shape: [1]),
            "causal_mask": try mask(rows: 1, columns: 2048, past: 2000),
            "current_pos": try intArray([2000], shape: [1]),
        ])
        let output = try infer.prediction(from: input, using: state)
        XCTAssertTrue(values(try XCTUnwrap(output.featureValue(for: "output_hidden_states")?.multiArrayValue)).allSatisfy(\.isFinite))
    }
}
#endif
