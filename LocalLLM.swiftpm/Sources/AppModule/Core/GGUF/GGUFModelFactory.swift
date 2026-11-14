import Foundation

/// GGUF のモデルを計算するものを作る。GPU（Metal）で計算できれば GPU 用、できなければ CPU 用。
enum GGUFModelFactory {
    struct Made {
        var model: any GGUFEvaluator
        var isMetal: Bool
        /// GPU を使わなかった理由（使えたとき・使わない設定のときは nil）
        var note: String?
    }

    /// 確かめ済みのモデル（同じモデルを読み込み直すたびに確かめ直さない）
    private static var verified: Set<String> = []
    private static let lock = NSLock()

    static func make(store url: URL, context: Int, useGPU: Bool) throws -> Made {
        let store = try GGUFWeightStore(url: url)
        #if canImport(Metal)
        if useGPU {
            do {
                let model = try GGUFMetalModel(store: store, maxContext: context)
                try verifyOnce(model, store: store, key: url.path + "|\(store.header.dataBytes)")
                return Made(model: model, isMetal: true, note: nil)
            } catch {
                let reason = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                let cpu = try GGUFModel(store: store, maxContext: context)
                return Made(model: cpu, isMetal: false, note: reason)
            }
        }
        #endif
        return Made(model: try GGUFModel(store: store, maxContext: context), isMetal: false, note: nil)
    }

    #if canImport(Metal)
    private static func verifyOnce(_ model: GGUFMetalModel, store: GGUFWeightStore, key: String) throws {
        lock.lock()
        let done = verified.contains(key)
        lock.unlock()
        if done { return }
        try verify(model, against: try GGUFModel(store: store, maxContext: 16))
        lock.lock()
        verified.insert(key)
        lock.unlock()
    }
    #endif

    /// GPU の計算結果が CPU と同じになるかを、数トークンで確かめる（合わなければエラー）。
    /// 端末や OS によって GPU の計算がおかしいときに、気づかずに変な文章を出さないため。確かめたあと、どちらも空に戻す。
    static func verify(_ model: any GGUFEvaluator, against reference: any GGUFEvaluator) throws {
        let vocab = model.config.vocabSize
        let tokens = [1, 2, 3, 4].map { min($0, vocab - 1) }
        defer {
            model.truncate(to: 0)
            reference.truncate(to: 0)
        }
        // まとめて読む計算と、1 トークンずつの計算の両方を通す
        _ = try model.evaluate(Array(tokens[0..<3]))
        let actual = try model.evaluate([tokens[3]])
        _ = try reference.evaluate(Array(tokens[0..<3]))
        let expected = try reference.evaluate([tokens[3]])
        let difference = relativeDifference(actual, expected)
        guard difference < 0.02 else {
            throw GGUFError.invalid(L("GPU の計算結果が CPU と合いません（ずれ \(difference)）", "GPU results differ from the CPU (\(difference))"))
        }
    }

    /// 2 つの logits のずれ（差の二乗平均 ÷ 正解のばらつき）。数として扱えない値があれば無限大
    static func relativeDifference(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !expected.isEmpty else { return .infinity }
        var sum: Double = 0, sumOfSquares: Double = 0, error: Double = 0
        for index in expected.indices {
            let a = Double(actual[index]), e = Double(expected[index])
            guard a.isFinite, e.isFinite else { return .infinity }
            sum += e
            sumOfSquares += e * e
            error += (a - e) * (a - e)
        }
        let count = Double(expected.count)
        let mean = sum / count
        let variance = max(sumOfSquares / count - mean * mean, 1e-12)
        return Float((error / count / variance).squareRoot())
    }
}
