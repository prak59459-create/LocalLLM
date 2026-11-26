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

    /// - Parameters:
    ///   - memory: 重みのメモリへの置き方（使う分だけ置くときは CPU で計算する）
    ///   - availableBytes: 空いているメモリ（使う分だけ置くときに、残す層の数を決めるのに使う）
    static func make(store url: URL, context: Int, useGPU: Bool, memory: GGUFMemoryOptions = GGUFMemoryOptions(),
                     availableBytes: Int64? = nil) throws -> Made {
        let store = try GGUFWeightStore(url: url)
        let residency = GGUFResidency(store: store, options: memory, availableBytes: availableBytes)
        defer { residency.prepare() }
        func cpu() throws -> GGUFModel {
            let model = try GGUFModel(store: store, maxContext: context)
            model.attach(residency, options: memory)
            return model
        }
        #if canImport(Metal)
        if useGPU && memory.allowsGPU {
            do {
                let model = try GGUFMetalModel(store: store, maxContext: context)
                try verifyOnce(model, store: store, key: url.path + "|\(store.header.dataBytes)")
                return Made(model: model, isMetal: true, note: nil)
            } catch {
                let reason = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                return Made(model: try cpu(), isMetal: false, note: reason)
            }
        }
        #endif
        var note: String? = residency.summary.isEmpty ? nil : residency.summary
        if useGPU && !memory.allowsGPU {
            note = L("使う分だけメモリに置く設定なので CPU で計算", "Computing on the CPU to keep only the weights in use in memory")
                + (note.map { "・" + $0 } ?? "")
        }
        return Made(model: try cpu(), isMetal: false, note: note)
    }

    #if canImport(Metal)
    private static func verifyOnce(_ model: GGUFMetalModel, store: GGUFWeightStore, key: String) throws {
        lock.lock()
        let done = verified.contains(key)
        lock.unlock()
        if done {
            // 確かめ済みでも 1 トークンだけ計算しておく（重みを GPU から読める状態にする最初の 1 回は、メモリの使い方が
            // 大きく動く。生成の途中でそれが起きると、メモリの見張りが「急に減った」と見なしてしまう）
            _ = try model.evaluate([min(1, store.config.vocabSize - 1)])
            model.truncate(to: 0)
            return
        }
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
