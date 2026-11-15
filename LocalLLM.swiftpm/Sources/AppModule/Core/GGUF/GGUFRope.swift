import Foundation

/// RoPE（位置に応じた回転）の周波数。CPU 用と GPU 用のどちらの計算でも使う（llama.cpp の ggml_rope_ext と同じ式）。
enum GGUFRope {
    struct Table {
        /// 組ごとの周波数（位置 p の i 番目の組の角度 = p × frequencies[i]）
        var frequencies: [Float]
        /// cos / sin に掛ける倍率（YaRN）
        var magnitude: Float

        var pairs: Int { frequencies.count }
    }

    /// 全体を見る層用（sliding = false）か、直近だけを見る層用（true）の表
    static func table(config: GGUFModelConfig, sliding: Bool, factors: UnsafePointer<Float>?) -> Table {
        let dimensions = sliding ? config.slidingRopeDimensions : config.ropeDimensions
        let base = Double(sliding ? config.slidingRopeBase : config.ropeBase)
        let scale = Double(sliding ? config.slidingRopeScale : config.ropeScale)
        let usesFactors = !sliding || config.ropeFactorsOnSlidingLayers
        let pairs = dimensions / 2
        guard pairs > 0 else { return Table(frequencies: [], magnitude: 1) }
        let yarn = scale != 1 ? config.yarn : nil
        var low = 0.0, high = 0.0
        if let yarn {
            (low, high) = correctionRange(dimensions: dimensions, originalContext: yarn.originalContext, base: base,
                                          betaFast: Double(yarn.betaFast), betaSlow: Double(yarn.betaSlow))
        }
        var frequencies: [Float] = []
        frequencies.reserveCapacity(pairs)
        for index in 0..<pairs {
            let factor = usesFactors ? Double(factors?[index] ?? 1) : 1
            let extrapolated = Foundation.pow(base, -2 * Double(index) / Double(dimensions)) / factor
            var frequency = extrapolated * scale
            if yarn != nil {
                // 低い周波数は伸ばし、高い周波数はそのまま（その間はなめらかにつなぐ）
                let y = (Double(index) - low) / max(0.001, high - low)
                let ramp = 1 - min(1, max(0, y))
                frequency = frequency * (1 - ramp) + extrapolated * ramp
            }
            frequencies.append(Float(frequency))
        }
        return Table(frequencies: frequencies, magnitude: yarn?.magnitude ?? 1)
    }

    /// YaRN で、伸ばす周波数と伸ばさない周波数の境目（ggml_rope_yarn_corr_dims）
    static func correctionRange(dimensions: Int, originalContext: Int, base: Double, betaFast: Double, betaSlow: Double)
        -> (Double, Double) {
        func dimension(_ rotations: Double) -> Double {
            Double(dimensions) * Foundation.log(Double(originalContext) / (rotations * 2 * Double.pi)) / (2 * Foundation.log(base))
        }
        let start = Foundation.floor(dimension(betaFast))
        let end = Foundation.ceil(dimension(betaSlow))
        return (max(0, start), min(Double(dimensions - 1), end))
    }
}
