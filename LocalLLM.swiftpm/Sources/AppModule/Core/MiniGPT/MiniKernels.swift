import Foundation

/// MiniGPT の各層の順伝播・逆伝播（行優先の Float 配列）。
/// 逆伝播の dWeight / dBias は「加算」、dInput は名前に Accumulate がない限り「上書き」。
enum MiniKernels {
    static let layerNormEpsilon: Float = 1e-5
    private static let geluScale = Float((2.0 / Double.pi).squareRoot())
    private static let geluCubic: Float = 0.044715

    // MARK: - 線形層 y = x W + b（W は [入力, 出力]）

    static func linear(input: [Float], weight: [Float], bias: [Float], rows: Int, inputSize: Int, outputSize: Int,
                       output: inout [Float]) {
        input.withUnsafeBufferPointer { x in
            weight.withUnsafeBufferPointer { w in
                bias.withUnsafeBufferPointer { b in
                    output.withUnsafeMutableBufferPointer { y in
                        MiniMath.gemm(transA: false, transB: false, m: rows, n: outputSize, k: inputSize, alpha: 1,
                                      a: x.baseAddress!, lda: inputSize, b: w.baseAddress!, ldb: outputSize,
                                      beta: 0, c: y.baseAddress!, ldc: outputSize)
                        MiniMath.addBias(b.baseAddress!, to: y.baseAddress!, rows: rows, columns: outputSize)
                    }
                }
            }
        }
    }

    /// dInput = dOutput Wᵀ（上書き）、dWeight += xᵀ dOutput、dBias += 列の和
    static func linearBackward(dOutput: [Float], input: [Float], weight: [Float], rows: Int, inputSize: Int, outputSize: Int,
                               dInput: inout [Float], dWeight: inout [Float], dBias: inout [Float]) {
        dOutput.withUnsafeBufferPointer { dy in
            input.withUnsafeBufferPointer { x in
                weight.withUnsafeBufferPointer { w in
                    dInput.withUnsafeMutableBufferPointer { dx in
                        MiniMath.gemm(transA: false, transB: true, m: rows, n: inputSize, k: outputSize, alpha: 1,
                                      a: dy.baseAddress!, lda: outputSize, b: w.baseAddress!, ldb: outputSize,
                                      beta: 0, c: dx.baseAddress!, ldc: inputSize)
                    }
                }
                dWeight.withUnsafeMutableBufferPointer { dw in
                    MiniMath.gemm(transA: true, transB: false, m: inputSize, n: outputSize, k: rows, alpha: 1,
                                  a: x.baseAddress!, lda: inputSize, b: dy.baseAddress!, ldb: outputSize,
                                  beta: 1, c: dw.baseAddress!, ldc: outputSize)
                }
            }
            dBias.withUnsafeMutableBufferPointer { db in
                MiniMath.addColumnSums(dy.baseAddress!, rows: rows, columns: outputSize, into: db.baseAddress!)
            }
        }
    }

    // MARK: - LayerNorm

    static func layerNormForward(input: [Float], weight: [Float], bias: [Float], rows: Int, columns: Int,
                                 output: inout [Float], mean: inout [Float], rstd: inout [Float]) {
        input.withUnsafeBufferPointer { x in
            weight.withUnsafeBufferPointer { w in
                bias.withUnsafeBufferPointer { b in
                    output.withUnsafeMutableBufferPointer { y in
                        for r in 0..<rows {
                            let xr = x.baseAddress! + r * columns
                            let yr = y.baseAddress! + r * columns
                            let m = MiniVector.mean(xr, columns)
                            MiniVector.shift(xr, -m, yr, columns)
                            let variance = MiniMath.sumOfSquares(yr, count: columns) / Float(columns)
                            let s = 1 / (variance + layerNormEpsilon).squareRoot()
                            MiniVector.scale(yr, s, yr, columns)
                            MiniVector.multiplyAdd(yr, w.baseAddress!, b.baseAddress!, yr, columns)
                            mean[r] = m
                            rstd[r] = s
                        }
                    }
                }
            }
        }
    }

    /// dInput に「加算」する（残差接続の勾配と合わせるため）
    static func layerNormBackward(dOutput: [Float], input: [Float], weight: [Float], mean: [Float], rstd: [Float],
                                  rows: Int, columns: Int,
                                  dInput: inout [Float], dWeight: inout [Float], dBias: inout [Float]) {
        var xhat = [Float](repeating: 0, count: columns)
        var dxhat = [Float](repeating: 0, count: columns)
        dOutput.withUnsafeBufferPointer { dy in
            input.withUnsafeBufferPointer { x in
                weight.withUnsafeBufferPointer { w in
                    xhat.withUnsafeMutableBufferPointer { xh in
                        dxhat.withUnsafeMutableBufferPointer { dxh in
                            dInput.withUnsafeMutableBufferPointer { dx in
                                dWeight.withUnsafeMutableBufferPointer { dw in
                                    dBias.withUnsafeMutableBufferPointer { db in
                                        let xh = xh.baseAddress!, dxh = dxh.baseAddress!
                                        for r in 0..<rows {
                                            let s = rstd[r]
                                            let dyr = dy.baseAddress! + r * columns
                                            let dxr = dx.baseAddress! + r * columns
                                            MiniVector.shift(x.baseAddress! + r * columns, -mean[r], xh, columns)
                                            MiniVector.scale(xh, s, xh, columns)
                                            MiniVector.add(dyr, db.baseAddress!, db.baseAddress!, columns)
                                            MiniVector.multiplyAdd(dyr, xh, dw.baseAddress!, dw.baseAddress!, columns)
                                            MiniVector.multiply(dyr, w.baseAddress!, dxh, columns)
                                            let meanD = MiniVector.mean(dxh, columns)
                                            let meanDX = MiniMath.dot(dxh, xh, count: columns) / Float(columns)
                                            // dx += s * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
                                            MiniVector.scaleAdd(xh, -meanDX, dxh, dxh, columns)
                                            MiniVector.shift(dxh, -meanD, dxh, columns)
                                            MiniVector.scaleAdd(dxh, s, dxr, dxr, columns)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - GELU（tanh 近似）

    static func geluForward(_ input: [Float], output: inout [Float]) {
        let n = input.count
        input.withUnsafeBufferPointer { x in
            output.withUnsafeMutableBufferPointer { y in
                let x = x.baseAddress!, y = y.baseAddress!
                // y = 0.5 x (1 + tanh(k (x + c x³)))
                MiniVector.multiply(x, x, y, n)
                MiniVector.scaleShift(y, geluCubic, 1, y, n)
                MiniVector.multiply(y, x, y, n)
                MiniVector.scale(y, geluScale, y, n)
                MiniVector.tanh(y, y, n)
                MiniVector.scaleShift(y, 0.5, 0.5, y, n)
                MiniVector.multiply(y, x, y, n)
            }
        }
    }

    /// dOutput（GELU の出力の勾配）を、その場で入力の勾配にする
    static func geluBackward(input: [Float], dOutput: inout [Float]) {
        let n = input.count
        var a = [Float](repeating: 0, count: n)
        var b = [Float](repeating: 0, count: n)
        var c = [Float](repeating: 0, count: n)
        input.withUnsafeBufferPointer { x in
            a.withUnsafeMutableBufferPointer { a in
                b.withUnsafeMutableBufferPointer { b in
                    c.withUnsafeMutableBufferPointer { c in
                        dOutput.withUnsafeMutableBufferPointer { dy in
                            let x = x.baseAddress!, a = a.baseAddress!, b = b.baseAddress!, c = c.baseAddress!
                            // th = tanh(k (x + c x³))
                            MiniVector.multiply(x, x, a, n)
                            MiniVector.scaleShift(a, 3 * geluCubic, 1, b, n)     // b = 1 + 3c x²
                            MiniVector.scaleShift(a, geluCubic, 1, a, n)
                            MiniVector.multiply(a, x, a, n)
                            MiniVector.scale(a, geluScale, a, n)
                            MiniVector.tanh(a, a, n)                              // a = th
                            // 0.5 x (1 - th²) k (1 + 3c x²)
                            MiniVector.multiply(b, x, b, n)
                            MiniVector.scale(b, 0.5 * geluScale, b, n)
                            MiniVector.multiply(a, a, c, n)
                            MiniVector.scaleShift(c, -1, 1, c, n)
                            MiniVector.multiply(b, c, b, n)
                            // + 0.5 (1 + th)
                            MiniVector.scaleShift(a, 0.5, 0.5, a, n)
                            MiniVector.add(a, b, a, n)
                            MiniVector.multiply(dy.baseAddress!, a, dy.baseAddress!, n)
                        }
                    }
                }
            }
        }
    }

    // MARK: - 因果注意機構

    /// qkv [N, 3C] → output [N, C]。probabilities [B*H, S, S] に softmax の結果をとっておく。
    static func attentionForward(qkv: [Float], batch: Int, length: Int, channels: Int, heads: Int,
                                 probabilities: inout [Float], output: inout [Float]) {
        let s = length, c = channels, hs = channels / heads
        let scale = 1 / Float(hs).squareRoot()
        qkv.withUnsafeBufferPointer { qkv in
            probabilities.withUnsafeMutableBufferPointer { prob in
                output.withUnsafeMutableBufferPointer { out in
                    for b in 0..<batch {
                        for h in 0..<heads {
                            let q = qkv.baseAddress! + b * s * 3 * c + h * hs
                            let k = q + c
                            let v = q + 2 * c
                            let p = prob.baseAddress! + (b * heads + h) * s * s
                            // P = softmax(mask(q kᵀ / sqrt(hs)))
                            MiniMath.gemm(transA: false, transB: true, m: s, n: s, k: hs, alpha: scale,
                                          a: q, lda: 3 * c, b: k, ldb: 3 * c, beta: 0, c: p, ldc: s)
                            for t in 0..<s {
                                let row = p + t * s
                                let valid = t + 1
                                let maxValue = MiniVector.maximum(row, valid)
                                MiniVector.shift(row, -maxValue, row, valid)
                                MiniVector.exp(row, row, valid)
                                let total = MiniVector.sum(row, valid)
                                MiniVector.scale(row, 1 / total, row, valid)
                                MiniVector.fillZero(row + valid, s - valid)
                            }
                            // out = P v
                            MiniMath.gemm(transA: false, transB: false, m: s, n: hs, k: s, alpha: 1,
                                          a: p, lda: s, b: v, ldb: 3 * c, beta: 0,
                                          c: out.baseAddress! + b * s * c + h * hs, ldc: c)
                        }
                    }
                }
            }
        }
    }

    /// dOutput [N, C] → dQKV [N, 3C]（上書き）
    static func attentionBackward(dOutput: [Float], qkv: [Float], probabilities: [Float],
                                  batch: Int, length: Int, channels: Int, heads: Int, dQKV: inout [Float]) {
        let s = length, c = channels, hs = channels / heads
        let scale = 1 / Float(hs).squareRoot()
        var dScores = [Float](repeating: 0, count: s * s)
        dOutput.withUnsafeBufferPointer { dout in
            qkv.withUnsafeBufferPointer { qkv in
                probabilities.withUnsafeBufferPointer { prob in
                    dScores.withUnsafeMutableBufferPointer { ds in
                        dQKV.withUnsafeMutableBufferPointer { dqkv in
                            let ds = ds.baseAddress!
                            for b in 0..<batch {
                                for h in 0..<heads {
                                    let offset = b * s * 3 * c + h * hs
                                    let q = qkv.baseAddress! + offset
                                    let k = q + c
                                    let v = q + 2 * c
                                    let dq = dqkv.baseAddress! + offset
                                    let dk = dq + c
                                    let dv = dq + 2 * c
                                    let p = prob.baseAddress! + (b * heads + h) * s * s
                                    let dO = dout.baseAddress! + b * s * c + h * hs
                                    // dP = dO vᵀ, dv = Pᵀ dO
                                    MiniMath.gemm(transA: false, transB: true, m: s, n: s, k: hs, alpha: 1,
                                                  a: dO, lda: c, b: v, ldb: 3 * c, beta: 0, c: ds, ldc: s)
                                    MiniMath.gemm(transA: true, transB: false, m: s, n: hs, k: s, alpha: 1,
                                                  a: p, lda: s, b: dO, ldb: c, beta: 0, c: dv, ldc: 3 * c)
                                    // dS = P ⊙ (dP − rowsum(P ⊙ dP)) / sqrt(hs)（マスクの外は P = 0 なので 0）
                                    for t in 0..<s {
                                        let row = ds + t * s
                                        let pRow = p + t * s
                                        let valid = t + 1
                                        let rowDot = MiniMath.dot(pRow, row, count: valid)
                                        MiniVector.shift(row, -rowDot, row, valid)
                                        MiniVector.multiply(row, pRow, row, valid)
                                        MiniVector.scale(row, scale, row, valid)
                                        MiniVector.fillZero(row + valid, s - valid)
                                    }
                                    // dq = dS k, dk = dSᵀ q
                                    MiniMath.gemm(transA: false, transB: false, m: s, n: hs, k: s, alpha: 1,
                                                  a: ds, lda: s, b: k, ldb: 3 * c, beta: 0, c: dq, ldc: 3 * c)
                                    MiniMath.gemm(transA: true, transB: false, m: s, n: hs, k: s, alpha: 1,
                                                  a: ds, lda: s, b: q, ldb: 3 * c, beta: 0, c: dk, ldc: 3 * c)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - その他

    /// into += a
    static func addInPlace(_ a: [Float], into: inout [Float]) {
        let n = min(a.count, into.count)
        a.withUnsafeBufferPointer { a in
            into.withUnsafeMutableBufferPointer { y in
                MiniVector.add(a.baseAddress!, y.baseAddress!, y.baseAddress!, n)
            }
        }
    }
}
