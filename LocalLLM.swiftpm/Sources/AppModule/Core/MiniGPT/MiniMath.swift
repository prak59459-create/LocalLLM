import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// MiniGPT 用の行列演算（行優先）。
/// iPad では Accelerate の BLAS（cblas_sgemm）を使う。Linux のテストでは同じ計算を素朴なループで行う。
enum MiniMath {
    /// C = alpha · op(A) · op(B) + beta · C
    /// - op(A) は M×K、op(B) は K×N、C は M×N（すべて行優先、lda / ldb / ldc は行の長さ）
    static func gemm(transA: Bool, transB: Bool, m: Int, n: Int, k: Int, alpha: Float,
                     a: UnsafePointer<Float>, lda: Int, b: UnsafePointer<Float>, ldb: Int,
                     beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        guard m > 0, n > 0 else { return }
        #if canImport(Accelerate)
        cblas_sgemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    Int32(m), Int32(n), Int32(k), alpha,
                    a, Int32(lda), b, Int32(ldb), beta, c, Int32(ldc))
        #else
        naiveGemm(transA: transA, transB: transB, m: m, n: n, k: k, alpha: alpha,
                  a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc)
        #endif
    }

    /// BLAS と同じ計算の素朴な実装（テストでの比較用にも使う）
    static func naiveGemm(transA: Bool, transB: Bool, m: Int, n: Int, k: Int, alpha: Float,
                          a: UnsafePointer<Float>, lda: Int, b: UnsafePointer<Float>, ldb: Int,
                          beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        for i in 0..<m {
            let row = c + i * ldc
            if beta == 0 {
                for j in 0..<n { row[j] = 0 }
            } else if beta != 1 {
                for j in 0..<n { row[j] *= beta }
            }
        }
        guard k > 0, alpha != 0 else { return }
        if !transB {
            // C[i, :] += alpha * A[i, p] * B[p, :]（B の行を連続して読む）
            for i in 0..<m {
                let row = c + i * ldc
                for p in 0..<k {
                    let aValue = alpha * (transA ? a[p * lda + i] : a[i * lda + p])
                    if aValue == 0 { continue }
                    let bRow = b + p * ldb
                    for j in 0..<n { row[j] += aValue * bRow[j] }
                }
            }
        } else {
            // C[i, j] += alpha * dot(A[i, :], B[j, :])
            for i in 0..<m {
                let row = c + i * ldc
                for j in 0..<n {
                    let bRow = b + j * ldb
                    var sum: Float = 0
                    if transA {
                        for p in 0..<k { sum += a[p * lda + i] * bRow[p] }
                    } else {
                        let aRow = a + i * lda
                        for p in 0..<k { sum += aRow[p] * bRow[p] }
                    }
                    row[j] += alpha * sum
                }
            }
        }
    }

    /// y += x（長さ count）
    static func add(_ x: UnsafePointer<Float>, into y: UnsafeMutablePointer<Float>, count: Int) {
        #if canImport(Accelerate)
        vDSP_vadd(x, 1, y, 1, y, 1, vDSP_Length(count))
        #else
        for i in 0..<count { y[i] += x[i] }
        #endif
    }

    /// 各行に bias を足す
    static func addBias(_ bias: UnsafePointer<Float>, to matrix: UnsafeMutablePointer<Float>, rows: Int, columns: Int) {
        for r in 0..<rows { add(bias, into: matrix + r * columns, count: columns) }
    }

    /// 列ごとの合計を grad に足す（bias の勾配）
    static func addColumnSums(_ matrix: UnsafePointer<Float>, rows: Int, columns: Int, into grad: UnsafeMutablePointer<Float>) {
        for r in 0..<rows { add(matrix + r * columns, into: grad, count: columns) }
    }

    /// 内積
    static func dot(_ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>, count: Int) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_dotpr(x, 1, y, 1, &result, vDSP_Length(count))
        return result
        #else
        var sum: Float = 0
        for i in 0..<count { sum += x[i] * y[i] }
        return sum
        #endif
    }

    /// 二乗の和
    static func sumOfSquares(_ x: UnsafePointer<Float>, count: Int) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_svesq(x, 1, &result, vDSP_Length(count))
        return result
        #else
        var sum: Float = 0
        for i in 0..<count { sum += x[i] * x[i] }
        return sum
        #endif
    }
}

/// 要素ごとの演算（iPad では vDSP / vForce）。
/// Swift Playgrounds は最適化なしでビルドするので、1 要素ずつのループはなるべくここに集める。
/// 出力は入力と同じ場所でもよい。
enum MiniVector {
    typealias In = UnsafePointer<Float>
    typealias Out = UnsafeMutablePointer<Float>

    /// out = a + b
    static func add(_ a: In, _ b: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        vDSP_vadd(a, 1, b, 1, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] + b[i] }
        #endif
    }

    /// out = a * b
    static func multiply(_ a: In, _ b: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        vDSP_vmul(a, 1, b, 1, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] * b[i] }
        #endif
    }

    /// out = a / b
    static func divide(_ a: In, _ b: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        vDSP_vdiv(b, 1, a, 1, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] / b[i] }
        #endif
    }

    /// out = a * b + c
    static func multiplyAdd(_ a: In, _ b: In, _ c: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        vDSP_vma(a, 1, b, 1, c, 1, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] * b[i] + c[i] }
        #endif
    }

    /// out = a * s
    static func scale(_ a: In, _ s: Float, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var s = s
        vDSP_vsmul(a, 1, &s, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] * s }
        #endif
    }

    /// out = a * s + b（b はベクトル）
    static func scaleAdd(_ a: In, _ s: Float, _ b: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var s = s
        vDSP_vsma(a, 1, &s, b, 1, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] * s + b[i] }
        #endif
    }

    /// out = a * s + t（s, t はスカラー）
    static func scaleShift(_ a: In, _ s: Float, _ t: Float, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var s = s, t = t
        vDSP_vsmsa(a, 1, &s, &t, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] * s + t }
        #endif
    }

    /// out = a + t（t はスカラー）
    static func shift(_ a: In, _ t: Float, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var t = t
        vDSP_vsadd(a, 1, &t, out, 1, vDSP_Length(n))
        #else
        for i in 0..<n { out[i] = a[i] + t }
        #endif
    }

    static func fillZero(_ out: Out, _ n: Int) {
        guard n > 0 else { return }
        out.update(repeating: 0, count: n)
    }

    static func sum(_ a: In, _ n: Int) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_sve(a, 1, &result, vDSP_Length(n))
        return result
        #else
        var result: Float = 0
        for i in 0..<n { result += a[i] }
        return result
        #endif
    }

    static func mean(_ a: In, _ n: Int) -> Float {
        n > 0 ? sum(a, n) / Float(n) : 0
    }

    static func maximum(_ a: In, _ n: Int) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_maxv(a, 1, &result, vDSP_Length(n))
        return result
        #else
        var result = -Float.infinity
        for i in 0..<n where a[i] > result { result = a[i] }
        return result
        #endif
    }

    static func exp(_ a: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var count = Int32(n)
        vvexpf(out, a, &count)
        #else
        for i in 0..<n { out[i] = Foundation.exp(a[i]) }
        #endif
    }

    static func tanh(_ a: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var count = Int32(n)
        vvtanhf(out, a, &count)
        #else
        for i in 0..<n { out[i] = Foundation.tanh(a[i]) }
        #endif
    }

    static func sqrt(_ a: In, _ out: Out, _ n: Int) {
        #if canImport(Accelerate)
        var count = Int32(n)
        vvsqrtf(out, a, &count)
        #else
        for i in 0..<n { out[i] = a[i].squareRoot() }
        #endif
    }
}

/// 決定的な正規乱数（Box–Muller）
struct MiniRandom {
    private var generator: SplitMix64
    private var spare: Double?

    init(seed: UInt64) {
        generator = SplitMix64(seed: seed)
    }

    mutating func nextUniform() -> Double {
        Double(generator.next() >> 11) / Double(1 << 53)
    }

    mutating func nextInt(_ upperBound: Int) -> Int {
        guard upperBound > 0 else { return 0 }
        return Int(generator.next() % UInt64(upperBound))
    }

    mutating func nextNormal() -> Double {
        if let spare {
            self.spare = nil
            return spare
        }
        var u1 = nextUniform()
        if u1 < 1e-300 { u1 = 1e-300 }
        let u2 = nextUniform()
        let radius = (-2 * log(u1)).squareRoot()
        spare = radius * sin(2 * .pi * u2)
        return radius * cos(2 * .pi * u2)
    }
}
