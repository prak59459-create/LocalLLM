import Accelerate
import CoreML
import Foundation

/// 次のトークンを選ぶための情報。
enum NextTokenScores {
    /// 語彙ごとの logits（サンプラーで選ぶ）
    case logits([Float])
    /// モデルの中で選ばれたトークン（出力層が argmax まで計算する形式。温度などは使えない）
    case token(Int)
}

/// Core ML の言語モデルを動かすもの（1 つのモデル、または分割モデル）。
/// どれもスレッドセーフではないので、InferenceEngine（actor）の中からだけ使う。
protocol LanguageModelRunner: AnyObject {
    var maxContext: Int { get }
    var interfaceDescription: String { get }
    /// 端末で使った計算ユニットの説明
    var computeDescription: String { get }

    /// モデル内部の系列を tokens にそろえ、最後のトークンの次のスコアを返す（共通の先頭部分は再計算しない）。
    func evaluate(tokens: [Int], chunkSize: Int, progress: ((Int, Int) -> Void)?) async throws
        -> (scores: NextTokenScores, reusedTokens: Int)
    /// 1 トークン追加して次のスコアを返す。
    func append(token: Int) async throws -> NextTokenScores
    /// キャッシュを捨てる。
    func resetCache()
}

/// MLMultiArray の作成・読み出しの共通処理（最適化なしのビルドでも速いように、まとめて書き込む）。
enum CoreMLTensors {
    static func int32Array(_ values: [Int], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { pointer, strides in
            let stride = strides.last ?? 1
            for (index, value) in values.enumerated() {
                pointer[index * stride] = Int32(truncatingIfNeeded: value)
            }
        }
        return array
    }

    /// 因果マスク [1, 1, rows, columns]。行 r は列 0...(firstRowPosition + r) が見える（0）、ほかは大きな負の値。
    static func causalMask(rows: Int, columns: Int, firstRowPosition: Int, dataType: TensorSpec.DataType) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: rows), NSNumber(value: columns)]
        if dataType == .float32 {
            let array = try MLMultiArray(shape: shape, dataType: .float32)
            array.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, strides in
                fillMask(pointer, rowStride: strides[2], columnStride: strides[3], rows: rows, columns: columns,
                         pastLength: firstRowPosition, visible: 0, hidden: -Float.greatestFiniteMagnitude)
            }
            return array
        }
        let array = try MLMultiArray(shape: shape, dataType: .float16)
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { pointer, strides in
            fillMask(pointer, rowStride: strides[2], columnStride: strides[3], rows: rows, columns: columns,
                     pastLength: firstRowPosition, visible: 0, hidden: -Float16.greatestFiniteMagnitude)
        }
        return array
    }

    static func fillMask<T>(_ pointer: UnsafeMutableBufferPointer<T>, rowStride: Int, columnStride: Int,
                            rows: Int, columns: Int, pastLength: Int, visible: T, hidden: T) {
        guard let base = pointer.baseAddress else { return }
        for row in 0..<rows {
            let rowStart = base + row * rowStride
            let visibleCount = max(0, min(pastLength + row + 1, columns))
            if columnStride == 1 {
                rowStart.update(repeating: visible, count: visibleCount)
                if columns > visibleCount { (rowStart + visibleCount).update(repeating: hidden, count: columns - visibleCount) }
            } else {
                for column in 0..<columns { rowStart[column * columnStride] = column < visibleCount ? visible : hidden }
            }
        }
    }

    /// 1 行（count 個）を Float に変換して output[offset...] に書く。
    static func copyRow(of array: MLMultiArray, elementOffset: Int, count: Int, stride: Int,
                        into output: inout [Float], at outputOffset: Int) {
        guard count > 0, outputOffset + count <= output.count else { return }
        switch array.dataType {
        case .float16:
            array.withUnsafeBufferPointer(ofType: Float16.self) { pointer in
                guard let base = pointer.baseAddress else { return }
                if stride == 1 {
                    output.withUnsafeMutableBufferPointer { out in
                        var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: base + elementOffset), height: 1,
                                                   width: vImagePixelCount(count), rowBytes: count * 2)
                        var destination = vImage_Buffer(data: out.baseAddress! + outputOffset, height: 1,
                                                        width: vImagePixelCount(count), rowBytes: count * 4)
                        _ = vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
                    }
                } else {
                    for index in 0..<count { output[outputOffset + index] = Float(base[elementOffset + index * stride]) }
                }
            }
        case .float32:
            array.withUnsafeBufferPointer(ofType: Float.self) { pointer in
                guard let base = pointer.baseAddress else { return }
                if stride == 1 {
                    output.withUnsafeMutableBufferPointer { out in
                        (out.baseAddress! + outputOffset).update(from: base + elementOffset, count: count)
                    }
                } else {
                    for index in 0..<count { output[outputOffset + index] = base[elementOffset + index * stride] }
                }
            }
        case .double:
            array.withUnsafeBufferPointer(ofType: Double.self) { pointer in
                for index in 0..<count { output[outputOffset + index] = Float(pointer[elementOffset + index * stride]) }
            }
        default:
            for index in 0..<count { output[outputOffset + index] = array[elementOffset + index * stride].floatValue }
        }
    }

    /// [1, n, V] / [n, V] / [1, V] / [V] の position 行目の語彙ぶんの先頭位置・語彙数・間隔
    static func rowLayout(of array: MLMultiArray, position: Int) throws -> (offset: Int, count: Int, stride: Int) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard let vocab = shape.last, vocab > 0 else { throw EngineError.unexpectedOutputShape(shape) }
        let offset: Int
        switch shape.count {
        case 3:
            guard position < shape[1] else { throw EngineError.unexpectedOutputShape(shape) }
            offset = position * strides[1]
        case 2:
            offset = shape[0] == 1 ? 0 : position * strides[0]
        case 1:
            offset = 0
        default:
            throw EngineError.unexpectedOutputShape(shape)
        }
        return (offset, vocab, strides.last ?? 1)
    }

    /// 行の位置の数（[1, n, V] なら n）
    static func positions(of array: MLMultiArray) -> Int {
        let shape = array.shape.map(\.intValue)
        switch shape.count {
        case 3: return shape[1]
        case 2: return shape[0]
        default: return 1
        }
    }
}

extension ComputeUnitsOption {
    /// 「自動」のときは、分割モデル（Neural Engine 向け）なら CPU + Neural Engine、それ以外は CPU + GPU
    func resolved(isPipeline: Bool) -> MLComputeUnits {
        if self == .automatic { return isPipeline ? .cpuAndNeuralEngine : .cpuAndGPU }
        return computeUnits
    }
}

extension MLComputeUnits {
    var label: String {
        switch self {
        case .all: return L("すべて", "All")
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .cpuOnly: return L("CPU のみ", "CPU only")
        @unknown default: return "?"
        }
    }
}
