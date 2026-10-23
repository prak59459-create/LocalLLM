import CoreML
import Foundation

/// 端末上で読み込んだ MLModel の入出力を、Core の共通形式（TensorSpec）に変換する。
/// こうすると metadata.json（ダウンロード前の判定）と同じ判定ロジック（ModelInterface.classify）を使える。
extension TensorSpec.DataType {
    init(_ type: MLMultiArrayDataType) {
        switch type {
        case .int32: self = .int32
        case .float16: self = .float16
        case .float32: self = .float32
        case .double: self = .float64
        default: self = .other
        }
    }
}

extension TensorSpec {
    /// 入力・出力（MultiArray 型）の説明から作る。MultiArray 以外なら nil。
    init?(feature: MLFeatureDescription) {
        guard let constraint = feature.multiArrayConstraint else {
            // MultiArray 以外（画像や文字列など）。判定では「対応外の入力」として扱えるよう other で返す。
            self.init(name: feature.name, dataType: .other, shape: [], shapeRange: nil)
            return
        }
        let shape = constraint.shape.map(\.intValue)
        var ranges: [ClosedRange<Int>]? = nil
        let shapeConstraint = constraint.shapeConstraint
        switch shapeConstraint.type {
        case .range:
            ranges = shapeConstraint.sizeRangeForDimension.map { value in
                let range = value.rangeValue
                let lower = max(range.location, 0)
                // NSRange の length が極端に大きい場合は上限なし（Int.max を避けて大きな値にする）。
                let upper: Int
                if range.length <= 0 || range.length >= Int(Int32.max) {
                    upper = max(lower, 1 << 20)
                } else {
                    upper = max(lower, range.location + range.length - 1)
                }
                return lower...upper
            }
        case .enumerated:
            // 列挙された形状の中で、次元ごとの最小〜最大を範囲とみなす。
            let shapes = shapeConstraint.enumeratedShapes.map { $0.map(\.intValue) }
            if let first = shapes.first {
                ranges = (0..<first.count).map { axis in
                    let values = shapes.compactMap { axis < $0.count ? $0[axis] : nil }
                    return (values.min() ?? 1)...(values.max() ?? 1)
                }
            }
        default:
            break
        }
        self.init(name: feature.name, dataType: TensorSpec.DataType(constraint.dataType), shape: shape, shapeRange: ranges)
    }

    /// state（KV キャッシュ）の説明から作る。
    init?(state feature: MLFeatureDescription) {
        guard let constraint = feature.stateConstraint else { return nil }
        self.init(name: feature.name,
                  dataType: TensorSpec.DataType(constraint.dataType),
                  shape: constraint.bufferShape.map { Int(truncating: $0 as NSNumber) },
                  shapeRange: nil)
    }
}

extension MLModelDescription {
    /// 入力・出力・state を TensorSpec の配列にする（名前順）。
    var tensorSpecs: (inputs: [TensorSpec], outputs: [TensorSpec], states: [TensorSpec]) {
        let inputs = inputDescriptionsByName.values
            .sorted { $0.name < $1.name }
            .compactMap { TensorSpec(feature: $0) }
        let outputs = outputDescriptionsByName.values
            .sorted { $0.name < $1.name }
            .compactMap { TensorSpec(feature: $0) }
        let states = stateDescriptionsByName.values
            .sorted { $0.name < $1.name }
            .compactMap { TensorSpec(state: $0) }
        return (inputs, outputs, states)
    }
}
