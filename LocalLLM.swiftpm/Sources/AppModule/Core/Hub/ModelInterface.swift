import Foundation

/// Core ML のテンソルの仕様（metadata.json からも、端末上の MLModelDescription からも作れる共通形式）。
struct TensorSpec: Codable, Hashable, Sendable {
    enum DataType: String, Codable, Sendable {
        case int32, float16, float32, float64, other

        init(metadataName: String) {
            switch metadataName.lowercased() {
            case "int32": self = .int32
            case "float16": self = .float16
            case "float32", "float": self = .float32
            case "double", "float64": self = .float64
            default: self = .other
            }
        }

        var byteSize: Int {
            switch self {
            case .int32, .float32: return 4
            case .float16: return 2
            case .float64: return 8
            case .other: return 2
            }
        }

        var isFloat: Bool { self == .float16 || self == .float32 || self == .float64 }
    }

    var name: String
    var dataType: DataType
    /// 既定の形状
    var shape: [Int]
    /// 可変形状の範囲（次元ごと）。固定なら nil
    var shapeRange: [ClosedRange<Int>]?

    init(name: String, dataType: DataType, shape: [Int], shapeRange: [ClosedRange<Int>]?) {
        self.name = name
        self.dataType = dataType
        self.shape = shape
        self.shapeRange = shapeRange
    }

    var rank: Int { shapeRange?.count ?? shape.count }

    /// 次元ごとの最大値（可変なら範囲の上限）
    func upperBound(axis: Int) -> Int? {
        if let shapeRange, axis < shapeRange.count { return shapeRange[axis].upperBound }
        if axis < shape.count { return shape[axis] }
        return nil
    }

    var isFlexible: Bool {
        guard let shapeRange else { return false }
        return shapeRange.contains { $0.lowerBound != $0.upperBound }
    }
}

/// モデルの入出力から判定した推論方式。
enum ModelInterface: Hashable, Sendable {
    /// input_ids と causal_mask を受け取り、KV キャッシュを MLState に持つ形式
    case statefulCausalLM(inputIDsName: String, causalMaskName: String, logitsName: String,
                          maskDataType: TensorSpec.DataType, maxContext: Int)
    /// input_ids（と任意の attention_mask）だけを受け取る形式（毎回全体を再計算）
    case statelessCausalLM(inputIDsName: String, attentionMaskName: String?, logitsName: String,
                           fixedLength: Int?, maxContext: Int)
    case unsupported(reason: String)

    var isSupported: Bool {
        if case .unsupported = self { return false }
        return true
    }

    var maxContext: Int? {
        switch self {
        case .statefulCausalLM(_, _, _, _, let maxContext): return maxContext
        case .statelessCausalLM(_, _, _, _, let maxContext): return maxContext
        case .unsupported: return nil
        }
    }

    static func classify(inputs: [TensorSpec], outputs: [TensorSpec], states: [TensorSpec]) -> ModelInterface {
        // input_ids: Int32 の 2 次元
        let idCandidates = inputs.filter { $0.dataType == .int32 && $0.rank == 2 }
        let preferredIDNames = ["input_ids", "inputIds", "inputs", "tokens"]
        guard let ids = idCandidates.first(where: { preferredIDNames.contains($0.name) })
                ?? (idCandidates.count == 1 ? idCandidates.first : nil) else {
            return .unsupported(reason: L("トークン ID の入力（input_ids）が見つかりません", "no token ID input (input_ids)"))
        }
        // logits の出力
        guard let logits = outputs.first(where: { $0.name == "logits" })
                ?? (outputs.count == 1 ? outputs.first : nil) else {
            return .unsupported(reason: L("logits の出力が見つかりません", "no logits output"))
        }

        if !states.isEmpty {
            // ステートフル: 4 次元の float のマスク
            let maskCandidates = inputs.filter { $0.name != ids.name && $0.dataType.isFloat && $0.rank == 4 }
            guard let mask = maskCandidates.first(where: { ["causal_mask", "causalMask", "attention_mask", "attentionMask"].contains($0.name) })
                    ?? (maskCandidates.count == 1 ? maskCandidates.first : nil) else {
                return .unsupported(reason: L("4 次元の因果マスク（causal_mask）の入力が見つかりません", "no 4-D causal mask input"))
            }
            let others = inputs.filter { $0.name != ids.name && $0.name != mask.name }
            if !others.isEmpty {
                let names = others.map(\.name).prefix(4).joined(separator: ", ")
                return .unsupported(reason: L("未対応の入力があります（\(names)）", "unsupported extra inputs (\(names))"))
            }
            // 1 回に入れられるトークン数が 1 だけ（デコード専用）だと、プロンプトの読み込みが極端に遅いが動く
            var limits: [Int] = []
            if let bound = ids.upperBound(axis: 1) { limits.append(bound) }
            if let bound = mask.upperBound(axis: 3) { limits.append(bound) }
            // KV キャッシュの長さ（[層, バッチ, ヘッド, 長さ, 次元] の 4 番目）
            for state in states where state.shape.count == 5 { limits.append(state.shape[3]) }
            guard let maxContext = limits.filter({ $0 > 1 }).min() ?? limits.min(), maxContext >= 2 else {
                return .unsupported(reason: L("最大の長さが分かりません", "unknown maximum length"))
            }
            return .statefulCausalLM(inputIDsName: ids.name, causalMaskName: mask.name, logitsName: logits.name,
                                     maskDataType: mask.dataType, maxContext: maxContext)
        }

        // ステートレス
        let rest = inputs.filter { $0.name != ids.name }
        var attentionMaskName: String?
        if let mask = rest.first(where: { $0.rank == 2 && ["attention_mask", "attentionMask"].contains($0.name) }) {
            attentionMaskName = mask.name
        }
        let others = rest.filter { $0.name != attentionMaskName }
        if !others.isEmpty {
            let names = others.map(\.name).prefix(4).joined(separator: ", ")
            return .unsupported(reason: L("未対応の入力があります（\(names)）", "unsupported extra inputs (\(names))"))
        }
        let fixedLength: Int? = ids.isFlexible ? nil : ids.shape.last
        guard let maxContext = ids.upperBound(axis: 1), maxContext >= 1 else {
            return .unsupported(reason: L("最大の長さが分かりません", "unknown maximum length"))
        }
        return .statelessCausalLM(inputIDsName: ids.name, attentionMaskName: attentionMaskName, logitsName: logits.name,
                                  fixedLength: fixedLength, maxContext: maxContext)
    }
}

enum CoreMLMetadataError: Error, LocalizedError {
    case invalid

    var errorDescription: String? {
        L("metadata.json の形式が想定外です", "Unexpected metadata.json format")
    }
}

/// .mlmodelc/metadata.json の内容。
struct CoreMLMetadata: Hashable, Sendable {
    var inputs: [TensorSpec]
    var outputs: [TensorSpec]
    var states: [TensorSpec]
    var specificationVersion: Int?
    var storagePrecision: String?
    var availabilityIOS: String?

    static func parse(_ data: Data) throws -> CoreMLMetadata {
        let root = try JSONSerialization.jsonObject(with: data)
        let object: [String: Any]
        if let array = root as? [[String: Any]], let first = array.first {
            object = first
        } else if let dictionary = root as? [String: Any] {
            object = dictionary
        } else {
            throw CoreMLMetadataError.invalid
        }
        func specs(_ key: String) -> [TensorSpec] {
            (object[key] as? [[String: Any]] ?? []).compactMap(spec)
        }
        var availability: String?
        if let table = object["availability"] as? [String: Any] {
            availability = table["iOS"] as? String
        }
        let version = (object["specificationVersion"] as? Int) ?? Int(object["specificationVersion"] as? String ?? "")
        return CoreMLMetadata(inputs: specs("inputSchema"), outputs: specs("outputSchema"), states: specs("stateSchema"),
                              specificationVersion: version,
                              storagePrecision: object["storagePrecision"] as? String,
                              availabilityIOS: availability)
    }

    private static func spec(_ item: [String: Any]) -> TensorSpec? {
        guard let name = item["name"] as? String else { return nil }
        let dataType = TensorSpec.DataType(metadataName: item["dataType"] as? String ?? "")
        let shape = parseIntegers(item["shape"] as? String ?? "")
        var ranges: [ClosedRange<Int>]?
        if let text = item["shapeRange"] as? String {
            let numbers = parseIntegers(text)
            if numbers.count >= 2 && numbers.count % 2 == 0 {
                ranges = stride(from: 0, to: numbers.count, by: 2).map { index in
                    let lower = numbers[index]
                    let upper = numbers[index + 1] < 0 ? Int(Int32.max) : numbers[index + 1]
                    return lower...max(lower, upper)
                }
            }
        }
        return TensorSpec(name: name, dataType: dataType, shape: shape, shapeRange: ranges)
    }

    /// "[[1, 1], [1, 2048]]" や "[24, 1, 2, 2048, 64]" から数を取り出す
    private static func parseIntegers(_ text: String) -> [Int] {
        var numbers: [Int] = []
        var current = ""
        for character in text {
            if character.isNumber || (character == "-" && current.isEmpty) {
                current.append(character)
            } else if !current.isEmpty {
                if let number = Int(current) { numbers.append(number) }
                current = ""
            }
        }
        if let number = Int(current) { numbers.append(number) }
        return numbers
    }

    var interface: ModelInterface {
        ModelInterface.classify(inputs: inputs, outputs: outputs, states: states)
    }

    /// KV キャッシュ（state）の合計バイト数
    var stateBytes: Int64 {
        states.reduce(Int64(0)) { total, state in
            total + state.shape.reduce(Int64(1)) { $0 * Int64(max($1, 1)) } * Int64(state.dataType.byteSize)
        }
    }
}

/// メモリ使用量の見積もり。
enum MemoryEstimator {
    /// 重み + KV キャッシュ + logits の作業領域（語彙数 × 1 回に読む長さ × 2 バイト × 2）+ 固定分（約 150MB）。
    /// 実際の使用量はモデルの作りや計算ユニットで変わるので、目安として使う。
    static func estimateBytes(weightBytes: Int64, stateBytes: Int64, vocabSize: Int?, prefillChunk: Int) -> Int64 {
        let vocab = Int64(vocabSize ?? 152_000)
        let logits = vocab * Int64(max(prefillChunk, 1)) * 2 * 2
        let overhead: Int64 = 150_000_000
        return weightBytes + stateBytes + logits + overhead
    }
}
