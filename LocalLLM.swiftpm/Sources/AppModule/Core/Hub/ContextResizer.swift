import Foundation

/// コンパイル済みモデル（.mlmodelc）の最大コンテキスト長を変える。
///
/// 重み（weights/weight.bin）は変えずに、次のものだけを書き換える:
/// - model.mil（MIL のテキスト）: KV キャッシュの state の形、そこから切り出すテンソルの形と切り出し位置、
///   可変長入力の上限（RangeDims）、固定長モデル（ANEMLL）のマスクとアテンションの形
/// - coremldata.bin（入出力の説明）: 入力の上限、state の形
/// - metadata.json（アプリが読み込み前に読む情報）
///
/// RoPE の sin / cos の表が重みに入っているモデル（ANEMLL）は、表を新しい長さで作り直して
/// weights/context_rope.bin に書き、その表を指すように書き換える。
enum ContextResizer {
    enum ResizeError: Error, LocalizedError {
        case missingFile(String)
        case noStatefulCache
        case unknownContext
        case unsupported(String)
        case nothingChanged

        var errorDescription: String? {
            switch self {
            case .missingFile(let name):
                return L("モデルの中に \(name) がありません", "\(name) is missing from the model")
            case .noStatefulCache:
                return L("KV キャッシュ（state）を持つモデルではないため、コンテキスト長を変えられません。",
                         "This model has no KV cache state, so its context length cannot be changed.")
            case .unknownContext:
                return L("今のコンテキスト長が分かりません", "Cannot determine the current context length")
            case .unsupported(let reason):
                return L("このモデルのコンテキスト長は変えられません: \(reason)", "The context length of this model cannot be changed: \(reason)")
            case .nothingChanged:
                return L("書き換える場所が見つかりませんでした", "Nothing to rewrite was found")
            }
        }
    }

    /// モデルの今の状態
    struct Analysis: Sendable {
        /// 今の最大コンテキスト長
        var context: Int
        /// KV キャッシュの形（例 [24, 1, 2, 2048, 64]）
        var stateShapes: [[Int]]
        /// state の中でコンテキストの軸
        var axis: Int
        /// 入力の形が固定（ANEMLL など）
        var fixedShapes: Bool
        /// 関数の名前（multifunction）
        var functions: [String]
        /// 重みに入っている RoPE の表
        var ropeTables: [RopeReference]
        /// 1 トークンあたりの KV キャッシュのバイト数（float16 として）
        var kvBytesPerToken: Int64

        var stateBytes: Int64 { kvBytesPerToken * Int64(context) }
    }

    /// 重みに入っている RoPE の表（const の BLOBFILE）
    struct RopeReference: Sendable, Hashable {
        var variable: String
        var dataType: String
        var shape: [Int]
        /// 位置の軸（shape の中で行の数の軸）
        var rowAxis: Int
        var path: String
        var offset: UInt64

        var rows: Int { shape[rowAxis] }
        var columns: Int { shape.last ?? 0 }
    }

    // MARK: - 解析

    private static let listPattern = try! NSRegularExpression(pattern: #"\[(-?[0-9?][-0-9?, ]*)\]"#)
    private static let statePattern = try! NSRegularExpression(pattern: #"state<tensor<(\w+), \[([0-9, ]+)\]>>"#)
    private static let maskInputPattern = try! NSRegularExpression(pattern: #"tensor<\w+, \[([0-9?, ]+)\]> (\w*mask\w*)"#)
    private static let rangePattern = try! NSRegularExpression(pattern: #"\[(\d+), (\d+)\]"#)
    private static let funcPattern = try! NSRegularExpression(pattern: #"^\s*func (\w+)<"#)
    private static let blobConstPattern = try! NSRegularExpression(
        pattern: #"^\s*tensor<(\w+), \[([0-9, ]+)\]> (\w+) = const\(\)\[.*BLOBFILE\(path = string\("([^"]+)"\), offset = uint64\((\d+)\)\)"#)
    private static let gatherPattern = try! NSRegularExpression(pattern: #"= gather\((.*?)\)\[name"#)

    static func parseList(_ text: String) -> [Int?] {
        text.split(separator: ",").map { item in
            let trimmed = item.trimmingCharacters(in: .whitespaces)
            return trimmed == "?" ? nil : Int(trimmed)
        }
    }

    static func formatList(_ values: [Int?]) -> String {
        "[" + values.map { $0.map(String.init) ?? "?" }.joined(separator: ", ") + "]"
    }

    private static func matches(_ regex: NSRegularExpression, _ text: String) -> [[String]] {
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map { match in
            (0..<match.numberOfRanges).map { index in
                Range(match.range(at: index), in: text).map { String(text[$0]) } ?? ""
            }
        }
    }

    static func analyze(mil: String) throws -> Analysis {
        let lines = mil.components(separatedBy: "\n")
        var stateShapes: [[Int]] = []
        var stateBytes: [Int] = []
        var maskDims: [Int] = []
        var rangeUppers: [Int] = []
        var functions: [String] = []
        for line in lines where line.contains("func ") {
            guard let name = matches(funcPattern, line).first?[1] else { continue }
            functions.append(name)
            for match in matches(statePattern, line) {
                let shape = parseList(match[2]).compactMap { $0 }
                stateShapes.append(shape)
                stateBytes.append(match[1].hasPrefix("fp32") ? 4 : 2)
            }
            for match in matches(maskInputPattern, line) {
                maskDims.append(contentsOf: parseList(match[1]).compactMap { $0 })
            }
            if let range = line.range(of: "RangeDims") {
                for match in matches(rangePattern, String(line[range.lowerBound...])) {
                    if let upper = Int(match[2]) { rangeUppers.append(upper) }
                }
            }
        }
        guard let firstState = stateShapes.first else { throw ResizeError.noStatefulCache }

        // コンテキスト長: state の次元のうち、マスクの次元か入力の上限と一致するもの
        var counts: [Int: Int] = [:]
        for shape in stateShapes {
            for dimension in Set(shape) where dimension >= 8 && (maskDims.contains(dimension) || rangeUppers.contains(dimension)) {
                counts[dimension, default: 0] += 1
            }
        }
        let context: Int
        if let best = counts.max(by: { ($0.value, $0.key) < ($1.value, $1.key) })?.key {
            context = best
        } else if firstState.count >= 2 {
            // 分からなければ、最後から 2 番目（[層, …, 長さ, 次元]）
            context = firstState[firstState.count - 2]
        } else {
            throw ResizeError.unknownContext
        }
        guard context >= 8, let axis = firstState.indices.last(where: { firstState[$0] == context && $0 < firstState.count - 1 }) else {
            throw ResizeError.unknownContext
        }
        if functions.contains(where: { $0.lowercased().contains("rotate") }) {
            throw ResizeError.unsupported(L("スライディング窓（rotate）の関数があるモデルには対応していません",
                                            "models with sliding-window (rotate) functions are not supported"))
        }

        // 1 トークンあたりの KV キャッシュ
        var perToken: Int64 = 0
        for (index, shape) in stateShapes.enumerated() {
            let elements = shape.reduce(Int64(1)) { $0 * Int64($1) }
            perToken += elements / Int64(max(context, 1)) * Int64(stateBytes[index])
        }

        let fixed = maskDims.contains(context)
        let trailing = Array(firstState[(axis + 1)...])
        let ropeTables = findRopeTables(lines: lines, headDimensions: Set([trailing.last ?? 0]).filter { $0 > 0 })
        return Analysis(context: context, stateShapes: stateShapes, axis: axis, fixedShapes: fixed, functions: functions,
                        ropeTables: ropeTables, kvBytesPerToken: perToken)
    }

    /// gather で位置ごとに引かれる、重みの中の表（[1, 行, 次元] や [行, 次元]）
    private static func findRopeTables(lines: [String], headDimensions: Set<Int>) -> [RopeReference] {
        var candidates: [String: RopeReference] = [:]
        for line in lines where line.contains("BLOBFILE") && line.contains("= const()") {
            guard let match = matches(blobConstPattern, line).first else { continue }
            let shape = parseList(match[2]).compactMap { $0 }
            guard shape.count >= 2, let last = shape.last,
                  headDimensions.contains(last) || headDimensions.contains(last * 2) else { continue }
            // 位置の軸: 最後以外で 1 でない唯一の軸
            let nonUnit = shape.indices.dropLast().filter { shape[$0] != 1 }
            guard nonUnit.count == 1, let rowAxis = nonUnit.first, shape[rowAxis] >= 16 else { continue }
            candidates[match[3]] = RopeReference(variable: match[3], dataType: match[1], shape: shape, rowAxis: rowAxis,
                                                 path: match[4], offset: UInt64(match[5]) ?? 0)
        }
        guard !candidates.isEmpty else { return [] }
        var used: Set<String> = []
        for line in lines where line.contains("= gather(") {
            for name in candidates.keys where line.contains("x = \(name))") || line.contains("x = \(name),") {
                used.insert(name)
            }
        }
        return candidates.values.filter { used.contains($0.variable) }.sorted { $0.offset < $1.offset }
    }

    // MARK: - MIL の書き換え

    /// テンソルの形・値のリストのうち、コンテキストの軸に当たる位置（無ければ nil）
    static func contextIndex(in list: [Int?], analysis: Analysis) -> Int? {
        let context = analysis.context
        let state = analysis.stateShapes[0]
        let trailing = Array(state[(analysis.axis + 1)...])
        // 1. KV キャッシュと同じ並び（後ろの次元が head_dim などと一致）
        if !trailing.isEmpty, list.count <= state.count {
            for index in list.indices.reversed() where list[index] == context {
                let after = Array(list[(index + 1)...])
                guard after.count == trailing.count, zip(after, trailing).allSatisfy({ $0 == $1 }) else { continue }
                let before = list[..<index]
                if before.allSatisfy({ value in value.map { $0 > 0 || $0 == -1 } ?? false }) { return index }
            }
        }
        // 2. 固定長モデルのマスク・アテンション（[1, ヘッド, クエリ, 長さ] や [ヘッド, クエリ, 長さ]）
        if analysis.fixedShapes, let last = list.last, last == context, list.count == 3 || list.count == 4 {
            let values = list.compactMap { $0 }
            guard values.count == list.count else { return nil }
            if list.count == 4 && values[0] != 1 { return nil }
            if list.count == 3 && values[0] == 1 { return nil }
            if values.dropLast().allSatisfy({ $0 < context }) { return list.count - 1 }
        }
        return nil
    }

    /// MIL のテキストを書き換える。戻り値は新しいテキストと、書き換えた数。
    static func rewriteMIL(_ mil: String, analysis: Analysis, newContext: Int,
                           ropeRedirects: [String: (path: String, offset: UInt64, rows: Int)] = [:]) throws -> (text: String, count: Int) {
        let context = analysis.context
        var count = 0
        var lines = mil.components(separatedBy: "\n")
        var castsToWiden: Set<String> = []
        if newContext > 32_767, !ropeRedirects.isEmpty {
            // 位置を 16 ビットの整数に変えて表を引いていると、32767 を超えたところで壊れるので 32 ビットにする
            for line in lines where line.contains("= gather(") {
                guard ropeRedirects.keys.contains(where: { line.contains("x = \($0))") || line.contains("x = \($0),") }),
                      let range = line.range(of: "indices = ") else { continue }
                let rest = line[range.upperBound...]
                let name = rest.prefix { $0.isLetter || $0.isNumber || $0 == "_" }
                if !name.isEmpty { castsToWiden.insert(String(name)) }
            }
        }

        var index = 0
        while index < lines.count {
            var line = lines[index]
            if line.contains("BLOBFILE") {
                // 重みの行は変えない（RoPE の表だけは作り直した表を指すようにする）
                if let match = matches(blobConstPattern, line).first, let redirect = ropeRedirects[match[3]],
                   let reference = analysis.ropeTables.first(where: { $0.variable == match[3] }) {
                    var shape = reference.shape
                    shape[reference.rowAxis] = redirect.rows
                    let oldType = formatList(reference.shape.map { $0 })
                    line = line.replacingOccurrences(of: oldType, with: formatList(shape.map { $0 }))
                    line = line.replacingOccurrences(of: "path = string(\"\(reference.path)\"), offset = uint64(\(reference.offset))",
                                                     with: "path = string(\"\(redirect.path)\"), offset = uint64(\(redirect.offset))")
                    count += 1
                }
                lines[index] = line
                index += 1
                continue
            }
            let isSignature = matches(funcPattern, line).first != nil
            if isSignature {
                // 可変長入力の上限
                if let range = line.range(of: "RangeDims") {
                    let head = String(line[..<range.lowerBound])
                    var tail = String(line[range.lowerBound...])
                    let found = matches(rangePattern, tail)
                    for match in found where Int(match[2]) == context {
                        tail = tail.replacingOccurrences(of: match[0], with: "[\(match[1]), \(newContext)]")
                        count += 1
                    }
                    line = head + tail
                }
                // マスクの入力（固定長）
                for match in matches(maskInputPattern, line) {
                    let values = parseList(match[1])
                    guard values.contains(context) else { continue }
                    let replaced = values.map { $0 == context ? newContext : $0 }
                    line = line.replacingOccurrences(of: "[\(match[1])]> \(match[2])", with: "\(formatList(replaced))> \(match[2])")
                    count += 1
                }
            }
            // 形と値のリスト
            let ns = line as NSString
            let found = listPattern.matches(in: line, range: NSRange(location: 0, length: ns.length))
            if !found.isEmpty {
                var rebuilt = ""
                var cursor = 0
                for match in found {
                    let inner = ns.substring(with: match.range(at: 1))
                    var values = parseList(inner)
                    rebuilt += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
                    if values.contains(context), let position = contextIndex(in: values, analysis: analysis) {
                        values[position] = newContext
                        rebuilt += formatList(values)
                        count += 1
                    } else {
                        rebuilt += ns.substring(with: match.range)
                    }
                    cursor = match.range.location + match.range.length
                }
                rebuilt += ns.substring(from: cursor)
                line = rebuilt
            }
            // 表を引く位置の型を 32 ビットに
            if !castsToWiden.isEmpty, let name = castsToWiden.first(where: { line.contains("> \($0) = cast(") }) {
                for type in ["int16", "uint16"] {
                    line = line.replacingOccurrences(of: "tensor<\(type), ", with: "tensor<int32, ")
                    if line.hasPrefix("\(type) ") { line = "int32" + line.dropFirst(type.count) }
                }
                if let dtypeRange = line.range(of: "cast(dtype = ") {
                    let rest = line[dtypeRange.upperBound...]
                    let dtypeVar = String(rest.prefix { $0.isLetter || $0.isNumber || $0 == "_" })
                    let newVar = dtypeVar + "_ctx_int32"
                    line = line.replacingOccurrences(of: "cast(dtype = \(dtypeVar),", with: "cast(dtype = \(newVar),")
                    let indent = String(line.prefix { $0 == " " })
                    lines.insert("\(indent)string \(newVar) = const()[name = string(\"\(newVar)\"), val = string(\"int32\")];", at: index)
                    index += 1
                    count += 1
                }
                castsToWiden.remove(name)
            }
            lines[index] = line
            index += 1
        }
        guard count > 0 else { throw ResizeError.nothingChanged }
        return (lines.joined(separator: "\n"), count)
    }

    // MARK: - coremldata.bin の書き換え

    /// coremldata.bin の中の ModelDescription（protobuf）の位置
    private static func descriptionRange(in bytes: [UInt8]) -> (lengthOffset: Int, start: Int, length: Int)? {
        // 既知の位置（75）を先に試し、だめなら近くを探す
        var candidates = [75]
        candidates.append(contentsOf: (8..<min(bytes.count - 8, 400)).filter { $0 != 75 })
        for offset in candidates where offset + 8 <= bytes.count {
            let length = Int(MILBlob.readUInt64(bytes, offset))
            let start = offset + 8
            guard length > 0, start + length + 4 <= bytes.count,
                  MILBlob.readUInt32(bytes, start + length) == MILBlob.readUInt32(bytes, 0),
                  let parsed = try? ProtoWire.parse(Data(bytes[start..<(start + length)])),
                  parsed.contains(where: { [1, 10, 13, 20].contains($0.number) }) else { continue }
            return (offset, start, length)
        }
        return nil
    }

    static func rewriteDescriptionFile(_ data: Data, analysis: Analysis, newContext: Int) throws -> Data {
        let bytes = [UInt8](data)
        guard let range = descriptionRange(in: bytes) else {
            throw ResizeError.unsupported(L("coremldata.bin の形式が想定外です", "unexpected coremldata.bin format"))
        }
        let description = try ProtoWire.parse(Data(bytes[range.start..<(range.start + range.length)]))
        let rewritten = ProtoWire.serialize(try rewriteDescription(description, analysis: analysis, newContext: newContext))
        var output = Array(bytes[..<range.lengthOffset])
        var length = [UInt8](repeating: 0, count: 8)
        MILBlob.writeUInt64(UInt64(rewritten.count), into: &length, at: 0)
        output.append(contentsOf: length)
        output.append(contentsOf: rewritten)
        output.append(contentsOf: bytes[(range.start + range.length)...])
        return Data(output)
    }

    /// ModelDescription（input = 1, output = 10, state = 13, functions = 20, metadata = 100）
    static func rewriteDescription(_ fields: [ProtoField], analysis: Analysis, newContext: Int) throws -> [ProtoField] {
        var result = try ProtoWire.mapMessages(fields, number: 1) { try rewriteFeature($0, role: .input, analysis: analysis, newContext: newContext) }
        result = try ProtoWire.mapMessages(result, number: 13) { try rewriteFeature($0, role: .state, analysis: analysis, newContext: newContext) }
        result = try ProtoWire.mapMessages(result, number: 20) { function in
            // FunctionDescription（input = 2, output = 3, state = 6）
            var updated = try ProtoWire.mapMessages(function, number: 2) { try rewriteFeature($0, role: .input, analysis: analysis, newContext: newContext) }
            updated = try ProtoWire.mapMessages(updated, number: 6) { try rewriteFeature($0, role: .state, analysis: analysis, newContext: newContext) }
            return updated
        }
        result = ProtoWire.mapMessages(result, number: 100) { metadata in
            // Metadata.userDefined（100 番の map: key = 1, value = 2）
            metadata.map { entry in
                guard entry.number == 100, let data = entry.bytes, let pairs = try? ProtoWire.parse(data),
                      pairs.first(where: { $0.number == 1 })?.string == "com.anemll.context_length" else { return entry }
                let updated = pairs.map { pair in
                    pair.number == 2 ? ProtoField(number: 2, value: .bytes(Data(String(newContext).utf8))) : pair
                }
                return ProtoField(number: 100, value: .bytes(ProtoWire.serialize(updated)))
            }
        }
        return result
    }

    private enum FeatureRole {
        case input
        case state
    }

    /// FeatureDescription（name = 1, type = 3 → FeatureType: multiArrayType = 5, stateType = 8 → arrayType = 1）
    private static func rewriteFeature(_ feature: [ProtoField], role: FeatureRole, analysis: Analysis, newContext: Int) throws -> [ProtoField] {
        let name = feature.first(where: { $0.number == 1 })?.string ?? ""
        let isMask = name.lowercased().contains("mask")
        return try ProtoWire.mapMessages(feature, number: 3) { type in
            switch role {
            case .input:
                return try ProtoWire.mapMessages(type, number: 5) {
                    try rewriteArray($0, analysis: analysis, newContext: newContext, rewriteShape: isMask, isState: false)
                }
            case .state:
                return try ProtoWire.mapMessages(type, number: 8) { stateType in
                    try ProtoWire.mapMessages(stateType, number: 1) {
                        try rewriteArray($0, analysis: analysis, newContext: newContext, rewriteShape: true, isState: true)
                    }
                }
            }
        }
    }

    /// ArrayFeatureType（shape = 1, enumeratedShapes = 21, shapeRange = 31）
    private static func rewriteArray(_ array: [ProtoField], analysis: Analysis, newContext: Int,
                                     rewriteShape: Bool, isState: Bool) throws -> [ProtoField] {
        let context = Int64(analysis.context)
        func replaceShape(_ shape: [Int64]) -> [Int64] {
            if isState {
                guard let axis = shape.indices.last(where: { shape[$0] == context && $0 < shape.count - 1 }) else { return shape }
                var updated = shape
                updated[axis] = Int64(newContext)
                return updated
            }
            return shape.map { $0 == context ? Int64(newContext) : $0 }
        }
        var result: [ProtoField] = []
        var shapeValues: [Int64] = []
        var shapeIsPacked = true
        for field in array {
            if field.number == 1 {
                if let data = field.bytes {
                    shapeValues.append(contentsOf: try ProtoWire.packedVarints(data))
                } else if let value = field.varint {
                    shapeValues.append(Int64(bitPattern: value))
                    shapeIsPacked = false
                }
                continue
            }
            result.append(field)
        }
        if !shapeValues.isEmpty {
            let updated = rewriteShape ? replaceShape(shapeValues) : shapeValues
            let fields: [ProtoField] = shapeIsPacked
                ? [ProtoField(number: 1, value: .bytes(ProtoWire.packVarints(updated)))]
                : updated.map { ProtoField(number: 1, value: .varint(UInt64(bitPattern: $0))) }
            result.insert(contentsOf: fields, at: 0)
        }
        // 列挙された形（固定長のマスクなど）
        if rewriteShape && !isState {
            result = ProtoWire.mapMessages(result, number: 21) { enumerated in
                ProtoWire.mapMessages(enumerated, number: 1) { shapeMessage in
                    shapeMessage.map { field in
                        guard field.number == 1, let data = field.bytes, let values = try? ProtoWire.packedVarints(data) else { return field }
                        return ProtoField(number: 1, value: .bytes(ProtoWire.packVarints(replaceShape(values))))
                    }
                }
            }
        }
        // 可変長の上限（SizeRange: lowerBound = 1, upperBound = 2）
        result = ProtoWire.mapMessages(result, number: 31) { shapeRange in
            ProtoWire.mapMessages(shapeRange, number: 1) { sizeRange in
                sizeRange.map { field in
                    guard field.number == 2, let value = field.varint, Int64(bitPattern: value) == context else { return field }
                    return ProtoField(number: 2, value: .varint(UInt64(newContext)))
                }
            }
        }
        return result
    }

    // MARK: - metadata.json の書き換え

    static func rewriteMetadata(_ data: Data, analysis: Analysis, newContext: Int) throws -> Data {
        let root = try JSONSerialization.jsonObject(with: data)
        let context = analysis.context
        func replaceNumbers(_ text: String, onlyAxis: Bool) -> String {
            // "[24, 1, 2, 2048, 64]" の中の数を置き換える
            guard text.hasPrefix("[") else { return text }
            if text.hasPrefix("[[") {
                // shapeRange: 上限だけ
                let pairs = matches(rangePattern, text)
                var output = text
                for pair in pairs where Int(pair[2]) == context {
                    output = output.replacingOccurrences(of: pair[0], with: "[\(pair[1]), \(newContext)]")
                }
                return output
            }
            var values = parseList(String(text.dropFirst().dropLast()))
            if onlyAxis {
                guard let axis = values.indices.last(where: { values[$0] == context && $0 < values.count - 1 }) else { return text }
                values[axis] = newContext
            } else {
                values = values.map { $0 == context ? newContext : $0 }
            }
            return formatList(values)
        }
        func rewrite(_ item: [String: Any], isState: Bool) -> [String: Any] {
            var item = item
            let name = (item["name"] as? String ?? "").lowercased()
            let rewriteShape = isState || name.contains("mask")
            if rewriteShape, let shape = item["shape"] as? String {
                item["shape"] = replaceNumbers(shape, onlyAxis: isState)
            }
            if let range = item["shapeRange"] as? String {
                item["shapeRange"] = replaceNumbers(range, onlyAxis: false)
            }
            if let flexibility = item["shapeFlexibility"] as? String {
                item["shapeFlexibility"] = flexibility.replacingOccurrences(of: "...\(context)", with: "...\(newContext)")
            }
            if rewriteShape, let formatted = item["formattedType"] as? String {
                item["formattedType"] = formatted.replacingOccurrences(of: " \(context) ", with: " \(newContext) ")
                    .replacingOccurrences(of: " \(context))", with: " \(newContext))")
            }
            return item
        }
        func rewriteModel(_ object: [String: Any]) -> [String: Any] {
            var object = object
            for key in ["inputSchema", "stateSchema"] {
                if let items = object[key] as? [[String: Any]] {
                    object[key] = items.map { rewrite($0, isState: key == "stateSchema") }
                }
            }
            if var functions = object["functions"] as? [[String: Any]] {
                functions = functions.map { rewriteModel($0) }
                object["functions"] = functions
            }
            if var user = object["userDefinedMetadata"] as? [String: Any], user["com.anemll.context_length"] != nil {
                user["com.anemll.context_length"] = String(newContext)
                object["userDefinedMetadata"] = user
            }
            return object
        }
        let updated: Any
        if let array = root as? [[String: Any]] {
            updated = array.map(rewriteModel)
        } else if let object = root as? [String: Any] {
            updated = rewriteModel(object)
        } else {
            throw CoreMLMetadataError.invalid
        }
        return try JSONSerialization.data(withJSONObject: updated, options: [.prettyPrinted, .sortedKeys])
    }

    // MARK: - モデルのフォルダに対して

    static let backupFiles = ["model.mil", "coremldata.bin", "metadata.json"]
    static let ropeFileName = "context_rope.bin"

    /// .mlmodelc を調べる
    static func analyze(modelDirectory: URL) throws -> Analysis {
        let milURL = modelDirectory.appendingPathComponent("model.mil")
        guard let mil = try? String(contentsOf: milURL, encoding: .utf8) else { throw ResizeError.missingFile("model.mil") }
        return try analyze(mil: mil)
    }

    /// .mlmodelc のコンテキスト長を変える。backup には最初の 1 回だけ元のファイルを残す（restore で戻せる）。
    @discardableResult
    static func resize(modelDirectory: URL, to newContext: Int, backup: URL) throws -> Analysis {
        let fileManager = FileManager.default
        for name in backupFiles where !fileManager.fileExists(atPath: modelDirectory.appendingPathComponent(name).path) {
            if name != "metadata.json" { throw ResizeError.missingFile(name) }
        }
        // 元のファイルを残す（2 回目以降は、最初の元のまま）
        if !fileManager.fileExists(atPath: backup.appendingPathComponent("model.mil").path) {
            try fileManager.createDirectory(at: backup, withIntermediateDirectories: true)
            for name in backupFiles {
                let source = modelDirectory.appendingPathComponent(name)
                guard fileManager.fileExists(atPath: source.path) else { continue }
                try? fileManager.removeItem(at: backup.appendingPathComponent(name))
                try fileManager.copyItem(at: source, to: backup.appendingPathComponent(name))
            }
        }
        // いつも元のファイルから書き換える（何度変えても誤差がたまらないように）
        let mil = try String(contentsOf: backup.appendingPathComponent("model.mil"), encoding: .utf8)
        let analysis = try analyze(mil: mil)
        guard newContext != analysis.context else {
            try restore(modelDirectory: modelDirectory, backup: backup)
            return analysis
        }

        // RoPE の表を作り直す
        var redirects: [String: (path: String, offset: UInt64, rows: Int)] = [:]
        if !analysis.ropeTables.isEmpty {
            redirects = try rebuildRopeTables(analysis.ropeTables, modelDirectory: modelDirectory, rows: newContext)
        } else {
            try? fileManager.removeItem(at: modelDirectory.appendingPathComponent("weights").appendingPathComponent(ropeFileName))
        }

        let rewrittenMIL = try rewriteMIL(mil, analysis: analysis, newContext: newContext, ropeRedirects: redirects)
        let description = try Data(contentsOf: backup.appendingPathComponent("coremldata.bin"))
        let rewrittenDescription = try rewriteDescriptionFile(description, analysis: analysis, newContext: newContext)
        var rewrittenMetadata: Data?
        if let metadata = try? Data(contentsOf: backup.appendingPathComponent("metadata.json")) {
            rewrittenMetadata = try rewriteMetadata(metadata, analysis: analysis, newContext: newContext)
        }

        try Data(rewrittenMIL.text.utf8).write(to: modelDirectory.appendingPathComponent("model.mil"), options: .atomic)
        try rewrittenDescription.write(to: modelDirectory.appendingPathComponent("coremldata.bin"), options: .atomic)
        if let rewrittenMetadata {
            try rewrittenMetadata.write(to: modelDirectory.appendingPathComponent("metadata.json"), options: .atomic)
        }
        return analysis
    }

    /// 元のコンテキスト長に戻す
    static func restore(modelDirectory: URL, backup: URL) throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: backup.appendingPathComponent("model.mil").path) else { return }
        for name in backupFiles {
            let source = backup.appendingPathComponent(name)
            guard fileManager.fileExists(atPath: source.path) else { continue }
            let destination = modelDirectory.appendingPathComponent(name)
            try? fileManager.removeItem(at: destination)
            try fileManager.copyItem(at: source, to: destination)
        }
        try? fileManager.removeItem(at: modelDirectory.appendingPathComponent("weights").appendingPathComponent(ropeFileName))
    }

    /// 元の重みファイルから表を読み、新しい長さの表を weights/context_rope.bin に書く
    private static func rebuildRopeTables(_ tables: [RopeReference], modelDirectory: URL,
                                          rows: Int) throws -> [String: (path: String, offset: UInt64, rows: Int)] {
        // 関数ごとに別の名前で同じ表を指していることがあるので、元の位置ごとに 1 つだけ作る
        var blobs: [(type: MILBlob.DataType, values: [Float])] = []
        var blobIndex: [String: Int] = [:]
        for table in tables {
            let key = "\(table.path)#\(table.offset)"
            if blobIndex[key] != nil { continue }
            let relative = table.path.replacingOccurrences(of: "@model_path/", with: "")
            let url = modelDirectory.appendingPathComponent(relative)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let values = try MILBlob.readFloats(in: handle, at: table.offset)
            let fit = try RopeTable.fit(values, rows: table.rows, columns: table.columns)
            let type: MILBlob.DataType = table.dataType == "fp32" ? .float32 : .float16
            blobIndex[key] = blobs.count
            blobs.append((type, RopeTable.generate(fit, rows: rows)))
        }
        let weights = modelDirectory.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        let offsets = try MILBlob.write(blobs, to: weights.appendingPathComponent(ropeFileName))
        var redirects: [String: (path: String, offset: UInt64, rows: Int)] = [:]
        for table in tables {
            guard let index = blobIndex["\(table.path)#\(table.offset)"] else { continue }
            redirects[table.variable] = ("@model_path/weights/\(ropeFileName)", offsets[index], rows)
        }
        return redirects
    }
}
