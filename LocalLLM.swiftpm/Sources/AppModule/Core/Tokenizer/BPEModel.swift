import Foundation

/// BPE（Byte Pair Encoding）モデル。tokenizer.json の "model" 部分。
final class TokBPEModel {
    /// 語彙: トークン（バイト列）→ ID
    let vocab: [TokByteKey: Int]
    /// ID → トークン文字列
    let idToToken: [Int: String]
    /// 結合規則: (左ID, 右ID) → (順位, 結合後のID)
    private let merges: [UInt64: (rank: Int32, id: Int32)]
    private let unkID: Int?
    private let fuseUnk: Bool
    private let byteFallback: Bool
    private let ignoreMerges: Bool
    private let continuingSubwordPrefix: String?
    private let endOfWordSuffix: String?
    /// "<0x00>"〜"<0xFF>" の ID（byte_fallback 用）
    private let byteTokenIDs: [Int?]
    /// 1 文字 → ID のキャッシュ（よく使うので）
    private var scalarCache: [UInt32: Int] = [:]
    /// 単語 → トークン列のキャッシュ
    private var wordCache: [TokByteKey: [Int]] = [:]
    private let wordCacheLimit = 20_000
    private let lock = NSLock()

    /// - Parameter vocabEntries: TokVocabScanner で元のバイト列から読んだ語彙（あればこちらを優先）
    init(json: [String: Any], vocabEntries: [([UInt8], Int)]?) throws {
        var vocab: [TokByteKey: Int] = [:]
        var idToToken: [Int: String] = [:]
        if let vocabEntries {
            vocab.reserveCapacity(vocabEntries.count)
            idToToken.reserveCapacity(vocabEntries.count)
            for (bytes, id) in vocabEntries {
                vocab[TokByteKey(bytes: bytes)] = id
                idToToken[id] = String(decoding: bytes, as: UTF8.self)
            }
        } else {
            // 予備: JSONSerialization の辞書から（正規等価なキーが重なると一部失われる）
            guard let vocabObject = json["vocab"] as? [String: Any] else {
                throw TokenizerError.invalidFormat(L("model.vocab がありません", "model.vocab is missing"))
            }
            for (token, value) in vocabObject {
                guard let id = (value as? Int) ?? (value as? NSNumber)?.intValue else { continue }
                vocab[TokByteKey(token)] = id
                idToToken[id] = token
            }
        }
        self.vocab = vocab
        self.idToToken = idToToken

        var merges: [UInt64: (rank: Int32, id: Int32)] = [:]
        let mergeList = json["merges"] as? [Any] ?? []
        merges.reserveCapacity(mergeList.count)
        for (rank, entry) in mergeList.enumerated() {
            let left: String
            let right: String
            if let text = entry as? String {
                // "a b" 形式（最初の空白で分ける）
                guard let space = text.unicodeScalars.firstIndex(of: " ") else { continue }
                left = String(text.unicodeScalars[..<space])
                right = String(text.unicodeScalars[text.unicodeScalars.index(after: space)...])
            } else if let pair = entry as? [Any], pair.count == 2, let a = pair[0] as? String, let b = pair[1] as? String {
                left = a
                right = b
            } else {
                continue
            }
            guard let leftID = vocab[TokByteKey(left)], let rightID = vocab[TokByteKey(right)] else { continue }
            // continuing_subword_prefix がある場合、右側の接頭辞は結合後に付かない
            var mergedText = left + right
            if let prefix = json["continuing_subword_prefix"] as? String, !prefix.isEmpty,
               TokText.hasPrefix(right, prefix) {
                mergedText = left + String(right.unicodeScalars.dropFirst(prefix.unicodeScalars.count))
            }
            guard let mergedID = vocab[TokByteKey(mergedText)] else { continue }
            let key = UInt64(UInt32(leftID)) << 32 | UInt64(UInt32(rightID))
            if merges[key] == nil {
                merges[key] = (Int32(rank), Int32(mergedID))
            }
        }
        self.merges = merges

        if let unk = json["unk_token"] as? String {
            unkID = vocab[TokByteKey(unk)]
        } else {
            unkID = nil
        }
        fuseUnk = json["fuse_unk"] as? Bool ?? false
        byteFallback = json["byte_fallback"] as? Bool ?? false
        ignoreMerges = json["ignore_merges"] as? Bool ?? false
        let prefix = json["continuing_subword_prefix"] as? String
        continuingSubwordPrefix = (prefix?.isEmpty ?? true) ? nil : prefix
        let suffix = json["end_of_word_suffix"] as? String
        endOfWordSuffix = (suffix?.isEmpty ?? true) ? nil : suffix
        byteTokenIDs = (0..<256).map { byte in
            vocab[TokByteKey(String(format: "<0x%02X>", byte))]
        }
    }

    var maxID: Int { idToToken.keys.max() ?? -1 }

    /// 1 つの断片（pre-tokenize 後の単語）をトークン ID 列にする。
    func tokenize(_ word: String) -> [Int] {
        if word.isEmpty { return [] }
        let key = TokByteKey(word)
        if ignoreMerges, let id = vocab[key] { return [id] }
        lock.lock()
        if let cached = wordCache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let result = mergeWord(word)
        lock.lock()
        if wordCache.count >= wordCacheLimit { wordCache.removeAll(keepingCapacity: true) }
        wordCache[key] = result
        lock.unlock()
        return result
    }

    private func id(forScalar scalar: Unicode.Scalar) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        if let cached = scalarCache[scalar.value] { return cached < 0 ? nil : cached }
        let found = vocab[TokByteKey(String(scalar))]
        scalarCache[scalar.value] = found ?? -1
        return found
    }

    /// 文字ごとに分けてから、順位の高い結合規則を繰り返し適用する（Hugging Face と同じ手順）。
    private func mergeWord(_ word: String) -> [Int] {
        let scalars = Array(word.unicodeScalars)
        var symbols: [Int] = []
        symbols.reserveCapacity(scalars.count)
        var pendingUnk = false

        func flushUnk() {
            if pendingUnk, let unkID { symbols.append(unkID) }
            pendingUnk = false
        }

        for (index, scalar) in scalars.enumerated() {
            let isFirst = index == 0
            let isLast = index == scalars.count - 1
            var found: Int?
            var piece = String(scalar)
            if (!isFirst && continuingSubwordPrefix != nil) || (isLast && endOfWordSuffix != nil) {
                if !isFirst, let prefix = continuingSubwordPrefix { piece = prefix + piece }
                if isLast, let suffix = endOfWordSuffix { piece += suffix }
                found = vocab[TokByteKey(piece)]
            } else {
                found = id(forScalar: scalar)
            }
            if let found {
                flushUnk()
                symbols.append(found)
                continue
            }
            if byteFallback {
                let bytes = Array(piece.utf8)
                let ids = bytes.compactMap { byteTokenIDs[Int($0)] }
                if ids.count == bytes.count {
                    flushUnk()
                    symbols.append(contentsOf: ids)
                    continue
                }
            }
            if let unkID {
                if pendingUnk && !fuseUnk {
                    symbols.append(unkID)
                }
                pendingUnk = true
            }
        }
        flushUnk()
        return applyMerges(symbols)
    }

    /// 優先度付きキュー（ヒープ）で結合していく。長い入力でも O(n log n)。
    private func applyMerges(_ initial: [Int]) -> [Int] {
        let count = initial.count
        if count < 2 { return initial }
        var ids = initial
        var prev = [Int](repeating: -1, count: count)
        var next = [Int](repeating: -1, count: count)
        var alive = [Bool](repeating: true, count: count)
        for index in 0..<count {
            prev[index] = index - 1
            next[index] = index + 1 < count ? index + 1 : -1
        }
        var heap = TokMergeHeap()
        for index in 0..<(count - 1) {
            if let merge = merges[pairKey(ids[index], ids[index + 1])] {
                heap.push(TokMergeCandidate(rank: merge.rank, position: Int32(index), newID: merge.id))
            }
        }
        while let top = heap.pop() {
            let position = Int(top.position)
            guard alive[position] else { continue }
            let right = next[position]
            guard right >= 0 else { continue }
            guard let current = merges[pairKey(ids[position], ids[right])], current.id == top.newID else { continue }
            // 結合する
            ids[position] = Int(top.newID)
            alive[right] = false
            let afterRight = next[right]
            next[position] = afterRight
            if afterRight >= 0 { prev[afterRight] = position }
            // 新しくできた隣どうしの組を追加
            let before = prev[position]
            if before >= 0, let merge = merges[pairKey(ids[before], ids[position])] {
                heap.push(TokMergeCandidate(rank: merge.rank, position: Int32(before), newID: merge.id))
            }
            if afterRight >= 0, let merge = merges[pairKey(ids[position], ids[afterRight])] {
                heap.push(TokMergeCandidate(rank: merge.rank, position: Int32(position), newID: merge.id))
            }
        }
        var result: [Int] = []
        result.reserveCapacity(count)
        var index = 0
        while index >= 0 {
            result.append(ids[index])
            index = next[index]
        }
        return result
    }

    @inline(__always)
    private func pairKey(_ left: Int, _ right: Int) -> UInt64 {
        UInt64(UInt32(truncatingIfNeeded: left)) << 32 | UInt64(UInt32(truncatingIfNeeded: right))
    }
}

/// 結合の候補（順位が小さいほど先、同じなら左が先）
struct TokMergeCandidate {
    var rank: Int32
    var position: Int32
    var newID: Int32

    @inline(__always)
    func precedes(_ other: TokMergeCandidate) -> Bool {
        rank != other.rank ? rank < other.rank : position < other.position
    }
}

/// 最小ヒープ
struct TokMergeHeap {
    private var items: [TokMergeCandidate] = []

    mutating func push(_ item: TokMergeCandidate) {
        items.append(item)
        var child = items.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            if items[child].precedes(items[parent]) {
                items.swapAt(child, parent)
                child = parent
            } else {
                break
            }
        }
    }

    mutating func pop() -> TokMergeCandidate? {
        guard !items.isEmpty else { return nil }
        let top = items[0]
        let last = items.removeLast()
        if !items.isEmpty {
            items[0] = last
            var parent = 0
            while true {
                let left = parent * 2 + 1
                let right = left + 1
                var best = parent
                if left < items.count && items[left].precedes(items[best]) { best = left }
                if right < items.count && items[right].precedes(items[best]) { best = right }
                if best == parent { break }
                items.swapAt(parent, best)
                parent = best
            }
        }
        return top
    }
}
