import Foundation

/// 検索用に文章を語（ターム）へ分ける。
///
/// - 英数字: 小文字にする。2 文字以上の語と、数字だけの語（"1" など）を使う。
///   よく出る英単語（the, a, an, and, or, of, to, in, is, it, for, on, with）は捨てる。
/// - 識別子: camelCase / PascalCase / 略語（"URLSessionDataTask" → url, session, data, task）や
///   snake_case / kebab-case を部分語に分け、最後に元の語の小文字（"urlsessiondatatask"）も加える。
/// - 漢字・ひらがな・カタカナ（ー を含む）の連なりは、1 文字ずつずらした 2 文字の組（bigram）にする。
///   連なりが 1 文字だけならその 1 文字。助詞（は・が・の など）は前後の文字と bigram になるので、
///   単独の語としては出てこない（空白などで区切られて 1 文字だけになったときは、その 1 文字が語になる）。
/// - 全角英数字や半角カナは NFKC で正規化してから分ける。絵文字や記号は区切りとして無視する。
enum KnowledgeTokenizer {
    /// 検索語に分ける（文中の順番どおり。重複も残す）。
    static func terms(_ text: String) -> [String] {
        var output: [String] = []
        appendTerms(of: text, to: &output)
        return output
    }

    /// terms と同じ結果を output の末尾に追加する（インデックス作成で配列を使い回すため）。
    static func appendTerms(of text: String, to output: inout [String]) {
        let startCount = output.count
        var copy = text
        let completed = copy.withUTF8 { buffer -> Bool in
            guard let base = buffer.baseAddress else { return true }
            // ASCII だけなら正規化は不要。ASCII 以外が出てきたら NFKC をかけてやり直す。
            return KnowledgeTermScanner.scan(base, count: buffer.count, asciiOnly: true, into: &output)
        }
        if completed { return }
        output.removeSubrange(startCount..<output.count)
        var normalized = text.precomposedStringWithCompatibilityMapping
        normalized.withUTF8 { buffer in
            guard let base = buffer.baseAddress else { return }
            _ = KnowledgeTermScanner.scan(base, count: buffer.count, asciiOnly: false, into: &output)
        }
    }
}

/// UTF-8 のバイト列を直接読んで語を切り出す（最適化なしのビルドでも速いように、ポインタと整数演算だけで書く）。
fileprivate enum KnowledgeTermScanner {
    /// 部分語として使う最大バイト数（これより長い語は base64 などのゴミとみなして捨てる）
    static let maxTermBytes = 64

    static let stopwords: Set<String> = [
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with"
    ]

    /// asciiOnly が true のとき、ASCII 以外のバイトを見つけたら false を返して中断する。
    static func scan(_ p: UnsafePointer<UInt8>, count n: Int, asciiOnly: Bool,
                     into output: inout [String]) -> Bool {
        var i = 0
        while i < n {
            let b = p[i]
            if b < 0x80 {
                if isASCIIWordStart(b) {
                    let end = wordRunEnd(p, count: n, from: i, asciiOnly: asciiOnly)
                    if end < 0 { return false }
                    emitWordRun(p, from: i, to: end, into: &output)
                    i = end
                } else {
                    i += 1
                }
                continue
            }
            if asciiOnly { return false }
            let decoded = decodeScalar(p, count: n, at: i)
            if isCJK(decoded.value) {
                i = emitCJKRun(p, count: n, from: i, into: &output)
            } else if isOtherLetter(decoded.value) {
                let end = wordRunEnd(p, count: n, from: i, asciiOnly: false)
                emitWordRun(p, from: i, to: end, into: &output)
                i = end
            } else {
                i += decoded.length
            }
        }
        return true
    }

    // MARK: - 文字の種類

    @inline(__always)
    static func isASCIIAlnum(_ b: UInt8) -> Bool {
        if b >= 0x61 && b <= 0x7A { return true }   // a-z
        if b >= 0x41 && b <= 0x5A { return true }   // A-Z
        return b >= 0x30 && b <= 0x39               // 0-9
    }

    @inline(__always)
    static func isASCIIWordStart(_ b: UInt8) -> Bool {
        return isASCIIAlnum(b) || b == 0x5F          // _
    }

    /// 漢字・ひらがな・カタカナ（長音符 ー と踊り字を含む。中黒 ・ は含まない）
    static func isCJK(_ v: UInt32) -> Bool {
        if v < 0x3005 { return false }
        if v <= 0x3007 { return true }                       // 々 〆 〇
        if v >= 0x3041 && v <= 0x3096 { return true }        // ひらがな
        if v >= 0x309D && v <= 0x309F { return true }        // ゝ ゞ ゟ
        if v >= 0x30A1 && v <= 0x30FA { return true }        // カタカナ
        if v >= 0x30FC && v <= 0x30FF { return true }        // ー ヽ ヾ ヿ
        if v >= 0x31F0 && v <= 0x31FF { return true }        // カタカナ拡張
        if v >= 0x3400 && v <= 0x4DBF { return true }        // CJK 統合漢字拡張 A
        if v >= 0x4E00 && v <= 0x9FFF { return true }        // CJK 統合漢字
        if v >= 0xF900 && v <= 0xFAFF { return true }        // CJK 互換漢字
        return v >= 0x20000 && v <= 0x3134F                  // 拡張 B〜G
    }

    /// ASCII と CJK 以外の文字（é、ハングルなど）。語の一部として扱う。
    static func isOtherLetter(_ v: UInt32) -> Bool {
        guard let scalar = Unicode.Scalar(v) else { return false }
        return scalar.properties.isAlphabetic
    }

    @inline(__always)
    static func decodeScalar(_ p: UnsafePointer<UInt8>, count n: Int, at i: Int) -> (value: UInt32, length: Int) {
        let b0 = UInt32(p[i])
        if b0 < 0x80 { return (b0, 1) }
        if b0 < 0xE0 {
            if i + 1 >= n { return (0xFFFD, 1) }
            let v = ((b0 & 0x1F) << 6) | (UInt32(p[i + 1]) & 0x3F)
            return (v, 2)
        }
        if b0 < 0xF0 {
            if i + 2 >= n { return (0xFFFD, 1) }
            var v = (b0 & 0x0F) << 12
            v |= (UInt32(p[i + 1]) & 0x3F) << 6
            v |= UInt32(p[i + 2]) & 0x3F
            return (v, 3)
        }
        if i + 3 >= n { return (0xFFFD, 1) }
        var v = (b0 & 0x07) << 18
        v |= (UInt32(p[i + 1]) & 0x3F) << 12
        v |= (UInt32(p[i + 2]) & 0x3F) << 6
        v |= UInt32(p[i + 3]) & 0x3F
        return (v, 4)
    }

    // MARK: - 英数字の語

    /// 語の終わりの位置。asciiOnly で ASCII 以外に出会ったら -1。
    /// "-" は前後が ASCII 英数字のときだけ語の一部にする（kebab-case）。
    static func wordRunEnd(_ p: UnsafePointer<UInt8>, count n: Int, from start: Int, asciiOnly: Bool) -> Int {
        var i = start
        while i < n {
            let b = p[i]
            if b < 0x80 {
                if isASCIIWordStart(b) {
                    i += 1
                    continue
                }
                if b == 0x2D && i > start && i + 1 < n && isASCIIAlnum(p[i - 1]) && isASCIIAlnum(p[i + 1]) {
                    i += 1
                    continue
                }
                return i
            }
            if asciiOnly { return -1 }
            let decoded = decodeScalar(p, count: n, at: i)
            if isCJK(decoded.value) || !isOtherLetter(decoded.value) { return i }
            i += decoded.length
        }
        return i
    }

    /// 文字の種類: 0 区切り（_ -）, 1 大文字, 2 小文字, 3 数字, 4 ASCII 以外の文字（大文字・小文字の切れ目にしない）
    @inline(__always)
    static func wordClass(_ b: UInt8) -> Int {
        if b >= 0x61 && b <= 0x7A { return 2 }
        if b >= 0x41 && b <= 0x5A { return 1 }
        if b >= 0x30 && b <= 0x39 { return 3 }
        if b >= 0x80 { return 4 }
        return 0
    }

    /// 1 つの語を部分語に分けて出力し、部分語が 2 つ以上なら元の語（小文字）も出力する。
    static func emitWordRun(_ p: UnsafePointer<UInt8>, from start: Int, to end: Int, into output: inout [String]) {
        var partStart = -1
        var partChars = 0
        var partCount = 0
        var prev = 0
        var prevPrev = 0
        var prevPos = start
        var i = start
        while i < end {
            let b = p[i]
            var length = 1
            if b >= 0x80 {
                // 先頭バイトから UTF-8 の長さを求める（続きのバイトは飛ばす）
                if b >= 0xF0 { length = 4 } else if b >= 0xE0 { length = 3 } else { length = 2 }
            }
            let cls = wordClass(b)
            if cls == 0 {
                if partStart >= 0 {
                    emitPart(p, from: partStart, to: i, characters: partChars, into: &output)
                    partCount += 1
                    partStart = -1
                }
            } else if partStart < 0 {
                partStart = i
                partChars = 0
            } else if cls == 1 && (prev == 2 || prev == 3) {
                // fooBar / http2Server: 大文字の前で切る
                emitPart(p, from: partStart, to: i, characters: partChars, into: &output)
                partCount += 1
                partStart = i
                partChars = 0
            } else if cls == 2 && prev == 1 && prevPrev == 1 && prevPos > partStart {
                // URLSession: 最後の大文字（S）の前で切る
                emitPart(p, from: partStart, to: prevPos, characters: partChars - 1, into: &output)
                partCount += 1
                partStart = prevPos
                partChars = 1
            }
            if cls != 0 { partChars += 1 }
            prevPrev = prev
            prev = cls
            prevPos = i
            i += length
        }
        if partStart >= 0 {
            emitPart(p, from: partStart, to: end, characters: partChars, into: &output)
            partCount += 1
        }
        if partCount < 2 { return }
        // 元の語（先頭と末尾の _ - は除く）
        var wholeStart = start
        var wholeEnd = end
        while wholeStart < wholeEnd && wordClass(p[wholeStart]) == 0 { wholeStart += 1 }
        while wholeEnd > wholeStart && wordClass(p[wholeEnd - 1]) == 0 { wholeEnd -= 1 }
        if wholeEnd - wholeStart > maxTermBytes { return }
        output.append(lowercasedString(p, from: wholeStart, to: wholeEnd))
    }

    static func emitPart(_ p: UnsafePointer<UInt8>, from start: Int, to end: Int, characters: Int,
                         into output: inout [String]) {
        let length = end - start
        if length <= 0 || length > maxTermBytes { return }
        var allDigits = true
        var k = start
        while k < end {
            let b = p[k]
            if b < 0x30 || b > 0x39 {
                allDigits = false
                break
            }
            k += 1
        }
        if !allDigits && characters < 2 { return }
        let term = lowercasedString(p, from: start, to: end)
        if length <= 4 && stopwords.contains(term) { return }
        output.append(term)
    }

    /// バイト列を小文字の String にする（ASCII はその場で変換、それ以外は lowercased()）。
    static func lowercasedString(_ p: UnsafePointer<UInt8>, from start: Int, to end: Int) -> String {
        let length = end - start
        var hasNonASCII = false
        let result = String(unsafeUninitializedCapacity: length) { out -> Int in
            guard let dst = out.baseAddress else { return 0 }
            var k = 0
            while k < length {
                var b = p[start + k]
                if b >= 0x41 && b <= 0x5A {
                    b += 0x20
                } else if b >= 0x80 {
                    hasNonASCII = true
                }
                dst[k] = b
                k += 1
            }
            return length
        }
        return hasNonASCII ? result.lowercased() : result
    }

    // MARK: - 漢字・かな

    /// CJK の連なりを bigram にして出力し、連なりの終わりの位置を返す。
    static func emitCJKRun(_ p: UnsafePointer<UInt8>, count n: Int, from start: Int,
                           into output: inout [String]) -> Int {
        var previous = -1
        var i = start
        var characters = 0
        while i < n {
            if p[i] < 0x80 { break }
            let decoded = decodeScalar(p, count: n, at: i)
            if !isCJK(decoded.value) { break }
            let next = i + decoded.length
            if previous >= 0 {
                output.append(String(decoding: UnsafeBufferPointer(start: p + previous, count: next - previous),
                                     as: UTF8.self))
            }
            previous = i
            characters += 1
            i = next
        }
        if characters == 1 {
            output.append(String(decoding: UnsafeBufferPointer(start: p + start, count: i - start), as: UTF8.self))
        }
        return i
    }
}
