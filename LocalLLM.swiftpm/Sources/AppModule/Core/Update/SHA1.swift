import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

/// SHA-1（git のファイルの識別子を計算するために使う。暗号の用途には使わない）。
/// Apple の端末では CryptoKit を使い（Swift Playgrounds は最適化なしでビルドするので、自前の計算は遅い）、
/// それ以外（Linux のテスト）では自前で計算する。
enum SHA1 {
    static func hash(_ data: Data) -> [UInt8] {
        #if canImport(CryptoKit)
        return Array(Insecure.SHA1.hash(data: data))
        #else
        return portableHash(data)
        #endif
    }

    static func hex(_ data: Data) -> String {
        hash(data).map { String(format: "%02x", $0) }.joined()
    }

    /// Foundation だけで計算する SHA-1
    static func portableHash(_ data: Data) -> [UInt8] {
        var state: [UInt32] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0]
        let message = padded(data)
        var words = [UInt32](repeating: 0, count: 80)
        var chunk = 0
        while chunk < message.count {
            for index in 0..<16 {
                words[index] = bigEndianWord(message, at: chunk + index * 4)
            }
            for index in 16..<80 {
                let mixed: UInt32 = words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16]
                words[index] = rotateLeft(mixed, by: 1)
            }
            compress(&state, words)
            chunk += 64
        }
        var output: [UInt8] = []
        output.reserveCapacity(20)
        for value in state {
            appendBigEndian(value, to: &output)
        }
        return output
    }

    /// 末尾に 1 ビットと 0 を足し、元の長さ（ビット数）を 64 ビットで付ける
    private static func padded(_ data: Data) -> [UInt8] {
        var message = [UInt8](data)
        let bitLength = UInt64(message.count) * 8
        message.append(0x80)
        while message.count % 64 != 56 { message.append(0) }
        for shift in stride(from: 56, through: 0, by: -8) {
            let byte: UInt64 = (bitLength >> UInt64(shift)) & 0xFF
            message.append(UInt8(byte))
        }
        return message
    }

    /// 64 バイトのかたまり 1 つ分の計算
    private static func compress(_ state: inout [UInt32], _ words: [UInt32]) {
        var a = state[0]
        var b = state[1]
        var c = state[2]
        var d = state[3]
        var e = state[4]
        for index in 0..<80 {
            let (f, k) = mix(index, b, c, d)
            var temp: UInt32 = rotateLeft(a, by: 5)
            temp = temp &+ f
            temp = temp &+ e
            temp = temp &+ k
            temp = temp &+ words[index]
            e = d
            d = c
            c = rotateLeft(b, by: 30)
            b = a
            a = temp
        }
        state[0] = state[0] &+ a
        state[1] = state[1] &+ b
        state[2] = state[2] &+ c
        state[3] = state[3] &+ d
        state[4] = state[4] &+ e
    }

    /// 回ごとの関数と定数
    private static func mix(_ index: Int, _ b: UInt32, _ c: UInt32, _ d: UInt32) -> (UInt32, UInt32) {
        if index < 20 {
            let choose: UInt32 = (b & c) | (~b & d)
            return (choose, 0x5A82_7999)
        }
        if index < 40 { return (b ^ c ^ d, 0x6ED9_EBA1) }
        if index < 60 {
            let majority: UInt32 = (b & c) | (b & d) | (c & d)
            return (majority, 0x8F1B_BCDC)
        }
        return (b ^ c ^ d, 0xCA62_C1D6)
    }

    private static func rotateLeft(_ value: UInt32, by amount: UInt32) -> UInt32 {
        (value << amount) | (value >> (32 - amount))
    }

    private static func bigEndianWord(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        var value = UInt32(bytes[offset]) << 24
        value |= UInt32(bytes[offset + 1]) << 16
        value |= UInt32(bytes[offset + 2]) << 8
        value |= UInt32(bytes[offset + 3])
        return value
    }

    private static func appendBigEndian(_ value: UInt32, to output: inout [UInt8]) {
        output.append(UInt8(truncatingIfNeeded: value >> 24))
        output.append(UInt8(truncatingIfNeeded: value >> 16))
        output.append(UInt8(truncatingIfNeeded: value >> 8))
        output.append(UInt8(truncatingIfNeeded: value))
    }
}

/// git がファイルの中身に付ける識別子（`git hash-object` と同じ）。GitHub のファイル一覧の sha と比べられる。
enum GitBlob {
    static func sha(of data: Data) -> String {
        var input = Data("blob \(data.count)\u{0}".utf8)
        input.append(data)
        return SHA1.hex(input)
    }
}
