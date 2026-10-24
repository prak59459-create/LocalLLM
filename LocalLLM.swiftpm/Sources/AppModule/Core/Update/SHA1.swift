import Foundation

/// SHA-1（git のファイルの識別子を計算するために使う。暗号の用途には使わない）。
/// Core は Foundation だけに依存するので、自前で実装している。
enum SHA1 {
    static func hash(_ data: Data) -> [UInt8] {
        var h0: UInt32 = 0x6745_2301
        var h1: UInt32 = 0xEFCD_AB89
        var h2: UInt32 = 0x98BA_DCFE
        var h3: UInt32 = 0x1032_5476
        var h4: UInt32 = 0xC3D2_E1F0

        // 末尾に 1 ビットと 0 を足し、元の長さ（ビット数）を 64 ビットで付ける
        var message = [UInt8](data)
        let bitLength = UInt64(message.count) * 8
        message.append(0x80)
        while message.count % 64 != 56 { message.append(0) }
        for shift in stride(from: 56, through: 0, by: -8) {
            message.append(UInt8((bitLength >> UInt64(shift)) & 0xFF))
        }

        var w = [UInt32](repeating: 0, count: 80)
        var chunk = 0
        while chunk < message.count {
            for index in 0..<16 {
                let base = chunk + index * 4
                w[index] = UInt32(message[base]) << 24 | UInt32(message[base + 1]) << 16
                    | UInt32(message[base + 2]) << 8 | UInt32(message[base + 3])
            }
            for index in 16..<80 {
                let value = w[index - 3] ^ w[index - 8] ^ w[index - 14] ^ w[index - 16]
                w[index] = (value << 1) | (value >> 31)
            }
            var a = h0, b = h1, c = h2, d = h3, e = h4
            for index in 0..<80 {
                let f: UInt32
                let k: UInt32
                switch index {
                case 0..<20:
                    f = (b & c) | (~b & d)
                    k = 0x5A82_7999
                case 20..<40:
                    f = b ^ c ^ d
                    k = 0x6ED9_EBA1
                case 40..<60:
                    f = (b & c) | (b & d) | (c & d)
                    k = 0x8F1B_BCDC
                default:
                    f = b ^ c ^ d
                    k = 0xCA62_C1D6
                }
                let rotated = (a << 5) | (a >> 27)
                let temp = rotated &+ f &+ e &+ k &+ w[index]
                e = d
                d = c
                c = (b << 30) | (b >> 2)
                b = a
                a = temp
            }
            h0 = h0 &+ a
            h1 = h1 &+ b
            h2 = h2 &+ c
            h3 = h3 &+ d
            h4 = h4 &+ e
            chunk += 64
        }
        var output: [UInt8] = []
        for value in [h0, h1, h2, h3, h4] {
            output.append(UInt8(value >> 24))
            output.append(UInt8((value >> 16) & 0xFF))
            output.append(UInt8((value >> 8) & 0xFF))
            output.append(UInt8(value & 0xFF))
        }
        return output
    }

    static func hex(_ data: Data) -> String {
        hash(data).map { String(format: "%02x", $0) }.joined()
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
