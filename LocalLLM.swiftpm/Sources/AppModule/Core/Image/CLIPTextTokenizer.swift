import Foundation

/// 画像生成（Stable Diffusion）のプロンプトをトークンにする: CLIP のトークナイザ。
///
/// モデルに付いてくる vocab.json と merges.txt から、Hugging Face の CLIP のトークナイザ（tokenizer.json）と
/// 同じ設定を組み立てて、このアプリのトークナイザで動かす。
struct CLIPTextTokenizer {
    private let tokenizer: HFTokenizer
    let startID: Int
    let endID: Int
    /// モデルが受け取る長さ
    static let sequenceLength = 77

    init(vocabJSON: Data, mergesText: String) throws {
        guard let vocab = try JSONSerialization.jsonObject(with: vocabJSON) as? [String: Any] else {
            throw TokenizerError.invalidFormat("CLIP vocab.json")
        }
        var merges: [String] = []
        for line in mergesText.split(whereSeparator: \.isNewline) {
            let text = line.trimmingCharacters(in: .whitespaces)
            if text.isEmpty || text.hasPrefix("#version") { continue }
            merges.append(text)
        }
        guard let start = (vocab["<|startoftext|>"] as? NSNumber)?.intValue,
              let end = (vocab["<|endoftext|>"] as? NSNumber)?.intValue else {
            throw TokenizerError.invalidFormat("CLIP special tokens")
        }
        startID = start
        endID = end
        let root: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": [
                ["id": start, "content": "<|startoftext|>", "single_word": false, "lstrip": false, "rstrip": false,
                 "normalized": true, "special": true],
                ["id": end, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false,
                 "normalized": false, "special": true],
            ],
            "normalizer": ["type": "Sequence", "normalizers": [
                ["type": "NFC"],
                ["type": "Replace", "pattern": ["Regex": #"\s+"#], "content": " "],
                ["type": "Lowercase"],
            ]],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": #"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#],
                 "behavior": "Removed", "invert": true],
                ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true],
            ]],
            "post_processor": NSNull(),
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": "<|endoftext|>", "continuing_subword_prefix": "",
                "end_of_word_suffix": "</w>", "fuse_unk": false, "vocab": vocab, "merges": merges,
            ] as [String: Any],
        ]
        tokenizer = try HFTokenizer(tokenizerJSON: try JSONSerialization.data(withJSONObject: root), tokenizerConfigJSON: nil)
    }

    /// プロンプトのトークン（始まり・終わりの印は含まない）
    func tokens(_ prompt: String) -> [Int] {
        tokenizer.encode(prompt, addSpecialTokens: false)
    }

    /// モデルに渡す決まった長さの列: 始まり + プロンプト（長すぎれば切る）+ 終わり + 埋め草。
    /// - Parameter padID: 埋め草のトークン（nil なら終わりの印で埋める。Stable Diffusion 2 は 0 で埋める）
    func encode(_ prompt: String, length: Int = CLIPTextTokenizer.sequenceLength, padID: Int? = nil) -> [Int] {
        var ids = [startID]
        ids.append(contentsOf: tokens(prompt).prefix(max(0, length - 2)))
        ids.append(endID)
        if ids.count < length { ids.append(contentsOf: repeatElement(padID ?? endID, count: length - ids.count)) }
        return ids
    }

    /// プロンプトが長すぎて切られるか
    func isTruncated(_ prompt: String, length: Int = CLIPTextTokenizer.sequenceLength) -> Bool {
        tokens(prompt).count > length - 2
    }
}
