#!/usr/bin/env python3
"""GGUF の IQ2 / IQ3 量子化で使う表（グリッド）を、gguf パッケージから取り出して Swift のソースにする。

    pip install gguf numpy
    python Tools/gguf/make_quant_tables.py > LocalLLM.swiftpm/Sources/AppModule/Core/GGUF/GGUFQuantTables.swift
"""
import base64

import numpy as np
from gguf import quants

GRIDS = [
    ("iq2xxs", quants.IQ2_XXS),
    ("iq2xs", quants.IQ2_XS),
    ("iq2s", quants.IQ2_S),
    ("iq3xxs", quants.IQ3_XXS),
    ("iq3s", quants.IQ3_S),
]


def chunks(text, size=120):
    return [text[i:i + size] for i in range(0, len(text), size)]


print("import Foundation")
print()
print("// このファイルは Tools/gguf/make_quant_tables.py が作る（手で書き換えない）。")
print()
print("/// IQ2 / IQ3 量子化の表（llama.cpp の iq2xxs_grid などと同じ内容。値は 0〜255 の整数）。")
print("enum GGUFQuantTables {")
for name, cls in GRIDS:
    cls.init_grid()
    grid = cls.grid.reshape(cls.grid_shape)
    assert np.all(grid == np.round(grid)) and grid.min() >= 0 and grid.max() <= 255
    encoded = base64.b64encode(grid.astype(np.uint8).tobytes()).decode()
    print(f"    /// {grid.shape[0]} 行 × {grid.shape[1]} 個")
    print(f"    static let {name}: [Float] = decode([")
    for part in chunks(encoded):
        print(f'        "{part}",')
    print("    ])")
    print()
signs = base64.b64encode(bytes(quants.IQ2_XXS.ksigns)).decode()
print("    /// 7 ビットの番号 → 8 個分の符号（ビットが 1 なら負）")
print("    static let signs: [UInt8] = bytes([")
for part in chunks(signs):
    print(f'        "{part}",')
print("    ])")
print()
print("    private static func bytes(_ parts: [String]) -> [UInt8] {")
print("        [UInt8](Data(base64Encoded: parts.joined()) ?? Data())")
print("    }")
print()
print("    private static func decode(_ parts: [String]) -> [Float] {")
print("        bytes(parts).map { Float($0) }")
print("    }")
print("}")
