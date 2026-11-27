#!/usr/bin/env python3
"""アプリのビルド時間の内訳を出す（CI で使う）。

  python3 Tools/build_time_report.py STATS_DIR FUNCTION_TIMES [--top N]

STATS_DIR: swiftc の -stats-output-dir で書かれた JSON（コンパイラの 1 回ごとに 1 つ）
FUNCTION_TIMES: -Xfrontend -debug-time-function-bodies の出力（"12.34ms<TAB>file:line:col<TAB>名前"）

段階ごとの時間（全部のジョブの合計）、型チェックに時間のかかったファイルと関数を表示する。
"""
import collections
import glob
import json
import os
import re
import sys


def phases(stats_dir):
    """段階ごとの時間（秒）を、全部のジョブで足す"""
    totals = collections.Counter()
    jobs = 0
    for path in glob.glob(os.path.join(stats_dir, "*.json")):
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        # ドライバー自身の統計は除く（フロントエンドのジョブだけ）
        if not any(key.startswith("time.swift.") for key in data):
            continue
        jobs += 1
        for key, value in data.items():
            match = re.match(r"time\.swift\.(.+)\.wall$", key)
            if match and isinstance(value, (int, float)):
                totals[match.group(1)] += value
    return jobs, totals


def report_files(stats_dir, top):
    """ジョブがファイルごとのとき（バッチにしないとき）: ファイルごとの段階の時間"""
    rows = []
    for path in glob.glob(os.path.join(stats_dir, "*.json")):
        match = re.search(r"swift-frontend-[^-]+-(.+?\.swift)-", os.path.basename(path))
        if not match:
            continue
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        value = lambda key: data.get(f"time.swift.{key}.wall", 0.0)
        # ジョブ全体の時間（型チェック・コード生成・LLVM・書き出しの全部）
        total = data.get("Frontend.WallClockMicroseconds", 0) / 1e6
        rows.append((total, value("Type checking and Semantic analysis"), value("SILGen") + value("IRGen"),
                     data.get("IRModule.NumIRInsts", 0), match.group(1)))
    if not rows:
        return
    print(f"\n== ファイルごとのコンパイル時間（1 ファイルずつのとき。上位 {top}、秒）: 合計 {sum(r[0] for r in rows):.1f} 秒")
    print("  全体   型  SILGen+IRGen  IR命令  ファイル")
    for total, sema, generate, insts, name in sorted(rows, reverse=True)[:top]:
        print(f"{total:6.2f} {sema:5.2f} {generate:6.2f} {insts:8d}  {name}")


def functions(path):
    rows = []
    pattern = re.compile(r"\s*([\d.]+)ms\t([^\t:]+):(\d+):\d+\t(.*)")
    for line in open(path, errors="replace"):
        match = pattern.match(line)
        if not match:
            continue
        name = match.group(4).split("@")[0]
        source = match.group(2).split("AppModule/")[-1]
        rows.append((float(match.group(1)), source, int(match.group(3)), name))
    return rows


def main():
    stats_dir, times = sys.argv[1], sys.argv[2]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
    jobs, totals = phases(stats_dir)
    print(f"== 段階ごとの時間（{jobs} ジョブの合計、秒）")
    for name, seconds in totals.most_common(30):
        print(f"{seconds:9.2f}  {name}")

    if "--per-file" in sys.argv:
        report_files(stats_dir, top)
    rows = functions(times)
    total = sum(r[0] for r in rows)
    print(f"\n== 関数の本体の型チェック: {len(rows)} 個、合計 {total / 1000:.1f} 秒")
    buckets = collections.Counter()
    for ms, *_ in rows:
        key = "<5ms" if ms < 5 else "5-20ms" if ms < 20 else "20-50ms" if ms < 50 else ">=50ms"
        buckets[key] += ms
    print("  " + ", ".join(f"{k}: {v / 1000:.1f} 秒" for k, v in sorted(buckets.items())))
    per_file = collections.Counter()
    for ms, source, _, _ in rows:
        per_file[source] += ms
    print("\n== ファイルごと（上位 25、ミリ秒）")
    for source, ms in per_file.most_common(25):
        print(f"{ms:8.0f}  {source}")
    print(f"\n== 関数ごと（上位 {top}、ミリ秒）")
    for ms, source, line, name in sorted(rows, reverse=True)[:top]:
        print(f"{ms:8.1f}  {source}:{line}  {name[-90:]}")


if __name__ == "__main__":
    main()
