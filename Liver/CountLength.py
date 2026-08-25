#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from pathlib import Path

# 为了兼容中文和常见本地编码，这里按顺序尝试多种编码
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "big5", "cp936", "latin-1"]

def read_text_best_effort(path: Path) -> str:
    last_err = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception as e:
            last_err = e
            continue
    # 兜底：忽略错误读取，尽量不因个别异常字符中断流程
    return path.read_text(encoding="utf-8", errors="ignore")

def count_chars(text: str, include_newlines: bool = False) -> int:
    if include_newlines:
        return len(text)
    # 默认不计换行符（\n 和 \r），更符合“内容长度”的直观统计
    return len(text.replace("\n", "").replace("\r", ""))

def main():
    parser = argparse.ArgumentParser(
        description="统计文件夹下所有 .txt 文件的内容长度，并保存为 CSV。"
    )
    parser.add_argument("folder", help="待统计的文件夹路径")
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归遍历子文件夹（默认只统计当前文件夹）"
    )
    parser.add_argument(
        "--include-newlines", action="store_true",
        help="计数时包含换行符（默认不包含）"
    )
    parser.add_argument(
        "-o", "--output", default="txt_length_stats.csv",
        help="输出 CSV 文件名（默认：txt_length_stats.csv）"
    )
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"路径不存在或不是文件夹: {root}")

    pattern = "**/*.txt" if args.recursive else "*.txt"
    files = sorted(root.glob(pattern))

    rows = []
    for f in files:
        try:
            text = read_text_best_effort(f)
            length = count_chars(text, include_newlines=args.include_newlines)
            # 只写相对路径，便于查看
            rel = f.relative_to(root).as_posix()
            rows.append((rel, length))
        except Exception as e:
            # 忽略单个文件的错误，继续处理其他文件
            rows.append((f"[读取失败] {f.name}", 0))

    # 将结果保存为 UTF-8 带 BOM，方便 Excel 打开显示中文
    out_path = (root / args.output)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "长度"])  # 表头：文件名 + 长度
        writer.writerows(rows)

    print(f"共统计 {len(rows)} 个 .txt 文件。结果已保存：{out_path}")

if __name__ == "__main__":
    main()
