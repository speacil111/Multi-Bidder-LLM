#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List
#  用法,例
# python average_summary_csv.py --base-dir fair_mind_Toyota_m2.0

def build_input_paths_from_base(base_dir: Path) -> List[Path]:
    return [base_dir / f"p{i}" / f"summary_{i}.csv" for i in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按行聚合多个 summary CSV：保留前N列，后续列按多个文件逐行取平均。"
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        help="输入 CSV 文件路径列表（例如 p1/summary_1.csv ... p5/summary_5.csv）。",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="品牌目录路径（例如 fair_mind_Uber_m2.0），会自动读取 p1~p5 的 summary_i.csv。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        # default="./new_fair_mind_plot/",
        help="输出 CSV 路径。默认输出到 base-dir/summary_avg_p1_p5.csv 或第一个输入文件目录。",
    )
    parser.add_argument(
        "--fixed-cols",
        type=int,
        default=5,
        help="前多少列保持不变，默认 5。",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="平均值保留小数位数，默认 2。",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="CSV 分隔符，默认制表符。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.base_dir is None and not args.inputs:
        raise ValueError("请提供 --base-dir 或 --inputs。")
    if args.base_dir is not None and args.inputs:
        raise ValueError("--base-dir 与 --inputs 二选一，不要同时传。")

    if args.base_dir is not None:
        input_paths = build_input_paths_from_base(args.base_dir)
    else:
        input_paths = args.inputs

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在: {p}")

    rows_by_file = []
    for p in input_paths:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows_by_file.append(list(csv.DictReader(f, delimiter=args.delimiter)))

    if not rows_by_file or not rows_by_file[0]:
        raise ValueError("输入文件为空或无数据行。")

    row_count = len(rows_by_file[0])
    if any(len(rows) != row_count for rows in rows_by_file):
        raise ValueError("输入文件行数不一致，无法逐行平均。")

    fieldnames = list(rows_by_file[0][0].keys())
    if len(fieldnames) <= args.fixed_cols:
        raise ValueError("列数不足，无法找到需要平均的列。")

    fixed_cols = fieldnames[: args.fixed_cols]
    avg_cols = fieldnames[args.fixed_cols :]

    output_rows = []
    for row_idx in range(row_count):
        base_row = rows_by_file[0][row_idx]

        # 前 fixed-cols 列要求所有文件同一行完全一致
        for file_idx in range(1, len(rows_by_file)):
            check_row = rows_by_file[file_idx][row_idx]
            for col in fixed_cols:
                if str(check_row[col]) != str(base_row[col]):
                    raise ValueError(f"第 {row_idx + 1} 行列 {col} 在不同文件不一致。")

        merged = {col: base_row[col] for col in fixed_cols}
        for col in avg_cols:
            avg = sum(float(rows_by_file[file_idx][row_idx][col]) for file_idx in range(len(rows_by_file))) / len(rows_by_file)
            merged[col] = f"{avg:.{args.decimals}f}"
        output_rows.append(merged)

    if args.output is not None:
        output_path = args.output
    elif args.base_dir is not None:
        output_path = args.base_dir / "summary_avg_p1_p5.csv"
    else:
        output_path = input_paths[0].parent / "summary_avg.csv"

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=args.delimiter,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"写入完成: {output_path}")
    print(f"输入文件数: {len(input_paths)}")
    print(f"数据行数: {len(output_rows)}")
    print(f"平均列: {', '.join(avg_cols)}")


if __name__ == "__main__":
    main()
