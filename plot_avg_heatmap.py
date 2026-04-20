#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm, colormaps, colors
from matplotlib.patches import Rectangle

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="拆分绘制热力图，可选扣除 Baseline (0,0) 的值。"
    )
    parser.add_argument("--csv", type=Path, default=None, help="输入 avg_csv 路径。")
    parser.add_argument("--base_dir", type=Path, default=None, help="包含 summary_avg*.csv 的目录。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./new_fair_mind_plot/",
        help="输出目录。",
    )
    parser.add_argument("--x-col", default=None, help="X 轴列名。")
    parser.add_argument("--y-col", default=None, help="Y 轴列名。")
    parser.add_argument("--top-col", default=None, help="品牌A hit 列名。")
    parser.add_argument("--bottom-col", default=None, help="品牌B hit 列名。")
    parser.add_argument("--dpi", type=int, default=220, help="输出 DPI。")
    parser.add_argument(
        "--sub",
        action="store_true",
        help="是否扣除 baseline 值；开启后输出文件名会追加 _sub 后缀。",
    )
    parser.add_argument("--baseline-x", type=float, default=0.0, help="Baseline 的 x 坐标，默认 0。")
    parser.add_argument("--baseline-y", type=float, default=0.0, help="Baseline 的 y 坐标，默认 0。")
    
    args = parser.parse_args()
    if args.csv is None and args.base_dir is None:
        parser.error("请提供 --csv 或 --base_dir。")
    return args

def _fmt_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)

def _fmt_cell(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")

def _resolve_csv_path(csv_path: Path | None, base_dir: Path | None) -> Path:
    if csv_path is not None: return csv_path
    assert base_dir is not None
    matches = sorted(base_dir.glob("summar*y_avg*.csv"))
    if not matches: raise FileNotFoundError("未找到 CSV 文件")
    for p in matches:
        if "p1_p5" in p.name: return p
    return matches[0]

def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), list(reader.fieldnames) if reader.fieldnames else []

def draw_single_heatmap(
    data_map: dict[tuple[float, float], float],
    x_vals: list[float],
    y_vals: list[float],
    col_name: str,
    title: str,
    cmap_name: str,
    output_path: Path,
    dpi: int,
    x_label: str,
    y_label: str,
    sub_enabled: bool,
):
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}
    
    # 过滤掉 0 后的最大值，用于设置色标上限
    vals = list(data_map.values())
    max_val = max(vals) if vals else 1.0
    
    # 使用 LogNorm，由于已经 0 截断，我们将 vmin 设为 0.1 以避免 log(0)
    norm = colors.LogNorm(vmin=0.1, vmax=max(1.1, max_val))
    cmap = colormaps[cmap_name]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f0f0f0")

    for (x, y), val in data_map.items():
        xi, yi = x_idx[x], y_idx[y]
        # 加上一个极小值避免 log(0) 绘图异常
        face_color = cmap(norm(max(0.1, val)))
        
        rect = Rectangle((xi, yi), 1.0, 1.0, facecolor=face_color, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        
        # 数值标注
        ax.text(xi + 0.5, yi + 0.5, _fmt_cell(val), ha="center", va="center", 
                fontsize=9, color="black" if val < (max_val * 0.5) else "white")

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, len(y_vals))
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(len(x_vals))])
    ax.set_xticklabels([_fmt_tick(v) for v in x_vals])
    ax.set_yticks([i + 0.5 for i in range(len(y_vals))])
    ax.set_yticklabels([_fmt_tick(v) for v in y_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if sub_enabled:
        ax.set_title(f"{title}\n(Subtracted Baseline & Zero-Clipped)")
    else:
        ax.set_title(title)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Value (Log Scale)")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output_path}")

def main() -> None:
    args = parse_args()
    csv_path = _resolve_csv_path(args.csv, args.base_dir)
    rows, fieldnames = load_rows(csv_path)
    
    x_col = args.x_col or fieldnames[1]
    y_col = args.y_col or fieldnames[2]
    top_col = args.top_col or fieldnames[-2]
    bottom_col = args.bottom_col or fieldnames[-1]

    x_vals = sorted({float(row[x_col]) for row in rows})
    y_vals = sorted({float(row[y_col]) for row in rows})

    base_top, base_bottom = 0.0, 0.0
    found_baseline = False
    if args.sub:
        for row in rows:
            if (
                abs(float(row[x_col]) - args.baseline_x) < 1e-7
                and abs(float(row[y_col]) - args.baseline_y) < 1e-7
            ):
                base_top = float(row[top_col])
                base_bottom = float(row[bottom_col])
                found_baseline = True
                break

        if not found_baseline:
            print(f"警告: 未找到坐标为 ({args.baseline_x}, {args.baseline_y}) 的 Baseline，将使用 0 作为基准。")

    top_data = {}
    bottom_data = {}
    for row in rows:
        coord = (float(row[x_col]), float(row[y_col]))
        if args.sub:
            top_data[coord] = max(0.0, float(row[top_col]) - base_top)
            bottom_data[coord] = max(0.0, float(row[bottom_col]) - base_bottom)
        else:
            top_data[coord] = float(row[top_col])
            bottom_data[coord] = float(row[bottom_col])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = "_sub" if args.sub else ""

    draw_single_heatmap(
        top_data, x_vals, y_vals, top_col, 
        f"Brand A: {top_col}", "Blues", 
        output_dir / f"blue_{top_col}{file_suffix}.png", 
        args.dpi, x_col, y_col, args.sub
    )

    draw_single_heatmap(
        bottom_data, x_vals, y_vals, bottom_col, 
        f"Brand B: {bottom_col}", "Reds", 
        output_dir / f"red_{bottom_col}{file_suffix}.png", 
        args.dpi, x_col, y_col, args.sub
    )

if __name__ == "__main__":
    main()
