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
        description=(
            "根据 summary_avg 类 CSV 绘制双半格热力图：上半格为品牌A hit，下半格为品牌B hit。"
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="输入 avg_csv 路径。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./fair_mind_plot_sub/",
        help="输出目录（默认自动保存到输入 csv 所在目录）。",
    )
    parser.add_argument("--x-col", default=None, help="X 轴列名（默认自动识别第2列）。")
    parser.add_argument("--y-col", default=None, help="Y 轴列名（默认自动识别第3列）。")
    parser.add_argument("--top-col", default=None, help="上半格 hit 列名（默认自动识别倒数第2列）。")
    parser.add_argument("--bottom-col", default=None, help="下半格 hit 列名（默认自动识别倒数第1列）。")
    parser.add_argument("--title", default=None, help="图标题。")
    parser.add_argument("--dpi", type=int, default=220, help="输出 DPI，默认 220。")
    parser.add_argument(
        "--sub",
        action="store_true",
        help="是否从所有 hit 值中减去 baseline 点（默认 baseline 为 x=0,y=0）。",
    )
    parser.add_argument("--baseline-x", type=float, default=0.0, help="baseline 的 x 值，默认 0。")
    parser.add_argument("--baseline-y", type=float, default=0.0, help="baseline 的 y 值，默认 0。")
    return parser.parse_args()


def _fmt_tick(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _fmt_cell(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_first_brand(x_col: str, top_col: str) -> str:
    if x_col.endswith("_top_k"):
        return x_col[: -len("_top_k")]
    if top_col.startswith("hit_"):
        return top_col[len("hit_") :]
    return x_col.split("_")[0]


def _find_baseline_key(
    cell_map: dict[tuple[float, float], tuple[float, float]],
    baseline_x: float,
    baseline_y: float,
) -> tuple[float, float]:
    for key in cell_map:
        if abs(key[0] - baseline_x) < 1e-9 and abs(key[1] - baseline_y) < 1e-9:
            return key
    raise ValueError(f"找不到 baseline 点: ({baseline_x}, {baseline_y})")


def _auto_pick_columns(fieldnames: list[str], x_col: str | None, y_col: str | None, top_col: str | None, bottom_col: str | None):
    if len(fieldnames) < 7:
        raise ValueError(f"列数过少，无法自动识别：{fieldnames}")

    final_x = x_col or fieldnames[1]
    final_y = y_col or fieldnames[2]
    final_top = top_col or fieldnames[-2]
    final_bottom = bottom_col or fieldnames[-1]

    for col in (final_x, final_y, final_top, final_bottom):
        if col not in fieldnames:
            raise ValueError(f"列不存在: {col}")
    return final_x, final_y, final_top, final_bottom


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"文件缺少表头: {path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"文件无数据行: {path}")
    return rows, fieldnames


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.csv}")

    rows, fieldnames = load_rows(args.csv)
    x_col, y_col, top_col, bottom_col = _auto_pick_columns(
        fieldnames, args.x_col, args.y_col, args.top_col, args.bottom_col
    )

    x_vals = sorted({float(row[x_col]) for row in rows})
    y_vals = sorted({float(row[y_col]) for row in rows})
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    cell_map: dict[tuple[float, float], tuple[float, float]] = {}
    for row in rows:
        key = (float(row[x_col]), float(row[y_col]))
        cell_map[key] = (float(row[top_col]), float(row[bottom_col]))

    if args.sub:
        baseline_key = _find_baseline_key(cell_map, args.baseline_x, args.baseline_y)
        baseline_top, baseline_bottom = cell_map[baseline_key]
        # 使用 log 色标时需要非负值，这里将扣 baseline 后的负值截断到 0。
        cell_map = {
            key: (
                max(0.0, value[0] - baseline_top),
                max(0.0, value[1] - baseline_bottom),
            )
            for key, value in cell_map.items()
        }

    max_top = max(v[0] for v in cell_map.values())
    max_bottom = max(v[1] for v in cell_map.values())

    top_norm = colors.LogNorm(vmin=1, vmax=max(1.0, max_top + 1))
    bottom_norm = colors.LogNorm(vmin=1, vmax=max(1.0, max_bottom + 1))
    top_cmap = colormaps["Blues"]
    bottom_cmap = colormaps["Reds"]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#d9d9d9")

    for y in y_vals:
        for x in x_vals:
            xi = x_idx[x]
            yi = y_idx[y]

            if (x, y) not in cell_map:
                ax.add_patch(
                    Rectangle((xi, yi), 1.0, 1.0, facecolor="#d9d9d9", edgecolor="white", linewidth=2)
                )
                continue

            top_v, bottom_v = cell_map[(x, y)]
            ax.add_patch(
                Rectangle((xi, yi + 0.5), 1.0, 0.5, facecolor=top_cmap(top_norm(top_v + 1)), edgecolor="none")
            )
            ax.add_patch(
                Rectangle((xi, yi), 1.0, 0.5, facecolor=bottom_cmap(bottom_norm(bottom_v + 1)), edgecolor="none")
            )
            ax.add_patch(
                Rectangle((xi, yi), 1.0, 1.0, fill=False, edgecolor="white", linewidth=2)
            )
            ax.text(xi + 0.5, yi + 0.75, _fmt_cell(top_v), ha="center", va="center", fontsize=7)
            ax.text(xi + 0.5, yi + 0.25, _fmt_cell(bottom_v), ha="center", va="center", fontsize=7)

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, len(y_vals))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([i + 0.5 for i in range(len(x_vals))])
    ax.set_xticklabels([_fmt_tick(v) for v in x_vals])
    ax.set_yticks([i + 0.5 for i in range(len(y_vals))])
    ax.set_yticklabels([_fmt_tick(v) for v in y_vals])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    default_title = f"Average Hit Counts Heatmap | {top_col} (top) / {bottom_col} (bottom)"
    ax.set_title(args.title or default_title)

    fig.subplots_adjust(right=0.88)

    top_sm = cm.ScalarMappable(norm=top_norm, cmap=top_cmap)
    top_sm.set_array([])
    top_cax = fig.add_axes([0.90, 0.56, 0.02, 0.35])
    top_cbar = fig.colorbar(top_sm, cax=top_cax)
    top_cbar.set_label(f"Avg {top_col} (Log Scale)")

    bottom_sm = cm.ScalarMappable(norm=bottom_norm, cmap=bottom_cmap)
    bottom_sm.set_array([])
    bottom_cax = fig.add_axes([0.90, 0.14, 0.02, 0.35])
    bottom_cbar = fig.colorbar(bottom_sm, cax=bottom_cax)
    bottom_cbar.set_label(f"Avg {bottom_col} (Log Scale)")

    first_brand = _extract_first_brand(x_col, top_col)
    output_dir = args.output_dir or args.csv.parent
    output_path = output_dir / f"{first_brand}_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
