import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 summary_3_scored.csv 绘制为双半格热图。"
            "自动检测 *_attr_sum 和 score_* 列名。"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="attr_sum_Toyota_Autop3_m2.25_rep_1.2/summary_3_scored.csv",
        help="输入 CSV/TSV 文件路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="double_neuron_plot/toyota_costco_attr_sum_matrix.png",
        help="输出图片路径。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="输出图片 DPI。",
    )
    return parser.parse_args()


def _to_float(row: Dict[str, str], key: str) -> float:
    if key not in row:
        raise KeyError(f"缺少列: {key}")
    return float(row[key])


def _detect_columns(fieldnames: List[str]) -> Dict[str, str]:
    attr_sum_cols = [c for c in fieldnames if c.endswith("_attr_sum")]
    score_cols = [c for c in fieldnames if c.startswith("score_")]
    if len(attr_sum_cols) < 2:
        raise ValueError(f"需要至少两个 *_attr_sum 列，找到: {attr_sum_cols}")
    if len(score_cols) < 2:
        raise ValueError(f"需要至少两个 score_* 列，找到: {score_cols}")
    return {
        "x_col": attr_sum_cols[0],
        "y_col": attr_sum_cols[1],
        "score1_col": score_cols[0],
        "score2_col": score_cols[1],
    }


def read_rows(input_csv: Path) -> Tuple[List[Dict[str, float]], Dict[str, str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        col_map = _detect_columns(list(reader.fieldnames or []))
        rows: List[Dict[str, float]] = []
        for raw in reader:
            rows.append(
                {
                    "x": _to_float(raw, col_map["x_col"]),
                    "y": _to_float(raw, col_map["y_col"]),
                    "score1": _to_float(raw, col_map["score1_col"]),
                    "score2": _to_float(raw, col_map["score2_col"]),
                }
            )
    if not rows:
        raise ValueError("输入文件没有有效数据行。")
    return rows, col_map


def build_value_map(
    rows: List[Dict[str, float]],
) -> Tuple[List[float], List[float], Dict[Tuple[float, float], Tuple[float, float]]]:
    xs = sorted({r["x"] for r in rows})
    ys = sorted({r["y"] for r in rows})

    grouped: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
    for r in rows:
        key = (r["x"], r["y"])
        grouped.setdefault(key, []).append((r["score1"], r["score2"]))

    value_map: Dict[Tuple[float, float], Tuple[float, float]] = {}
    for key, values in grouped.items():
        avg1 = sum(v[0] for v in values) / len(values)
        avg2 = sum(v[1] for v in values) / len(values)
        value_map[key] = (avg1, avg2)

    return xs, ys, value_map


def plot_split_half_matrix(
    xs: List[float],
    ys: List[float],
    value_map: Dict[Tuple[float, float], Tuple[float, float]],
    col_map: Dict[str, str],
    output_path: Path,
    dpi: int,
) -> None:
    x_label = col_map["x_col"]
    y_label = col_map["y_col"]
    score1_label = col_map["score1_col"]
    score2_label = col_map["score2_col"]

    norm1 = Normalize(vmin=0, vmax=5)
    norm2 = Normalize(vmin=0, vmax=5)
    cmap1 = plt.cm.Blues
    cmap2 = plt.cm.Reds

    fig, ax = plt.subplots(figsize=(12, 9))

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            key = (x, y)
            if key in value_map:
                v1, v2 = value_map[key]
                top_color = cmap1(norm1(v1))
                bottom_color = cmap2(norm2(v2))
            else:
                top_color = (0.9, 0.9, 0.9, 1.0)
                bottom_color = (0.9, 0.9, 0.9, 1.0)

            top_half = mpatches.Rectangle(
                (xi, yi + 0.5), 1, 0.5,
                facecolor=top_color, edgecolor="none",
            )
            bottom_half = mpatches.Rectangle(
                (xi, yi), 1, 0.5,
                facecolor=bottom_color, edgecolor="none",
            )
            ax.add_patch(top_half)
            ax.add_patch(bottom_half)

            border = mpatches.Rectangle(
                (xi, yi), 1, 1,
                fill=False, edgecolor="0.35", linewidth=0.8,
            )
            ax.add_patch(border)

    ax.set_xlim(0, len(xs))
    ax.set_ylim(0, len(ys))
    ax.set_xticks([i + 0.5 for i in range(len(xs))], [f"{v:g}" for v in xs], rotation=45)
    ax.set_yticks([i + 0.5 for i in range(len(ys))], [f"{v:g}" for v in ys])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Attr Sum Score Matrix (Top: {score1_label} / Bottom: {score2_label})")

    legend_items = [
        mpatches.Patch(facecolor="lightsteelblue", edgecolor="0.35",
                       label=f"Top half: {score1_label}"),
        mpatches.Patch(facecolor="mistyrose", edgecolor="0.35",
                       label=f"Bottom half: {score2_label}"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, framealpha=0.85)

    plt.subplots_adjust(right=0.83)
    cax1 = fig.add_axes([0.87, 0.42, 0.02, 0.25])
    cax2 = fig.add_axes([0.87, 0.10, 0.02, 0.25])
    cbar1 = fig.colorbar(ScalarMappable(norm=norm1, cmap=cmap1), cax=cax1)
    cbar2 = fig.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), cax=cax2)
    cbar1.set_label(score1_label)
    cbar2.set_label(score2_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_path = Path(args.output)

    rows, col_map = read_rows(input_csv)
    xs, ys, value_map = build_value_map(rows)
    plot_split_half_matrix(xs, ys, value_map, col_map, output_path=output_path, dpi=args.dpi)

    print(f"绘图完成: {output_path}")


if __name__ == "__main__":
    main()
