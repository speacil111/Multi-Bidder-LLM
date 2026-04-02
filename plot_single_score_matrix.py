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
            "读取 scored CSV，并为每个 score_* 列分别绘制单色矩阵图。"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="attr_sum_Uber_Ridesharep3_m2.25_rep_1.2/summary_3_scored.csv",
        help="输入 CSV/TSV 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="double_neuron_plot",
        help="输出目录。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="输出图片 DPI。",
    )
    return parser.parse_args()


def _pick_cmap(score_col: str, idx: int):
    lowered = score_col.lower()
    if "toyota" in lowered:
        return plt.cm.Blues
    if "costco" in lowered:
        return plt.cm.Reds
    palettes = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]
    return palettes[idx % len(palettes)]


def _to_float(row: Dict[str, str], key: str) -> float:
    if key not in row:
        raise KeyError(f"缺少列: {key}")
    return float(row[key])


def _detect_columns(fieldnames: List[str]) -> Dict[str, List[str] | str]:
    attr_sum_cols = [c for c in fieldnames if c.endswith("_attr_sum")]
    score_cols = [c for c in fieldnames if c.startswith("score_")]
    if len(attr_sum_cols) < 2:
        raise ValueError(f"需要至少两个 *_attr_sum 列，找到: {attr_sum_cols}")
    if len(score_cols) < 2:
        raise ValueError(f"需要至少两个 score_* 列，找到: {score_cols}")
    return {
        "x_col": attr_sum_cols[0],
        "y_col": attr_sum_cols[1],
        "score_cols": score_cols,
    }


def read_rows(input_csv: Path) -> Tuple[List[Dict[str, float]], Dict[str, List[str] | str]]:
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
            row = {
                "x": _to_float(raw, str(col_map["x_col"])),
                "y": _to_float(raw, str(col_map["y_col"])),
            }
            for score_col in col_map["score_cols"]:
                row[score_col] = _to_float(raw, score_col)
            rows.append(row)
    if not rows:
        raise ValueError("输入文件没有有效数据行。")
    return rows, col_map


def build_value_map(
    rows: List[Dict[str, float]],
    score_col: str,
) -> Tuple[List[float], List[float], Dict[Tuple[float, float], float]]:
    xs = sorted({r["x"] for r in rows})
    ys = sorted({r["y"] for r in rows})

    grouped: Dict[Tuple[float, float], List[float]] = {}
    for r in rows:
        key = (r["x"], r["y"])
        grouped.setdefault(key, []).append(r[score_col])

    value_map: Dict[Tuple[float, float], float] = {}
    for key, values in grouped.items():
        value_map[key] = sum(values) / len(values)

    return xs, ys, value_map


def plot_single_matrix(
    xs: List[float],
    ys: List[float],
    value_map: Dict[Tuple[float, float], float],
    x_label: str,
    y_label: str,
    score_label: str,
    output_path: Path,
    color_map,
    dpi: int,
) -> None:
    norm = Normalize(vmin=0, vmax=5)
    cmap = color_map if color_map is not None else plt.cm.Blues
    fig, ax = plt.subplots(figsize=(10, 8))

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            key = (x, y)
            if key in value_map:
                color = cmap(norm(value_map[key]))
            else:
                color = (0.9, 0.9, 0.9, 1.0)

            cell = mpatches.Rectangle(
                (xi, yi),
                1,
                1,
                facecolor=color,
                edgecolor="0.35",
                linewidth=0.8,
            )
            ax.add_patch(cell)

    ax.set_xlim(0, len(xs))
    ax.set_ylim(0, len(ys))
    ax.set_xticks([i + 0.5 for i in range(len(xs))], [f"{v:g}" for v in xs], rotation=45)
    ax.set_yticks([i + 0.5 for i in range(len(ys))], [f"{v:g}" for v in ys])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Attr Sum Score Matrix ({score_label})")

    plt.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.89, 0.14, 0.02, 0.72])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(score_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    rows, col_map = read_rows(input_csv)
    x_label = str(col_map["x_col"])
    y_label = str(col_map["y_col"])

    for idx, score_col in enumerate(col_map["score_cols"]):
        xs, ys, value_map = build_value_map(rows, score_col)
        suffix = score_col.replace("score_", "").lower()
        output_path = output_dir / f"{suffix}_attr_sum_matrix_single.png"
        cmap = _pick_cmap(score_col, idx)
        plot_single_matrix(
            xs=xs,
            ys=ys,
            value_map=value_map,
            x_label=x_label,
            y_label=y_label,
            score_label=score_col,
            output_path=output_path,
            color_map=cmap,
            dpi=args.dpi,
        )
        print(f"绘图完成: {output_path}")


if __name__ == "__main__":
    main()
