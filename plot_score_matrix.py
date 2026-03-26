#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm, colormaps, colors
from matplotlib.patches import Patch, Rectangle


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot score matrix from ad_counts.csv + summary.tsv. "
            "Top half of each square shows Delta score, "
            "bottom half shows Hilton score."
        )
    )
    parser.add_argument(
        "--summary_tsv",
        default="double_500_[1-4]_output/summary.tsv",
        help="TSV with run_id -> multipliers mapping.",
    )
    parser.add_argument(
        "--score_csv",
        default="ad_counts_seq.csv",
        help="CSV containing Run_ID/Run and Hilton_Score/Delta_Score columns.",
    )
    parser.add_argument(
        "--output_path",
        default="double_500_[1-4]_output/score_matrix.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="500_neuron Score Matrix (Delta/Hilton)",
        help="Plot title.",
    )
    parser.add_argument(
        "--x-key",
        default="delta_multiplier",
        help="X-axis key in summary TSV (e.g., delta_multiplier or delta_neuron_count).",
    )
    parser.add_argument(
        "--y-key",
        default="hilton_multiplier",
        help="Y-axis key in summary TSV (e.g., hilton_multiplier or hilton_neuron_count).",
    )
    parser.add_argument(
        "--x-label",
        default="Delta multiplier",
        help="X-axis label.",
    )
    parser.add_argument(
        "--y-label",
        default="Hilton multiplier",
        help="Y-axis label.",
    )
    return parser.parse_args()


def _get_value(row, keys):
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    raise KeyError(f"Missing keys {keys} in row: {list(row.keys())}")


def _parse_number(text: str):
    value = float(text)
    if value.is_integer():
        return int(value)
    return value


def _format_tick(value):
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}"


def load_summary(summary_tsv: Path, x_key: str, y_key: str):
    run_to_xy = {}
    with summary_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id = int(_get_value(row, ["run_id", "Run_ID", "run", "Run"]))
            run_to_xy[run_id] = (
                _parse_number(_get_value(row, [x_key])),
                _parse_number(_get_value(row, [y_key])),
            )
    return run_to_xy


def load_scores(score_csv: Path):
    run_to_score = {}
    with score_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = int(_get_value(row, ["Run_ID", "Run", "run_id", "run"]))
            delta_score = float(_get_value(row, ["Delta_Score", "delta_score"]))
            hilton_score = float(_get_value(row, ["Hilton_Score", "hilton_score"]))
            run_to_score[run_id] = (delta_score, hilton_score)
    return run_to_score


def build_rows(run_to_xy, run_to_score):
    rows = []
    for run_id, (x_value, y_value) in run_to_xy.items():
        delta_score, hilton_score = run_to_score.get(run_id, (0, 0))
        rows.append(
            {
                "x_value": x_value,
                "y_value": y_value,
                "delta_score": delta_score,
                "hilton_score": hilton_score,
            }
        )
    return rows


def get_max_score(rows):
    if not rows:
        return 1
    return max(max(row["delta_score"], row["hilton_score"]) for row in rows) or 1


def make_plot(rows, output_path: Path, title: str, x_label: str, y_label: str):
    if not rows:
        raise ValueError("No rows loaded for plotting.")

    x_levels = sorted({r["x_value"] for r in rows})
    y_levels = sorted({r["y_value"] for r in rows})
    x_index = {v: i for i, v in enumerate(x_levels)}
    y_index = {v: i for i, v in enumerate(y_levels)}

    cell_map = {}
    for row in rows:
        key = (row["x_value"], row["y_value"])
        cell_map[key] = (row["delta_score"], row["hilton_score"])

    max_score = get_max_score(rows)
    norm = colors.Normalize(vmin=0, vmax=max_score)
    delta_cmap = colormaps["Blues"]
    hilton_cmap = colormaps["Reds"]
    border_color = "#444444"

    fig, ax = plt.subplots(figsize=(12, 10))

    for y_val in y_levels:
        for x_val in x_levels:
            x = x_index[x_val]
            y = y_index[y_val]
            delta_score, hilton_score = cell_map.get((x_val, y_val), (0, 0))

            ax.add_patch(
                Rectangle(
                    (x, y),
                    1.0,
                    0.5,
                    facecolor=hilton_cmap(norm(hilton_score)),
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y + 0.5),
                    1.0,
                    0.5,
                    facecolor=delta_cmap(norm(delta_score)),
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=border_color,
                    linewidth=1.0,
                )
            )

    ax.set_xlim(0, len(x_levels))
    ax.set_ylim(0, len(y_levels))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([i + 0.5 for i in range(len(x_levels))])
    ax.set_xticklabels([_format_tick(v) for v in x_levels], rotation=45, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(y_levels))])
    ax.set_yticklabels([_format_tick(v) for v in y_levels])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)

    role_items = [
        Patch(facecolor="#c6dbef", edgecolor=border_color, label="Top half: Delta score"),
        Patch(facecolor="#fcbba1", edgecolor=border_color, label="Bottom half: Hilton score"),
    ]
    ax.legend(handles=role_items, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(right=0.84)

    delta_sm = cm.ScalarMappable(norm=norm, cmap=delta_cmap)
    delta_sm.set_array([])
    delta_cax = fig.add_axes([0.87, 0.55, 0.025, 0.28])
    delta_cbar = fig.colorbar(delta_sm, cax=delta_cax)
    delta_cbar.set_label("Delta score")

    hilton_sm = cm.ScalarMappable(norm=norm, cmap=hilton_cmap)
    hilton_sm.set_array([])
    hilton_cax = fig.add_axes([0.87, 0.16, 0.025, 0.28])
    hilton_cbar = fig.colorbar(hilton_sm, cax=hilton_cax)
    hilton_cbar.set_label("Hilton score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    summary_tsv = Path(args.summary_tsv)
    score_csv = Path(args.score_csv)
    output_path = Path(args.output_path)

    if not summary_tsv.exists():
        raise FileNotFoundError(f"Summary TSV not found: {summary_tsv}")
    if not score_csv.exists():
        raise FileNotFoundError(f"Score CSV not found: {score_csv}")

    run_to_xy = load_summary(summary_tsv, x_key=args.x_key, y_key=args.y_key)
    run_to_score = load_scores(score_csv)
    rows = build_rows(run_to_xy, run_to_score)
    make_plot(
        rows,
        output_path=output_path,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
