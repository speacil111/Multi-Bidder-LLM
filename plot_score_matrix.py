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
    return parser.parse_args()


def _get_value(row, keys):
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    raise KeyError(f"Missing keys {keys} in row: {list(row.keys())}")


def load_summary(summary_tsv: Path):
    run_to_mult = {}
    with summary_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id = int(_get_value(row, ["run_id", "Run_ID", "run", "Run"]))
            run_to_mult[run_id] = (
                float(_get_value(row, ["delta_multiplier", "Delta_Multiplier"])),
                float(_get_value(row, ["hilton_multiplier", "Hilton_Multiplier"])),
            )
    return run_to_mult


def load_scores(score_csv: Path):
    run_to_score = {}
    with score_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = int(_get_value(row, ["Run_ID", "Run", "run_id", "run"]))
            delta_score = int(float(_get_value(row, ["Delta_Score", "delta_score"])))
            hilton_score = int(float(_get_value(row, ["Hilton_Score", "hilton_score"])))
            run_to_score[run_id] = (delta_score, hilton_score)
    return run_to_score


def build_rows(run_to_mult, run_to_score):
    rows = []
    for run_id, (delta_mult, hilton_mult) in run_to_mult.items():
        delta_score, hilton_score = run_to_score.get(run_id, (0, 0))
        rows.append(
            {
                "delta_multiplier": delta_mult,
                "hilton_multiplier": hilton_mult,
                "delta_score": delta_score,
                "hilton_score": hilton_score,
            }
        )
    return rows


def get_max_score(rows):
    if not rows:
        return 1
    return max(max(row["delta_score"], row["hilton_score"]) for row in rows) or 1


def make_plot(rows, output_path: Path, title: str):
    if not rows:
        raise ValueError("No rows loaded for plotting.")

    delta_levels = sorted({r["delta_multiplier"] for r in rows})
    hilton_levels = sorted({r["hilton_multiplier"] for r in rows})
    delta_index = {v: i for i, v in enumerate(delta_levels)}
    hilton_index = {v: i for i, v in enumerate(hilton_levels)}

    cell_map = {}
    for row in rows:
        key = (row["delta_multiplier"], row["hilton_multiplier"])
        cell_map[key] = (row["delta_score"], row["hilton_score"])

    max_score = get_max_score(rows)
    norm = colors.Normalize(vmin=0, vmax=max_score)
    delta_cmap = colormaps["Blues"]
    hilton_cmap = colormaps["Reds"]
    border_color = "#444444"

    fig, ax = plt.subplots(figsize=(12, 10))

    for h in hilton_levels:
        for d in delta_levels:
            x = delta_index[d]
            y = hilton_index[h]
            delta_score, hilton_score = cell_map.get((d, h), (0, 0))

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

    ax.set_xlim(0, len(delta_levels))
    ax.set_ylim(0, len(hilton_levels))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([i + 0.5 for i in range(len(delta_levels))])
    ax.set_xticklabels([f"{v:.2f}" for v in delta_levels], rotation=45, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(hilton_levels))])
    ax.set_yticklabels([f"{v:.2f}" for v in hilton_levels])
    ax.set_xlabel("Delta multiplier")
    ax.set_ylabel("Hilton multiplier")
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

    run_to_mult = load_summary(summary_tsv)
    run_to_score = load_scores(score_csv)
    rows = build_rows(run_to_mult, run_to_score)
    make_plot(rows, output_path=output_path, title=args.title)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
