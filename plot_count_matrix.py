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
            "Plot count matrix from summary_counts.tsv. "
            "Top half of each square shows Delta mention count, "
            "bottom half shows Hilton mention count."
        )
    )
    parser.add_argument(
        "--dir_path",
        default="double_500_[1-4]_output/",
        help="Directory containing summary_counts.tsv; output image is saved there.",
    )
    parser.add_argument(
        "--input_name",
        default="summary.tsv",
        help="Input TSV filename inside dir_path.",
    )
    parser.add_argument(
        "--output_name",
        default="hit_count_matrix.png",
        help="Output image filename inside dir_path.",
    )
    parser.add_argument(
        "--title",
        default="500_neuron_Hit Count Matrix (ig20)",
        help="Plot title.",
    )
    return parser.parse_args()


def load_rows(tsv_path: Path):
    rows = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "delta_multiplier": float(row["delta_multiplier"]),
                    "hilton_multiplier": float(row["hilton_multiplier"]),
                    "hit_delta": int(row["hit_delta"]),
                    "hit_hilton": int(row["hit_hilton"]),
                }
            )
    return rows


def get_max_count(rows):
    if not rows:
        return 1
    return max(
        max(row["hit_delta"], row["hit_hilton"])
        for row in rows
    ) or 1


def make_plot(rows, output_path: Path, title: str):
    if not rows:
        raise ValueError("No rows loaded from TSV.")

    delta_levels = sorted({r["delta_multiplier"] for r in rows})
    hilton_levels = sorted({r["hilton_multiplier"] for r in rows})

    delta_index = {v: i for i, v in enumerate(delta_levels)}
    hilton_index = {v: i for i, v in enumerate(hilton_levels)}

    cell_map = {}
    for row in rows:
        key = (row["delta_multiplier"], row["hilton_multiplier"])
        cell_map[key] = (row["hit_delta"], row["hit_hilton"])

    max_count = get_max_count(rows)
    norm = colors.Normalize(vmin=0, vmax=max_count)
    delta_cmap = colormaps["Blues"]
    hilton_cmap = colormaps["Reds"]
    border_color = "#444444"

    fig, ax = plt.subplots(figsize=(12, 10))

    for h in hilton_levels:
        for d in delta_levels:
            x = delta_index[d]
            y = hilton_index[h]
            hit_delta, hit_hilton = cell_map.get((d, h), (0, 0))

            ax.add_patch(
                Rectangle(
                    (x, y),
                    1.0,
                    0.5,
                    facecolor=hilton_cmap(norm(hit_hilton)),
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y + 0.5),
                    1.0,
                    0.5,
                    facecolor=delta_cmap(norm(hit_delta)),
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
        Patch(facecolor="#c6dbef", edgecolor=border_color, label="Top half: Delta count"),
        Patch(facecolor="#fcbba1", edgecolor=border_color, label="Bottom half: Hilton count"),
    ]
    ax.legend(handles=role_items, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.subplots_adjust(right=0.84)

    delta_sm = cm.ScalarMappable(norm=norm, cmap=delta_cmap)
    delta_sm.set_array([])
    delta_cax = fig.add_axes([0.87, 0.55, 0.025, 0.28])
    delta_cbar = fig.colorbar(delta_sm, cax=delta_cax)
    delta_cbar.set_label("Delta mention count")

    hilton_sm = cm.ScalarMappable(norm=norm, cmap=hilton_cmap)
    hilton_sm.set_array([])
    hilton_cax = fig.add_axes([0.87, 0.16, 0.025, 0.28])
    hilton_cbar = fig.colorbar(hilton_sm, cax=hilton_cax)
    hilton_cbar.set_label("Hilton mention count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    dir_path = Path(args.dir_path)
    input_path = dir_path / args.input_name
    output_path = dir_path / args.output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {input_path}")

    rows = load_rows(input_path)
    make_plot(rows, output_path, title=args.title)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
