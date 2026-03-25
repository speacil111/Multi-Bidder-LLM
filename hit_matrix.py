#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot matrix for Delta/Hilton hit results. "
            "Top half of each square is Delta hit, bottom half is Hilton hit."
        )
    )
    parser.add_argument(
        "--dir_path",
        default="double_500_ig20_output/",
        help="Directory containing summary.tsv; output image is saved there.",
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


def make_plot(rows, output_path: Path):
    delta_levels = sorted({r["delta_multiplier"] for r in rows})
    hilton_levels = sorted({r["hilton_multiplier"] for r in rows})

    delta_index = {v: i for i, v in enumerate(delta_levels)}
    hilton_index = {v: i for i, v in enumerate(hilton_levels)}

    cell_map = {}
    for r in rows:
        key = (r["delta_multiplier"], r["hilton_multiplier"])
        cell_map[key] = (r["hit_delta"], r["hit_hilton"])

    fig, ax = plt.subplots(figsize=(10, 10))

    hit_color = "#1b9e77"
    miss_color = "#d9d9d9"
    border_color = "#444444"

    for h in hilton_levels:
        for d in delta_levels:
            x = delta_index[d]
            y = hilton_index[h]
            hit_delta, hit_hilton = cell_map.get((d, h), (0, 0))

            # Bottom half: Hilton
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1.0,
                    0.5,
                    facecolor=hit_color if hit_hilton else miss_color,
                    edgecolor="none",
                )
            )
            # Top half: Delta
            ax.add_patch(
                Rectangle(
                    (x, y + 0.5),
                    1.0,
                    0.5,
                    facecolor=hit_color if hit_delta else miss_color,
                    edgecolor="none",
                )
            )
            # Cell border
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
    ax.set_title("500_neuron_Hit Matrix (ig20)")

    ax.grid(False)

    legend_items = [
        Patch(facecolor=hit_color, edgecolor=border_color, label="Hit (1)"),
        Patch(facecolor=miss_color, edgecolor=border_color, label="Miss (0)"),
    ]
    role_items = [
        Patch(facecolor="white", edgecolor=border_color, label="Top half: Delta"),
        Patch(facecolor="white", edgecolor=border_color, label="Bottom half: Hilton"),
    ]
    first_legend = ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(first_legend)
    ax.legend(handles=role_items, loc="upper left", bbox_to_anchor=(1.02, 0.78))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    dir_path = Path(args.dir_path)
    input_path = dir_path / "summary.tsv"
    output_path = dir_path / "hit_matrix.png"

    if not input_path.exists():
        raise FileNotFoundError(f"summary.tsv not found: {input_path}")

    rows = load_rows(input_path)
    make_plot(rows, output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
