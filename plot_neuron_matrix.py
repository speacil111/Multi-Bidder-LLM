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
            "Plot neuron-count score matrix from CTR score CSV. "
            "Top half of each square is Brand-1 score; bottom half is Brand-2 score."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="./run_ctr_scores.csv",
        help="Input CSV path containing Uber/Starbucks score and neuron_count columns.",
    )
    parser.add_argument(
        "--output_png",
        default="double_neuron_plot/run_ctr_scores_matrix.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="CTR Score Matrix (Uber top / Starbucks bottom)",
        help="Figure title.",
    )
    parser.add_argument(
        "--x-label",
        default="Uber neuron count",
        help="X-axis label.",
    )
    parser.add_argument(
        "--y-label",
        default="Starbucks neuron count",
        help="Y-axis label.",
    )
    parser.add_argument(
        "--top-label",
        default="Uber score",
        help="Label for top half + top colorbar.",
    )
    parser.add_argument(
        "--bottom-label",
        default="Starbucks score",
        help="Label for bottom half + bottom colorbar.",
    )
    parser.add_argument(
        "--neuron_min",
        type=float,
        default=None,
        help=(
            "Optional lower bound for both brand neuron counts (inclusive)."
        ),
    )
    parser.add_argument(
        "--neuron_max",
        type=float,
        default=None,
        help=(
            "Optional upper bound for both brand neuron counts (inclusive)."
        ),
    )
    parser.add_argument(
        "--neuron_interval",
        type=int,
        default=50,
        help=(
            "Keep only rows where both neuron counts align to this interval. "
            "For example, 100 keeps 100/200/300... (anchored at min count in data)."
        ),
    )
    return parser.parse_args()


def _to_number(text):
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


def _get_value(row, keys):
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    raise KeyError(f"Missing keys {keys} in row: {list(row.keys())}")


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "brand1_score": float(
                        _get_value(
                            row,
                            [
                                "BMW_score",
                                "Uber_score",
                                "Uber_ctr_score",
                                "Nike_score",
                                "Delta_Score",
                                "delta_score",
                                "Delta_score",
                            ],
                        )
                    ),
                    "brand2_score": float(
                        _get_value(
                            row,
                            [
                                "Rolex_score",
                                "Starbucks_score",
                                "Starbucks_ctr_score",
                                "Spotify_score",
                                "Hilton_Score",
                                "hilton_score",
                                "Hilton_score",
                            ],
                        )
                    ),
                    "brand1_neuron_count": _to_number(
                        _get_value(
                            row,
                            [
                                "BMW_neuron_count",
                                "BMW_Auto_neuron_count",
                                "Uber_neuron_count",
                                "Uber_Rideshare_neuron_count",
                                "Nike_neuron_count",
                                "Nike_Sportswear_neuron_count",
                                "delta_neuron_count",
                                "Delta_neuron_count",
                                "Delta_neurons",
                            ],
                        )
                    ),
                    "brand2_neuron_count": _to_number(
                        _get_value(
                            row,
                            [
                                "Rolex_neuron_count",
                                "Rolex_Watch_neuron_count",
                                "Starbucks_neuron_count",
                                "Starbucks_Coffee_neuron_count",
                                "Spotify_neuron_count",
                                "Spotify_Music_neuron_count",
                                "hilton_neuron_count",
                                "Hilton_neuron_count",
                                "Hilton_neurons",
                            ],
                        )
                    ),
                }
            )
    return rows


def filter_rows_by_neuron_range(rows, neuron_min=None, neuron_max=None, neuron_interval=50):
    if neuron_interval <= 0:
        raise ValueError("neuron_interval must be a positive integer.")

    if neuron_min is None and neuron_max is None and neuron_interval == 50:
        return rows

    all_counts = [
        float(row["brand1_neuron_count"]) for row in rows
    ] + [
        float(row["brand2_neuron_count"]) for row in rows
    ]
    interval_anchor = min(all_counts) if all_counts else 0.0

    filtered = []
    for row in rows:
        b1_count = float(row["brand1_neuron_count"])
        b2_count = float(row["brand2_neuron_count"])

        if neuron_min is not None and (b1_count < neuron_min or b2_count < neuron_min):
            continue
        if neuron_max is not None and (b1_count > neuron_max or b2_count > neuron_max):
            continue

        if ((b1_count - interval_anchor) % neuron_interval != 0) or (
            (b2_count - interval_anchor) % neuron_interval != 0
        ):
            continue
        filtered.append(row)
    return filtered


def get_max_score(rows):
    if not rows:
        return 1.0
    return max(max(r["brand1_score"], r["brand2_score"]) for r in rows) or 1.0


def make_plot(
    rows,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    top_label: str,
    bottom_label: str,
):
    if not rows:
        raise ValueError("No rows loaded from CSV.")

    x_levels = sorted({r["brand1_neuron_count"] for r in rows})
    y_levels = sorted({r["brand2_neuron_count"] for r in rows})
    x_index = {v: i for i, v in enumerate(x_levels)}
    y_index = {v: i for i, v in enumerate(y_levels)}

    cell_map = {}
    for row in rows:
        key = (row["brand1_neuron_count"], row["brand2_neuron_count"])
        cell_map[key] = (row["brand1_score"], row["brand2_score"])

    max_score = get_max_score(rows)
    norm = colors.Normalize(vmin=0, vmax=max_score)
    brand1_cmap = colormaps["Blues"]
    brand2_cmap = colormaps["Reds"]
    border_color = "#444444"

    fig, ax = plt.subplots(figsize=(13, 11))

    for y_val in y_levels:
        for x_val in x_levels:
            x = x_index[x_val]
            y = y_index[y_val]
            brand1_score, brand2_score = cell_map.get((x_val, y_val), (0.0, 0.0))

            # Bottom half: brand-2
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1.0,
                    0.5,
                    facecolor=brand2_cmap(norm(brand2_score)),
                    edgecolor="none",
                )
            )
            # Top half: brand-1
            ax.add_patch(
                Rectangle(
                    (x, y + 0.5),
                    1.0,
                    0.5,
                    facecolor=brand1_cmap(norm(brand1_score)),
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
        Patch(facecolor="#c6dbef", edgecolor=border_color, label=f"Top half: {top_label}"),
        Patch(facecolor="#fcbba1", edgecolor=border_color, label=f"Bottom half: {bottom_label}"),
    ]
    ax.legend(handles=role_items, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(right=0.84)

    brand1_sm = cm.ScalarMappable(norm=norm, cmap=brand1_cmap)
    brand1_sm.set_array([])
    brand1_cax = fig.add_axes([0.87, 0.55, 0.025, 0.28])
    brand1_cbar = fig.colorbar(brand1_sm, cax=brand1_cax)
    brand1_cbar.set_label(top_label)

    brand2_sm = cm.ScalarMappable(norm=norm, cmap=brand2_cmap)
    brand2_sm.set_array([])
    brand2_cax = fig.add_axes([0.87, 0.16, 0.025, 0.28])
    brand2_cbar = fig.colorbar(brand2_sm, cax=brand2_cax)
    brand2_cbar.set_label(bottom_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_png = Path(args.output_png)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = load_rows(input_csv)
    rows = filter_rows_by_neuron_range(
        rows,
        neuron_min=args.neuron_min,
        neuron_max=args.neuron_max,
        neuron_interval=args.neuron_interval,
    )
    if not rows:
        raise ValueError(
            "No rows left after neuron range filtering. "
            "Please check --neuron_min/--neuron_max."
        )
    make_plot(
        rows,
        output_png,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        top_label=args.top_label,
        bottom_label=args.bottom_label,
    )
    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()
