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
            "Plot neuron-count score matrix from score_0.csv. "
            "Top half of each square is Delta score; bottom half is Hilton score."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="./newp6_m2.5_uni_new_cache/ad_eva_scores6.csv",
        help="Input CSV path containing Run/Hilton_Score/Delta_Score and neuron_count columns.",
    )
    parser.add_argument(
        "--output_png",
        default="double_neuron_plot/newp6_m2.5_uni_new_cache.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Prompt6 Score Matrix (Delta top / Hilton bottom)",
        help="Figure title.",
    )
    parser.add_argument(
        "--neuron_min",
        type=float,
        default=None,
        help=(
            "Optional lower bound for both hilton_neuron_count and "
            "delta_neuron_count (inclusive)."
        ),
    )
    parser.add_argument(
        "--neuron_max",
        type=float,
        default=None,
        help=(
            "Optional upper bound for both hilton_neuron_count and "
            "delta_neuron_count (inclusive)."
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
                    "hilton_score": float(_get_value(row, ["Hilton_Score", "hilton_score","Hilton_score"])),
                    "delta_score": float(_get_value(row, ["Delta_Score", "delta_score","Delta_score"])),
                    "hilton_neuron_count": _to_number(
                        _get_value(row, ["hilton_neuron_count", "Hilton_neuron_count","Hilton_neurons"])
                    ),
                    "delta_neuron_count": _to_number(
                        _get_value(row, ["delta_neuron_count", "Delta_neuron_count","Delta_neurons"])
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
        float(row["hilton_neuron_count"]) for row in rows
    ] + [
        float(row["delta_neuron_count"]) for row in rows
    ]
    interval_anchor = min(all_counts) if all_counts else 0.0

    filtered = []
    for row in rows:
        h_count = float(row["hilton_neuron_count"])
        d_count = float(row["delta_neuron_count"])

        if neuron_min is not None and (h_count < neuron_min or d_count < neuron_min):
            continue
        if neuron_max is not None and (h_count > neuron_max or d_count > neuron_max):
            continue

        if ((h_count - interval_anchor) % neuron_interval != 0) or (
            (d_count - interval_anchor) % neuron_interval != 0
        ):
            continue
        filtered.append(row)
    return filtered


def get_max_score(rows):
    if not rows:
        return 1.0
    return max(max(r["delta_score"], r["hilton_score"]) for r in rows) or 1.0


def make_plot(rows, output_path: Path, title: str):
    if not rows:
        raise ValueError("No rows loaded from CSV.")

    x_levels = sorted({r["delta_neuron_count"] for r in rows})
    y_levels = sorted({r["hilton_neuron_count"] for r in rows})
    x_index = {v: i for i, v in enumerate(x_levels)}
    y_index = {v: i for i, v in enumerate(y_levels)}

    cell_map = {}
    for row in rows:
        key = (row["delta_neuron_count"], row["hilton_neuron_count"])
        cell_map[key] = (row["delta_score"], row["hilton_score"])

    max_score = get_max_score(rows)
    norm = colors.Normalize(vmin=0, vmax=max_score)
    delta_cmap = colormaps["Blues"]
    hilton_cmap = colormaps["Reds"]
    border_color = "#444444"

    fig, ax = plt.subplots(figsize=(13, 11))

    for y_val in y_levels:
        for x_val in x_levels:
            x = x_index[x_val]
            y = y_index[y_val]
            delta_score, hilton_score = cell_map.get((x_val, y_val), (0.0, 0.0))

            # Bottom half: Hilton
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1.0,
                    0.5,
                    facecolor=hilton_cmap(norm(hilton_score)),
                    edgecolor="none",
                )
            )
            # Top half: Delta
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
    ax.set_xlabel("delta_neuron_count")
    ax.set_ylabel("hilton_neuron_count")
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
    make_plot(rows, output_png, title=args.title)
    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()
