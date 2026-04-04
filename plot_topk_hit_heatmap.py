#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm, colormaps, colors
from matplotlib.patches import Rectangle

PROMPT_INDEX=8
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a dual-half hit-count heatmap from summary CSV/TSV. "
            "Top half uses blue colormap, bottom half uses red colormap."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(f"more_hilton_mind_Deltap{PROMPT_INDEX}_m2.0/summary_{PROMPT_INDEX}.csv"),
        help="Input summary file path (CSV/TSV).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(f"more-hilton-heatmap/hit_heatmap_{PROMPT_INDEX}_m_2.0.png"),
        help="Output image path.",
    )
    parser.add_argument("--x-col", default="Delta_top_k", help="X-axis column name.")
    parser.add_argument("--y-col", default="Hilton_top_k", help="Y-axis column name.")
    parser.add_argument(
        "--top-col", default="hit_Delta", help="Top-half (blue) count column name."
    )
    parser.add_argument(
        "--bottom-col", default="hit_Hilton", help="Bottom-half (red) count column name."
    )
    parser.add_argument(
        "--top-label", default="Delta hits", help="Top-half colorbar label."
    )
    parser.add_argument(
        "--bottom-label", default="Hilton hits", help="Bottom-half colorbar label."
    )
    parser.add_argument(
        "--title",
        default="Hit Counts Heatmap (Top: Delta [Blue], Bottom: Hilton [Red])",
        help="Plot title.",
    )
    return parser.parse_args()


def _format_tick(v):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(fv)) if fv.is_integer() else str(fv)


def _format_cell(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    # sep=None lets pandas auto-detect comma/tab delimiters.
    return pd.read_csv(csv_path, sep=None, engine="python")


def build_cell_map(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    top_col: str,
    bottom_col: str,
):
    required = [x_col, y_col, top_col, bottom_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    x_vals = sorted(df[x_col].dropna().unique())
    y_vals = sorted(df[y_col].dropna().unique())
    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}

    cell_map = {}
    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        top_val = float(row[top_col])
        bottom_val = float(row[bottom_col])
        cell_map[(x, y)] = (top_val, bottom_val)

    return x_vals, y_vals, x_index, y_index, cell_map


def make_plot(
    df: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    top_col: str,
    bottom_col: str,
    top_label: str,
    bottom_label: str,
    title: str,
) -> None:
    x_vals, y_vals, x_index, y_index, cell_map = build_cell_map(
        df, x_col=x_col, y_col=y_col, top_col=top_col, bottom_col=bottom_col
    )
    if not x_vals or not y_vals:
        raise ValueError("No valid X/Y values found in input data.")

    max_top = max(v[0] for v in cell_map.values()) if cell_map else 0
    max_bottom = max(v[1] for v in cell_map.values()) if cell_map else 0
    top_norm = colors.LogNorm(vmin=1, vmax=max(1, max_top + 1))
    bottom_norm = colors.LogNorm(vmin=1, vmax=max(1, max_bottom + 1))
    top_cmap = colormaps["Blues"]
    bottom_cmap = colormaps["Reds"]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#d9d9d9")

    for y in y_vals:
        for x in x_vals:
            xi = x_index[x]
            yi = y_index[y]
            if (x, y) not in cell_map:
                ax.add_patch(
                    Rectangle(
                        (xi, yi), 1.0, 1.0, facecolor="#d9d9d9", edgecolor="white", linewidth=2
                    )
                )
                continue

            top_v, bottom_v = cell_map[(x, y)]
            ax.add_patch(
                Rectangle(
                    (xi, yi + 0.5),
                    1.0,
                    0.5,
                    facecolor=top_cmap(top_norm(top_v + 1)),
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (xi, yi),
                    1.0,
                    0.5,
                    facecolor=bottom_cmap(bottom_norm(bottom_v + 1)),
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle((xi, yi), 1.0, 1.0, fill=False, edgecolor="white", linewidth=2)
            )
            ax.text(
                xi + 0.5,
                yi + 0.75,
                _format_cell(top_v),
                ha="center",
                va="center",
                fontsize=7,
            )
            ax.text(
                xi + 0.5,
                yi + 0.25,
                _format_cell(bottom_v),
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, len(y_vals))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([i + 0.5 for i in range(len(x_vals))])
    ax.set_xticklabels([_format_tick(v) for v in x_vals])
    ax.set_yticks([i + 0.5 for i in range(len(y_vals))])
    ax.set_yticklabels([_format_tick(v) for v in y_vals])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    fig.subplots_adjust(right=0.88)
    top_sm = cm.ScalarMappable(norm=top_norm, cmap=top_cmap)
    top_sm.set_array([])
    top_cax = fig.add_axes([0.90, 0.56, 0.02, 0.35])
    top_cbar = fig.colorbar(top_sm, cax=top_cax)
    top_cbar.set_label(f"{top_label} (Log Scale)")

    bottom_sm = cm.ScalarMappable(norm=bottom_norm, cmap=bottom_cmap)
    bottom_sm.set_array([])
    bottom_cax = fig.add_axes([0.90, 0.14, 0.02, 0.35])
    bottom_cbar = fig.colorbar(bottom_sm, cax=bottom_cax)
    bottom_cbar.set_label(f"{bottom_label} (Log Scale)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv)
    make_plot(
        df=df,
        output_path=args.out,
        x_col=args.x_col,
        y_col=args.y_col,
        top_col=args.top_col,
        bottom_col=args.bottom_col,
        top_label=args.top_label,
        bottom_label=args.bottom_label,
        title=args.title,
    )
    print(f"Saved heatmap to: {args.out}")


if __name__ == "__main__":
    main()
