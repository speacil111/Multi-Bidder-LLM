#!/usr/bin/env python3
"""Average hit_Delta / hit_Hilton from multiple fair_mind summary CSVs and plot a
dual-half heatmap (top=Delta blue, bottom=Hilton red) identical in style to
plot_topk_hit_heatmap.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm, colormaps, colors
from matplotlib.patches import Rectangle


CSV_PATHS = [
    Path("fair_mind_Deltap1_m2.0/summary_1.csv"),
    Path("fair_mind_Deltap2_m2.0/summary_2.csv"),
    Path("fair_mind_Deltap3_m2.0/summary_3.csv"),
    Path("fair_mind_Deltap5_m2.0/summary_5.csv"),
]

OUTPUT_PATH = Path("more-hilton-heatmap/hit_heatmap_avg_fair_mind_1_2_3_5_subtract_baseline.png")

X_COL = "Delta_top_k"
Y_COL = "Hilton_top_k"
TOP_COL = "hit_Delta"
BOTTOM_COL = "hit_Hilton"


def _fmt_tick(v):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(fv)) if fv.is_integer() else str(fv)


def _fmt_cell(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    frames = [pd.read_csv(p, sep=None, engine="python") for p in CSV_PATHS]
    combined = pd.concat(frames, ignore_index=True)
    avg_df = (
        combined.groupby([X_COL, Y_COL])[[TOP_COL, BOTTOM_COL]]
        .mean()
        .reset_index()
    )

    x_vals = sorted(avg_df[X_COL].unique())
    y_vals = sorted(avg_df[Y_COL].unique())
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    DELTA_BASELINE = 1.75

    cell_map = {}
    for _, row in avg_df.iterrows():
        delta_v = max(0.0, float(row[TOP_COL]) - DELTA_BASELINE)
        cell_map[(row[X_COL], row[Y_COL])] = (delta_v, float(row[BOTTOM_COL]))

    max_top = max(v[0] for v in cell_map.values())
    max_bot = max(v[1] for v in cell_map.values())
    top_norm = colors.LogNorm(vmin=1, vmax=max(1, max_top + 1))
    bot_norm = colors.LogNorm(vmin=1, vmax=max(1, max_bot + 1))
    top_cmap = colormaps["Blues"]
    bot_cmap = colormaps["Reds"]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#d9d9d9")

    for y in y_vals:
        for x in x_vals:
            xi = x_idx[x]
            yi = y_idx[y]
            if (x, y) not in cell_map:
                ax.add_patch(
                    Rectangle((xi, yi), 1.0, 1.0,
                              facecolor="#d9d9d9", edgecolor="white", linewidth=2)
                )
                continue

            top_v, bot_v = cell_map[(x, y)]
            ax.add_patch(
                Rectangle((xi, yi + 0.5), 1.0, 0.5,
                           facecolor=top_cmap(top_norm(top_v + 1)), edgecolor="none")
            )
            ax.add_patch(
                Rectangle((xi, yi), 1.0, 0.5,
                           facecolor=bot_cmap(bot_norm(bot_v + 1)), edgecolor="none")
            )
            ax.add_patch(
                Rectangle((xi, yi), 1.0, 1.0, fill=False, edgecolor="white", linewidth=2)
            )
            ax.text(xi + 0.5, yi + 0.75, _fmt_cell(top_v),
                    ha="center", va="center", fontsize=7)
            ax.text(xi + 0.5, yi + 0.25, _fmt_cell(bot_v),
                    ha="center", va="center", fontsize=7)

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, len(y_vals))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([i + 0.5 for i in range(len(x_vals))])
    ax.set_xticklabels([_fmt_tick(v) for v in x_vals])
    ax.set_yticks([i + 0.5 for i in range(len(y_vals))])
    ax.set_yticklabels([_fmt_tick(v) for v in y_vals])
    ax.set_xlabel("Delta_top_k")
    ax.set_ylabel("Hilton_top_k")
    ax.set_title("Average Hit Counts Heatmap (fair_mind p1,p2,p3,p5) — Delta baseline subtracted")

    fig.subplots_adjust(right=0.88)

    top_sm = cm.ScalarMappable(norm=top_norm, cmap=top_cmap)
    top_sm.set_array([])
    top_cax = fig.add_axes([0.90, 0.56, 0.02, 0.35])
    top_cbar = fig.colorbar(top_sm, cax=top_cax)
    top_cbar.set_label("Avg Delta hits (Log Scale)")

    bot_sm = cm.ScalarMappable(norm=bot_norm, cmap=bot_cmap)
    bot_sm.set_array([])
    bot_cax = fig.add_axes([0.90, 0.14, 0.02, 0.35])
    bot_cbar = fig.colorbar(bot_sm, cax=bot_cax)
    bot_cbar.set_label("Avg Hilton hits (Log Scale)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
