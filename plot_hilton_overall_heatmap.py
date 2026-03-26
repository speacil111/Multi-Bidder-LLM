import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(
    csv_path: Path, output_path: Path, mask_zero: bool = False, title: str = "Overall_Score Heatmap"
) -> None:
    df = pd.read_csv(csv_path)
    pivot = (
        df.pivot(index="Multiplier", columns="Top-K", values="Overall_Score")
        .sort_index()
        .sort_index(axis=1)
    )

    values = pivot.values.astype(float)
    display_values = np.ma.masked_where(values == 0, values) if mask_zero else values

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(display_values, aspect="auto", cmap=cmap, vmin=0, vmax=5, origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Multiplier")
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = values[i, j]
            if val == 0 and mask_zero:
                ax.text(j, i, "0.0", ha="center", va="center", fontsize=7, color="dimgray")
            else:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Overall_Score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hilton Overall_Score heatmap.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("hilton_topk_search_output/hilton_llm_judge_scores_balanced.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("hilton_topk_search_output/hilton_overall_score_heatmap_balanced.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--no-mask-zero",
        action="store_true",
        help="Disable gray masking for zero-score cells.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Overall_Score Heatmap",
        help="Chart title.",
    )
    args = parser.parse_args()

    plot_heatmap(args.csv, args.out, mask_zero=not args.no_mask_zero, title=args.title)
    print(f"Saved heatmap to: {args.out}")


if __name__ == "__main__":
    main()
