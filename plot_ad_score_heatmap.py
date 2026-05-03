#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle


ID_PATTERN = re.compile(r"^(?P<k1>\d+)_(?P<k2>\d+)_ad(?P<ad>[12])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ad1/ad2 heatmaps from final_scored_ad_tag_test_scores.csv avg values."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ad_tag_test/final_scored_ad_tag_test_scores.csv"),
        help="Input CSV with columns: id,...,avg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ad_tag_test"),
        help="Directory for output PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI.")
    parser.add_argument("--vmin", type=float, default=1.0, help="Color scale minimum.")
    parser.add_argument("--vmax", type=float, default=5.0, help="Color scale maximum.")
    parser.add_argument(
        "--score-cols",
        nargs="+",
        default=["avg"],
        help="Score columns to plot, for example: avg q2 q3 q4.",
    )
    return parser.parse_args()


def fmt_tick(value: int) -> str:
    return str(value)


def fmt_cell(value: float) -> str:
    return f"{value:.2f}"


def load_score_maps(
    path: Path,
    score_cols: list[str],
) -> tuple[dict[str, dict[int, dict[tuple[int, int], float]]], list[int], list[int]]:
    score_maps: dict[str, dict[int, dict[tuple[int, int], float]]] = {
        score_col: {1: {}, 2: {}} for score_col in score_cols
    }
    k1_values: set[int] = set()
    k2_values: set[int] = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", *score_cols}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row_num, row in enumerate(reader, start=2):
            raw_id = (row.get("id") or "").strip()
            match = ID_PATTERN.match(raw_id)
            if not match:
                raise ValueError(f"Invalid id at row {row_num}: {raw_id!r}")

            k1 = int(match.group("k1"))
            k2 = int(match.group("k2"))
            ad_idx = int(match.group("ad"))

            for score_col in score_cols:
                score_maps[score_col][ad_idx][(k1, k2)] = float(row[score_col])
            k1_values.add(k1)
            k2_values.add(k2)

    return score_maps, sorted(k1_values), sorted(k2_values)


def draw_heatmap(
    data_map: dict[tuple[int, int], float],
    k1_values: list[int],
    k2_values: list[int],
    *,
    ad_idx: int,
    score_col: str,
    cmap_name: str,
    output_path: Path,
    dpi: int,
    vmin: float,
    vmax: float,
) -> None:
    x_idx = {value: idx for idx, value in enumerate(k1_values)}
    y_idx = {value: idx for idx, value in enumerate(k2_values)}

    cmap = plt.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f5f5f5")

    for k2 in k2_values:
        for k1 in k1_values:
            xi = x_idx[k1]
            yi = y_idx[k2]
            value = data_map.get((k1, k2))

            if value is None:
                face_color = "#e6e6e6"
                label = "NA"
                text_color = "#666666"
            else:
                face_color = cmap(norm(value))
                label = fmt_cell(value)
                text_color = "black" if value < (vmin + vmax) / 2 else "white"

            ax.add_patch(
                Rectangle(
                    (xi, yi),
                    1.0,
                    1.0,
                    facecolor=face_color,
                    edgecolor="white",
                    linewidth=1,
                )
            )
            ax.text(
                xi + 0.5,
                yi + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_xlim(0, len(k1_values))
    ax.set_ylim(0, len(k2_values))
    ax.set_aspect("equal")
    ax.set_xticks([idx + 0.5 for idx in range(len(k1_values))])
    ax.set_xticklabels([fmt_tick(value) for value in k1_values])
    ax.set_yticks([idx + 0.5 for idx in range(len(k2_values))])
    ax.set_yticklabels([fmt_tick(value) for value in k2_values])
    ax.set_xlabel("brand_1_top_k")
    ax.set_ylabel("brand_2_top_k")
    ax.set_title(f"Ad {ad_idx} {score_col} LLM Score")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=f"{score_col} LLM score")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    args = parse_args()
    score_maps, k1_values, k2_values = load_score_maps(args.csv, args.score_cols)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected_cells = len(k1_values) * len(k2_values)
    for score_col in args.score_cols:
        draw_heatmap(
            score_maps[score_col][1],
            k1_values,
            k2_values,
            ad_idx=1,
            score_col=score_col,
            cmap_name="Blues",
            output_path=args.output_dir / f"ad1_{score_col}_score_heatmap.png",
            dpi=args.dpi,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        draw_heatmap(
            score_maps[score_col][2],
            k1_values,
            k2_values,
            ad_idx=2,
            score_col=score_col,
            cmap_name="Reds",
            output_path=args.output_dir / f"ad2_{score_col}_score_heatmap.png",
            dpi=args.dpi,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        for ad_idx in (1, 2):
            missing_count = expected_cells - len(score_maps[score_col][ad_idx])
            if missing_count:
                print(f"warning: ad{ad_idx} {score_col} has {missing_count} missing cells")


if __name__ == "__main__":
    main()
