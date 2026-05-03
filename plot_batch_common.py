#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_batch_plot")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from matplotlib.patches import Rectangle


def parse_common_args(
    *,
    description: str,
    default_result_root: str,
    default_plot_dir: str,
    default_model_prefix: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(default_result_root),
        help=f"Result root. Default: {default_result_root}",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(default_plot_dir),
        help=f"Plot output root. Default: {default_plot_dir}",
    )
    parser.add_argument(
        "--model-prefix",
        default=default_model_prefix,
        help=f"Directory prefix to scan. Default: {default_model_prefix}",
    )
    parser.add_argument(
        "--sub",
        dest="use_sub",
        action="store_true",
        default=True,
        help="Subtract baseline (0,0) and clip at 0. Default enabled.",
    )
    parser.add_argument(
        "--no-sub",
        dest="use_sub",
        action="store_false",
        help="Disable baseline subtraction.",
    )
    parser.add_argument(
        "--outlier-hit-threshold",
        type=float,
        default=20.0,
        help="Global aggregation treats hit_* cells above this value as 0. Default: 20.",
    )
    parser.add_argument(
        "--outlier-example-limit",
        type=int,
        default=10,
        help="Number of outlier examples to print. Use 0 to suppress examples. Default: 10.",
    )
    parser.add_argument(
        "--outlier-report",
        type=Path,
        default=None,
        help=(
            "Path for detailed outlier TSV. Default: "
            "<plot-dir>/outlier_hit_count.tsv."
        ),
    )
    parser.add_argument(
        "--no-outlier-report",
        action="store_true",
        help="Do not write the detailed outlier TSV report.",
    )
    return parser.parse_args()


def data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def prompt_id_from_summary(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("summary_"):
        return 10**9
    suffix = stem.removeprefix("summary_")
    return int(suffix) if suffix.isdigit() else 10**9


def collect_input_paths(base_dir: Path) -> list[Path]:
    paths = []
    for prompt_dir in base_dir.glob("p*"):
        if not prompt_dir.is_dir():
            continue
        paths.extend(prompt_dir.glob("summary_*.csv"))
    return sorted(paths, key=lambda p: (prompt_id_from_summary(p), str(p)))


def run_checked(cmd: list[str]) -> None:
    print("[Cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fmt_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def fmt_cell(value: float) -> str:
    return f"{value:.2f}"


def normalize_map(data_map: dict[tuple[float, float], float]) -> dict[tuple[float, float], float]:
    values = list(data_map.values())
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return {key: 0.0 for key in data_map}
    return {key: (value - vmin) / (vmax - vmin) for key, value in data_map.items()}


def draw_global_heatmap(
    data_map: dict[tuple[float, float], float],
    cmap_name: str,
    title: str,
    out_path: Path,
) -> None:
    x_vals = sorted({x for x, _ in data_map})
    y_vals = sorted({y for _, y in data_map})
    x_idx = {value: idx for idx, value in enumerate(x_vals)}
    y_idx = {value: idx for idx, value in enumerate(y_vals)}

    cmap = colormaps[cmap_name]
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f5f5f5")

    for (x, y), value in data_map.items():
        xi = x_idx[x]
        yi = y_idx[y]
        ax.add_patch(
            Rectangle(
                (xi, yi),
                1.0,
                1.0,
                facecolor=cmap(norm(value)),
                edgecolor="white",
                linewidth=1,
            )
        )
        ax.text(
            xi + 0.5,
            yi + 0.5,
            fmt_cell(value),
            ha="center",
            va="center",
            fontsize=9,
            color="black" if value < 0.6 else "white",
        )

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, len(y_vals))
    ax.set_aspect("equal")
    ax.set_xticks([idx + 0.5 for idx in range(len(x_vals))])
    ax.set_xticklabels([fmt_tick(value) for value in x_vals])
    ax.set_yticks([idx + 0.5 for idx in range(len(y_vals))])
    ax.set_yticklabels([fmt_tick(value) for value in y_vals])
    ax.set_xlabel("brand_1_top_k")
    ax.set_ylabel("brand_2_top_k")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Normalized hit_count [0,1]")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def build_global_heatmaps(
    *,
    plot_dir: Path,
    use_sub: bool,
    outlier_threshold: float,
    outlier_example_limit: int,
    outlier_report: Path | None,
    csv_paths: list[Path],
) -> None:
    blue_sum: defaultdict[tuple[float, float], float] = defaultdict(float)
    blue_cnt: defaultdict[tuple[float, float], int] = defaultdict(int)
    red_sum: defaultdict[tuple[float, float], float] = defaultdict(float)
    red_cnt: defaultdict[tuple[float, float], int] = defaultdict(int)
    outlier_hit_count = 0
    outlier_examples = []
    outlier_records = []

    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            if len(fieldnames) < 7:
                raise ValueError(f"unexpected columns in {path}")
            run_col = fieldnames[0]
            x_col = fieldnames[1]
            y_col = fieldnames[2]
            blue_col = fieldnames[-2]
            red_col = fieldnames[-1]
            rows = list(reader)

        hit_cols = [col for col in fieldnames if col.startswith("hit_")]
        zeroed_cells = set()
        for row_idx, row in enumerate(rows, start=2):
            for col in hit_cols:
                value = float(row[col])
                if value > outlier_threshold:
                    outlier_hit_count += 1
                    zeroed_cells.add((row_idx, col))
                    record = {
                        "brand": path.parent.name,
                        "path": str(path),
                        "csv_row": row_idx,
                        "run_id": row.get(run_col, ""),
                        "x": row.get(x_col, ""),
                        "y": row.get(y_col, ""),
                        "hit_col": col,
                        "value": value,
                        "threshold": outlier_threshold,
                    }
                    outlier_records.append(record)
                    if len(outlier_examples) < outlier_example_limit:
                        outlier_examples.append(record)

        base_blue = 0.0
        base_red = 0.0
        if use_sub:
            baseline = None
            baseline_row_idx = None
            for row_idx, row in enumerate(rows, start=2):
                if abs(float(row[x_col])) < 1e-9 and abs(float(row[y_col])) < 1e-9:
                    baseline = row
                    baseline_row_idx = row_idx
                    break
            if baseline is not None:
                base_blue = (
                    0.0
                    if (baseline_row_idx, blue_col) in zeroed_cells
                    else float(baseline[blue_col])
                )
                base_red = (
                    0.0
                    if (baseline_row_idx, red_col) in zeroed_cells
                    else float(baseline[red_col])
                )
            else:
                print(f"warning: baseline (0,0) not found in {path}, use 0 as baseline")

        for row_idx, row in enumerate(rows, start=2):
            x = float(row[x_col])
            y = float(row[y_col])
            coord = (x, y)
            blue_val = 0.0 if (row_idx, blue_col) in zeroed_cells else float(row[blue_col])
            red_val = 0.0 if (row_idx, red_col) in zeroed_cells else float(row[red_col])
            if use_sub:
                blue_val = max(0.0, blue_val - base_blue)
                red_val = max(0.0, red_val - base_red)
            blue_sum[coord] += blue_val
            blue_cnt[coord] += 1
            red_sum[coord] += red_val
            red_cnt[coord] += 1

    if not blue_sum:
        raise ValueError("no rows found for global aggregation after outlier filtering")

    if outlier_report is not None:
        outlier_report.parent.mkdir(parents=True, exist_ok=True)
        report_fields = [
            "brand",
            "path",
            "csv_row",
            "run_id",
            "x",
            "y",
            "hit_col",
            "value",
            "threshold",
        ]
        with outlier_report.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=report_fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(outlier_records)

    if outlier_hit_count:
        print(f"[Outlier filter] threshold: hit_count > {outlier_threshold:g}")
        print(
            f"[Outlier filter] zeroed {outlier_hit_count} exploding hit_count cells in global mean"
        )
        affected_csvs = len({record["path"] for record in outlier_records})
        affected_coords = len({(record["x"], record["y"]) for record in outlier_records})
        print(f"[Outlier filter] affected csvs: {affected_csvs}")
        print(f"[Outlier filter] affected coordinates: {affected_coords}")
        if outlier_report is not None:
            print(f"[Outlier filter] detailed report: {outlier_report}")
        if outlier_examples:
            print(f"[Outlier filter] first {len(outlier_examples)} examples:")
        for item in outlier_examples:
            print(
                "  "
                f"{item['brand']}: {item['hit_col']}={item['value']:.6g} "
                f"at csv_row={item['csv_row']}, run_id={item['run_id']}, "
                f"({item['x']}, {item['y']}); csv={item['path']}"
            )
    else:
        print(f"[Outlier filter] no hit_count cell > {outlier_threshold:g}")
        if outlier_report is not None:
            print(f"[Outlier filter] wrote empty detailed report: {outlier_report}")

    print(f"[Outlier filter] global mean uses all {len(csv_paths)} brand csvs")

    blue_avg = {coord: total / blue_cnt[coord] for coord, total in blue_sum.items()}
    red_avg = {coord: total / red_cnt[coord] for coord, total in red_sum.items()}

    draw_global_heatmap(
        normalize_map(blue_avg),
        "Blues",
        "Global Blue Brands Mean Hit Count (Normalized, Sub)"
        if use_sub
        else "Global Blue Brands Mean Hit Count (Normalized)",
        plot_dir / ("global_blue_mean_normalized_sub.png" if use_sub else "global_blue_mean_normalized.png"),
    )
    draw_global_heatmap(
        normalize_map(red_avg),
        "Reds",
        "Global Red Brands Mean Hit Count (Normalized, Sub)"
        if use_sub
        else "Global Red Brands Mean Hit Count (Normalized)",
        plot_dir / ("global_red_mean_normalized_sub.png" if use_sub else "global_red_mean_normalized.png"),
    )


def run_plot_batch(args: argparse.Namespace) -> None:
    result_root = args.result_root
    plot_dir = args.plot_dir
    model_prefix = args.model_prefix

    if not result_root.is_dir():
        raise SystemExit(f"[ERROR] Result root not found: {result_root}")

    base_dirs = sorted(path for path in result_root.glob(f"{model_prefix}*") if path.is_dir())
    if not base_dirs:
        raise SystemExit(f"[ERROR] No {model_prefix} result directories found under {result_root}")

    plot_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    generated_avg_csvs: list[Path] = []
    generated_plot_dirs: list[Path] = []

    print(f"[Info] Result root: {result_root}")
    print(f"[Info] Plot output root: {plot_dir}")
    print("[Stage 1/2] Aggregating summary csv files...")

    for base_dir in base_dirs:
        run_plot_dir = plot_dir / base_dir.name
        avg_csv = run_plot_dir / "summary_avg.csv"
        prompt_avg_csv = base_dir / "prompt_hit_avg.csv"
        run_plot_dir.mkdir(parents=True, exist_ok=True)

        if prompt_avg_csv.is_file():
            print(f"[Run] {base_dir}: using aggregated csv {prompt_avg_csv} -> {avg_csv}")
            shutil.copyfile(prompt_avg_csv, avg_csv)
            generated_avg_csvs.append(avg_csv)
            generated_plot_dirs.append(run_plot_dir)
            processed_count += 1
            continue

        input_paths = collect_input_paths(base_dir)
        if not input_paths:
            print(f"[Skip] {base_dir}: no p*/summary_*.csv files found")
            skipped_count += 1
            continue

        row_counts = [(path, data_row_count(path)) for path in input_paths]
        if len({count for _, count in row_counts}) != 1:
            info = " ".join(f"{path.parent.name}/{path.name}={count}" for path, count in row_counts)
            print(f"[Skip] {base_dir}: summary csv row counts are not equal ({info})")
            skipped_count += 1
            continue

        print(f"[Run] {base_dir}: aggregating {len(input_paths)} csv files -> {avg_csv}")
        run_checked(
            [
                sys.executable,
                "average_summary_csv.py",
                "--inputs",
                *[str(path) for path in input_paths],
                "--output",
                str(avg_csv),
            ]
        )
        generated_avg_csvs.append(avg_csv)
        generated_plot_dirs.append(run_plot_dir)
        processed_count += 1

    if not generated_avg_csvs:
        print("")
        print("Done.")
        print("Processed dirs: 0")
        print(f"Skipped dirs: {skipped_count}")
        print(f"Output plot dir: {plot_dir}")
        return

    print("[Stage 2/2] Plotting aggregated csv files...")
    for avg_csv, run_plot_dir in zip(generated_avg_csvs, generated_plot_dirs):
        cmd = [
            sys.executable,
            "plot_avg_heatmap.py",
            "--csv",
            str(avg_csv),
            "--output-dir",
            str(run_plot_dir),
        ]
        if args.use_sub:
            cmd.append("--sub")
        print(f"[Run] {avg_csv}: plotting heatmaps -> {run_plot_dir}")
        run_checked(cmd)

    print("[Run] building global normalized heatmaps for blue/red brands")
    outlier_report = None
    if not args.no_outlier_report:
        outlier_report = args.outlier_report or (plot_dir / "outlier_hit_count.tsv")

    build_global_heatmaps(
        plot_dir=plot_dir,
        use_sub=args.use_sub,
        outlier_threshold=args.outlier_hit_threshold,
        outlier_example_limit=max(0, args.outlier_example_limit),
        outlier_report=outlier_report,
        csv_paths=generated_avg_csvs,
    )

    print("")
    print("Done.")
    print(f"Processed dirs: {processed_count}")
    print(f"Skipped dirs: {skipped_count}")
    print(f"Output plot dir: {plot_dir}")
