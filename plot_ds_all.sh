#!/bin/bash
set -uo pipefail

PLOT_DIR="./DS_plot"
USE_SUB=1
RESULT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sub)
      USE_SUB=1
      shift
      ;;
    --no-sub)
      USE_SUB=0
      shift
      ;;
    -h|--help)
      echo "Usage: bash plot_ds_all.sh [--sub|--no-sub]"
      echo "  --sub     Subtract baseline (0,0) and clip at 0 (default)"
      echo "  --no-sub  Disable baseline subtraction"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -d ".batch_result" ]]; then
  RESULT_ROOT=".batch_result"
elif [[ -d "batch_results" ]]; then
  RESULT_ROOT="batch_results"
else
  echo "[ERROR] Neither .batch_result nor batch_results exists." >&2
  exit 1
fi

mkdir -p "${PLOT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib_ds_plot"
mkdir -p "${MPLCONFIGDIR}"

readarray -t base_dirs < <(find "${RESULT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'DS*' | sort)

if (( ${#base_dirs[@]} == 0 )); then
  echo "[ERROR] No DS result directories found under ${RESULT_ROOT}" >&2
  exit 1
fi

processed_count=0
skipped_count=0
generated_avg_csvs=()
generated_plot_dirs=()

echo "[Info] Result root: ${RESULT_ROOT}"
echo "[Info] Plot output root: ${PLOT_DIR}"
echo "[Stage 1/2] Aggregating summary csv files..."

for base_dir in "${base_dirs[@]}"; do
  run_name="$(basename "${base_dir}")"
  run_plot_dir="${PLOT_DIR}/${run_name}"
  avg_csv="${run_plot_dir}/summary_avg.csv"
  prompt_avg_csv="${base_dir}/prompt_hit_avg.csv"
  mkdir -p "${run_plot_dir}"

  if [[ -f "${prompt_avg_csv}" ]]; then
    echo "[Run] ${base_dir}: using aggregated csv ${prompt_avg_csv} -> ${avg_csv}"
    cp "${prompt_avg_csv}" "${avg_csv}"
    generated_avg_csvs+=("${avg_csv}")
    generated_plot_dirs+=("${run_plot_dir}")
    processed_count=$((processed_count + 1))
    continue
  fi

  readarray -t input_paths < <(find "${base_dir}" -maxdepth 2 -type f -regextype posix-extended -regex '.*/p[0-9]+/summary_[0-9]+\.csv' | sort -V)

  if (( ${#input_paths[@]} == 0 )); then
    echo "[Skip] ${base_dir}: no p*/summary_*.csv files found"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  row_count_mismatch=0
  row_count_ref=-1
  row_count_info=()
  for csv_path in "${input_paths[@]}"; do
    data_rows="$(awk 'NR>1 {c++} END {print c+0}' "${csv_path}")"
    row_count_info+=("$(basename "$(dirname "${csv_path}")")/$(basename "${csv_path}")=${data_rows}")
    if (( row_count_ref < 0 )); then
      row_count_ref="${data_rows}"
    elif (( data_rows != row_count_ref )); then
      row_count_mismatch=1
    fi
  done

  if (( row_count_mismatch == 1 )); then
    echo "[Skip] ${base_dir}: summary csv row counts are not equal (${row_count_info[*]})"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo "[Run] ${base_dir}: aggregating ${#input_paths[@]} csv files -> ${avg_csv}"
  python average_summary_csv.py \
    --inputs "${input_paths[@]}" \
    --output "${avg_csv}"

  generated_avg_csvs+=("${avg_csv}")
  generated_plot_dirs+=("${run_plot_dir}")
  processed_count=$((processed_count + 1))
done

if (( ${#generated_avg_csvs[@]} == 0 )); then
  echo ""
  echo "Done."
  echo "Processed dirs: 0"
  echo "Skipped dirs: ${skipped_count}"
  echo "Output plot dir: ${PLOT_DIR}"
  exit 0
fi

echo "[Stage 2/2] Plotting aggregated csv files..."
for idx in "${!generated_avg_csvs[@]}"; do
  avg_csv="${generated_avg_csvs[$idx]}"
  run_plot_dir="${generated_plot_dirs[$idx]}"

  plot_cmd=(python plot_avg_heatmap.py \
    --csv "${avg_csv}" \
    --output-dir "${run_plot_dir}")
  if (( USE_SUB == 1 )); then
    plot_cmd+=(--sub)
  fi

  echo "[Run] ${avg_csv}: plotting heatmaps -> ${run_plot_dir}"
  "${plot_cmd[@]}"
done

echo "[Run] building global normalized heatmaps for blue/red brands"
python - "${PLOT_DIR}" "${USE_SUB}" "${generated_avg_csvs[@]}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from matplotlib.patches import Rectangle

plot_dir = Path(sys.argv[1])
use_sub = bool(int(sys.argv[2]))
csv_paths = [Path(x) for x in sys.argv[3:]]

blue_sum = defaultdict(float)
blue_cnt = defaultdict(int)
red_sum = defaultdict(float)
red_cnt = defaultdict(int)

for path in csv_paths:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 7:
            raise ValueError(f"unexpected columns in {path}")
        x_col = fieldnames[1]
        y_col = fieldnames[2]
        blue_col = fieldnames[-2]
        red_col = fieldnames[-1]

        rows = list(reader)
        base_blue = 0.0
        base_red = 0.0
        if use_sub:
            baseline = None
            for row in rows:
                if abs(float(row[x_col]) - 0.0) < 1e-9 and abs(float(row[y_col]) - 0.0) < 1e-9:
                    baseline = row
                    break
            if baseline is not None:
                base_blue = float(baseline[blue_col])
                base_red = float(baseline[red_col])
            else:
                print(f"warning: baseline (0,0) not found in {path}, use 0 as baseline")

        for row in rows:
            x = float(row[x_col])
            y = float(row[y_col])
            coord = (x, y)
            blue_val = float(row[blue_col])
            red_val = float(row[red_col])
            if use_sub:
                blue_val = max(0.0, blue_val - base_blue)
                red_val = max(0.0, red_val - base_red)
            blue_sum[coord] += blue_val
            blue_cnt[coord] += 1
            red_sum[coord] += red_val
            red_cnt[coord] += 1

if not blue_sum:
    raise ValueError("no rows found for global aggregation")

blue_avg = {}
red_avg = {}
for coord, total in blue_sum.items():
    if blue_cnt[coord] > 0:
        blue_avg[coord] = total / blue_cnt[coord]
for coord, total in red_sum.items():
    if red_cnt[coord] > 0:
        red_avg[coord] = total / red_cnt[coord]

def normalize_map(data_map):
    values = list(data_map.values())
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return {k: 0.0 for k in data_map}
    return {k: (v - vmin) / (vmax - vmin) for k, v in data_map.items()}

blue_norm = normalize_map(blue_avg)
red_norm = normalize_map(red_avg)

def fmt_tick(v):
    return str(int(v)) if float(v).is_integer() else str(v)

def fmt_cell(v):
    return f"{v:.2f}"

def draw_heatmap(data_map, cmap_name, title, out_path):
    x_vals = sorted({x for x, _ in data_map})
    y_vals = sorted({y for _, y in data_map})
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    cmap = colormaps[cmap_name]
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f5f5f5")

    for (x, y), val in data_map.items():
        xi, yi = x_idx[x], y_idx[y]
        face = cmap(norm(val))
        rect = Rectangle((xi, yi), 1.0, 1.0, facecolor=face, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(
            xi + 0.5, yi + 0.5, fmt_cell(val),
            ha="center", va="center",
            fontsize=9,
            color="black" if val < 0.6 else "white"
        )

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, len(y_vals))
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(len(x_vals))])
    ax.set_xticklabels([fmt_tick(v) for v in x_vals])
    ax.set_yticks([i + 0.5 for i in range(len(y_vals))])
    ax.set_yticklabels([fmt_tick(v) for v in y_vals])
    ax.set_xlabel("brand_1_top_k")
    ax.set_ylabel("brand_2_top_k")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Normalized hit_count [0,1]")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

draw_heatmap(
    blue_norm,
    "Blues",
    "Global Blue Brands Mean Hit Count (Normalized, Sub)" if use_sub else "Global Blue Brands Mean Hit Count (Normalized)",
    plot_dir / ("global_blue_mean_normalized_sub.png" if use_sub else "global_blue_mean_normalized.png"),
)
draw_heatmap(
    red_norm,
    "Reds",
    "Global Red Brands Mean Hit Count (Normalized, Sub)" if use_sub else "Global Red Brands Mean Hit Count (Normalized)",
    plot_dir / ("global_red_mean_normalized_sub.png" if use_sub else "global_red_mean_normalized.png"),
)
PY

echo ""
echo "Done."
echo "Processed dirs: ${processed_count}"
echo "Skipped dirs: ${skipped_count}"
echo "Output plot dir: ${PLOT_DIR}"
