#!/bin/bash
set -euo pipefail

# =======================
# Fixed intervention counts
# =======================
HILTON_NEURON_COUNT=500
DELTA_NEURON_COUNT=500

# =======================
# Multiplier sweep parameters
# =======================
HILTON_MULTIPLIERS=(1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00)
DELTA_MULTIPLIERS=(1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00)

# =======================
# Shared runtime arguments
# =======================
IG_STEPS=20
THRESHOLD=0.000
PARALLEL_GPUS="0,1"
PYTHON_BIN="python"
SCRIPT_PATH="neuron_test.py"
ATTR_CACHE_DIR="attr_score_cache"


export CUDA_VISIBLE_DEVICES=0,1

run_dir="double_500_[1-4]_output"
mkdir -p "${run_dir}/logs"
rm -f "${run_dir}/logs/"*.log

# Freeze code snapshot for the whole sweep so later edits
# to neuron_test.py/src do not affect in-flight runs.
snapshot_dir="${run_dir}/code_snapshot"
mkdir -p "${snapshot_dir}"
cp "${SCRIPT_PATH}" "${snapshot_dir}/$(basename "${SCRIPT_PATH}")"
cp -r "src" "${snapshot_dir}/src"
SCRIPT_PATH="${snapshot_dir}/$(basename "${SCRIPT_PATH}")"

report_txt="${run_dir}/report.txt"
summary_tsv="${run_dir}/summary.tsv"

cat > "${report_txt}" <<EOF
Fixed params:
  hilton_neuron_count=${HILTON_NEURON_COUNT}
  delta_neuron_count=${DELTA_NEURON_COUNT}
  multiplier_range=[1.00, 4.00], step=0.25
  attribution_cache_dir=${ATTR_CACHE_DIR}
  code_snapshot=${snapshot_dir}
EOF

printf "run_id\thilton_neuron_count\tdelta_neuron_count\thilton_multiplier\tdelta_multiplier\thit_delta\thit_hilton\n" > "${summary_tsv}"

run_id=0
for hilton_mult in "${HILTON_MULTIPLIERS[@]}"; do
  for delta_mult in "${DELTA_MULTIPLIERS[@]}"; do
    run_id=$((run_id + 1))
    log_file="${run_dir}/logs/run_${run_id}_hc${HILTON_NEURON_COUNT}_dc${DELTA_NEURON_COUNT}_hm${hilton_mult}_dm${delta_mult}.log"

    echo "[Run ${run_id}] hilton_neuron_count=${HILTON_NEURON_COUNT}, delta_neuron_count=${DELTA_NEURON_COUNT}, hilton_multiplier=${hilton_mult}, delta_multiplier=${delta_mult}"

    cmd=(
      "${PYTHON_BIN}" "${SCRIPT_PATH}"
      --enable_Hilton
      --enable_Delta
      --ig_steps "${IG_STEPS}"
      --hilton-neuron-count "${HILTON_NEURON_COUNT}"
      --hilton-multiplier "${hilton_mult}"
      --delta-neuron-count "${DELTA_NEURON_COUNT}"
      --delta-multiplier "${delta_mult}"
      --parallel-gpus "${PARALLEL_GPUS}"
      --delta-score-mode contrastive
      --hilton-score-mode contrastive
      --threshold "${THRESHOLD}"
      --attribution-cache-dir "${ATTR_CACHE_DIR}"
      --intervention_layer -1
    )

    # Save full stdout/stderr for each run
    "${cmd[@]}" > "${log_file}" 2>&1

    # Parse key output and append to report + summary
    parse_output="$(
      python - "${log_file}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()

# Find intervention "Result:" block
start_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("Result:"):
        start_idx = i

if start_idx is None:
    result_block = ""
else:
    block = [lines[start_idx]]
    for j in range(start_idx + 1, len(lines)):
        cur = lines[j]
        if cur.startswith("============ Job "):
            break
        if cur.startswith("  0%|") or cur.startswith("100%|"):
            continue
        block.append(cur)
    result_block = "\n".join(block).strip()

result_lower = result_block.lower()
hit_delta = len(re.findall(r"\bdelta\b", result_lower))
hit_hilton = len(re.findall(r"\bhilton\b", result_lower))

print(f"HIT_DELTA={hit_delta}")
print(f"HIT_HILTON={hit_hilton}")
print("RESULT_BLOCK_BEGIN")
print(result_block)
print("RESULT_BLOCK_END")
PY
    )"

    hit_delta="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_DELTA=/{print $2}')"
    hit_hilton="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_HILTON=/{print $2}')"
    result_block="$(printf '%s\n' "${parse_output}" | awk '/^RESULT_BLOCK_BEGIN$/{flag=1;next}/^RESULT_BLOCK_END$/{flag=0}flag')"

    {
      echo ""
      echo "------------------------------------------------------------"
      echo "Run ${run_id}"
      echo "hilton_neuron_count=${HILTON_NEURON_COUNT}"
      echo "delta_neuron_count=${DELTA_NEURON_COUNT}"
      echo "hilton_multiplier=${hilton_mult}"
      echo "delta_multiplier=${delta_mult}"
      echo "${result_block}"
      echo "hit_hilton=${hit_hilton}, hit_delta=${hit_delta}"
      echo "------------------------------------------------------------"
    } >> "${report_txt}"

    printf "%s\t%s\t%s\t%.2f\t%.2f\t%s\t%s\n" \
      "${run_id}" \
      "${HILTON_NEURON_COUNT}" \
      "${DELTA_NEURON_COUNT}" \
      "${hilton_mult}" \
      "${delta_mult}" \
      "${hit_delta}" \
      "${hit_hilton}" >> "${summary_tsv}"
  done
done

echo ""
echo "Sweep finished at: $(date)"
echo "Report: ${report_txt}"
echo "Summary TSV: ${summary_tsv}"
echo ""
echo "Recommended candidates (hit_delta>0 and hit_hilton>0):"
awk -F'\t' 'NR==1 || ($6>0 && $7>0)' "${summary_tsv}"

echo "Done. Please check:"
echo "  ${report_txt}"
echo "  ${summary_tsv}"
