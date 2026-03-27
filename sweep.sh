#!/bin/bash
set -euo pipefail

# =======================
# Fixed multiplier settings
# =======================
HILTON_MULTIPLIER=2.0
DELTA_MULTIPLIER=2.0

# =======================
# Neuron-count sweep parameters 
# =======================
DELTA_NEURON_COUNTS=(100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
HILTON_NEURON_COUNTS=(100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)

# =======================
# Shared runtime arguments
# =======================
IG_STEPS=20
THRESHOLD=0.000
PARALLEL_GPUS="0"
PYTHON_BIN="python"
SCRIPT_PATH="neuron_test.py"
ATTR_CACHE_DIR="attr_score_cache"
PROMPT_INDEX=0
GPU_ID=0

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_dir="prompt_old${PROMPT_INDEX}_m2.0_output"
mkdir -p "${run_dir}/logs"
rm -f "${run_dir}/logs/"*.log

# Freeze code snapshot for the whole sweep so later edits
# to neuron_test.py/src do not affect in-flight runs.
snapshot_dir="${run_dir}/code_snapshot"
mkdir -p "${snapshot_dir}"
cp "${SCRIPT_PATH}" "${snapshot_dir}/$(basename "${SCRIPT_PATH}")"
cp -r "src" "${snapshot_dir}/src"
SCRIPT_PATH="${snapshot_dir}/$(basename "${SCRIPT_PATH}")"

report_txt="${run_dir}/report_${PROMPT_INDEX}.txt"
summary_tsv="${run_dir}/summary_${PROMPT_INDEX}.tsv"

cat > "${report_txt}" <<EOF
Fixed params:
  hilton_multiplier=${HILTON_MULTIPLIER}
  delta_multiplier=${DELTA_MULTIPLIER}
  prompt_index=${PROMPT_INDEX}
  gpu_id=${GPU_ID}
  hilton_neuron_counts=${HILTON_NEURON_COUNTS[*]}
  delta_neuron_counts=${DELTA_NEURON_COUNTS[*]}
  attribution_cache_dir=${ATTR_CACHE_DIR}
  code_snapshot=${snapshot_dir}
EOF

printf "run_id\thilton_neuron_count\tdelta_neuron_count\thilton_multiplier\tdelta_multiplier\thit_delta\thit_hilton\n" > "${summary_tsv}"

run_id=0
for hilton_count in "${HILTON_NEURON_COUNTS[@]}"; do
  for delta_count in "${DELTA_NEURON_COUNTS[@]}"; do
    run_id=$((run_id + 1))
    log_file="${run_dir}/logs/run_${run_id}_hc${hilton_count}_dc${delta_count}_hm${HILTON_MULTIPLIER}_dm${DELTA_MULTIPLIER}.log"

    echo "[Run ${run_id}] hilton_neuron_count=${hilton_count}, delta_neuron_count=${delta_count}, hilton_multiplier=${HILTON_MULTIPLIER}, delta_multiplier=${DELTA_MULTIPLIER}"

    cmd=(
      "${PYTHON_BIN}" "${SCRIPT_PATH}"
      --enable_Hilton
      --enable_Delta
      --ig_steps "${IG_STEPS}"
      --hilton-neuron-count "${hilton_count}"
      --hilton-multiplier "${HILTON_MULTIPLIER}"
      --delta-neuron-count "${delta_count}"
      --delta-multiplier "${DELTA_MULTIPLIER}"
      --parallel-gpus "${PARALLEL_GPUS}"
      --delta-score-mode contrastive
      --hilton-score-mode contrastive
      --threshold "${THRESHOLD}"
      --attribution-cache-dir "${ATTR_CACHE_DIR}"
      --intervention_layer -1
      --prompt-index "${PROMPT_INDEX}"
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
      echo "hilton_neuron_count=${hilton_count}"
      echo "delta_neuron_count=${delta_count}"
      echo "hilton_multiplier=${HILTON_MULTIPLIER}"
      echo "delta_multiplier=${DELTA_MULTIPLIER}"
      echo "prompt_index=${PROMPT_INDEX}"
      echo "gpu_id=${GPU_ID}"
      echo "${result_block}"
      echo "hit_hilton=${hit_hilton}, hit_delta=${hit_delta}"
      echo "------------------------------------------------------------"
    } >> "${report_txt}"

    printf "%s\t%s\t%s\t%.2f\t%.2f\t%s\t%s\n" \
      "${run_id}" \
      "${hilton_count}" \
      "${delta_count}" \
      "${HILTON_MULTIPLIER}" \
      "${DELTA_MULTIPLIER}" \
      "${hit_delta}" \
      "${hit_hilton}" >> "${summary_tsv}"
  done
done

echo ""
echo "Sweep finished at: $(date)"
echo "Report: ${report_txt}"
echo "Summary TSV: ${summary_tsv}"
echo ""


echo "Done. Please check:"
echo "  ${report_txt}"
echo "  ${summary_tsv}"
