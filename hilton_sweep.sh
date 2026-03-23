#!/bin/bash
set -euo pipefail

# =======================
# Fixed Delta parameters
# =======================
DELTA_NEURON_COUNT=1500
DELTA_MULTIPLIER=2.0

# =======================
# Hilton sweep parameters
# =======================
HILTON_NEURON_COUNTS=(250 500 750 1000 1250 1500 1750 2000 2250 2500)
HILTON_MULTIPLIERS=(1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0)

# =======================
# Shared runtime arguments
# =======================
IG_STEPS=5
THRESHOLD=0.005
PARALLEL_GPUS="0,1"
PYTHON_BIN="python"
SCRIPT_PATH="neuron_test.py"

# Optional environment setup (same style as your train.sh)
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
set +u
source activate SVD
set -u
export CUDA_VISIBLE_DEVICES=0,1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888

run_dir="hilton_sweep_output"
mkdir -p "${run_dir}/logs"
rm -f "${run_dir}/logs/"*.log

# Freeze code snapshot for the whole sweep so later edits
# to neuron_test.py/src do not affect in-flight runs.
snapshot_dir="${run_dir}/code_snapshot"
mkdir -p "${snapshot_dir}"
cp "${SCRIPT_PATH}" "${snapshot_dir}/$(basename "${SCRIPT_PATH}")"
cp -r "src" "${snapshot_dir}/src"
SCRIPT_PATH="${snapshot_dir}/$(basename "${SCRIPT_PATH}")"

report_txt="${run_dir}/hilton_sweep_report.txt"
summary_tsv="${run_dir}/hilton_sweep_summary.tsv"

cat > "${report_txt}" <<EOF
Fixed Delta params:
  delta_neuron_count=${DELTA_NEURON_COUNT}
  delta_multiplier=${DELTA_MULTIPLIER}
  code_snapshot=${snapshot_dir}
EOF

printf "run_id\thilton_neuron_count\thilton_multiplier\thit_delta\thit_hilton\thit_four_seasons\thit_hawaiian\traw_log\n" > "${summary_tsv}"

run_id=0
for hilton_count in "${HILTON_NEURON_COUNTS[@]}"; do
  for hilton_mult in "${HILTON_MULTIPLIERS[@]}"; do
    run_id=$((run_id + 1))
    log_file="${run_dir}/logs/run_${run_id}_hc${hilton_count}_hm${hilton_mult}.log"

    echo "[Run ${run_id}] hilton_neuron_count=${hilton_count}, hilton_multiplier=${hilton_mult}"

    cmd=(
      "${PYTHON_BIN}" "${SCRIPT_PATH}"
      --enable_Hilton
      --enable_Delta
      --ig_steps "${IG_STEPS}"
      --hilton-neuron-count "${hilton_count}"
      --hilton-multiplier "${hilton_mult}"
      --delta-neuron-count "${DELTA_NEURON_COUNT}"
      --delta-multiplier "${DELTA_MULTIPLIER}"
      --parallel-gpus "${PARALLEL_GPUS}"
      --delta-score-mode contrastive
      --hilton-score-mode contrastive
      --threshold "${THRESHOLD}"
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
hit_delta = int(bool(re.search(r"\bdelta\b", result_lower)))
hit_hilton = int(bool(re.search(r"\bhilton\b", result_lower)))
hit_four_seasons = int("four seasons" in result_lower)
hit_hawaiian = int("hawaiian airlines" in result_lower)

print(f"HIT_DELTA={hit_delta}")
print(f"HIT_HILTON={hit_hilton}")
print(f"HIT_FOUR_SEASONS={hit_four_seasons}")
print(f"HIT_HAWAIIAN={hit_hawaiian}")
print("RESULT_BLOCK_BEGIN")
print(result_block)
print("RESULT_BLOCK_END")
PY
    )"

    hit_delta="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_DELTA=/{print $2}')"
    hit_hilton="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_HILTON=/{print $2}')"
    hit_four_seasons="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_FOUR_SEASONS=/{print $2}')"
    hit_hawaiian="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_HAWAIIAN=/{print $2}')"
    result_block="$(printf '%s\n' "${parse_output}" | awk '/^RESULT_BLOCK_BEGIN$/{flag=1;next}/^RESULT_BLOCK_END$/{flag=0}flag')"

    {
      echo ""
      echo "------------------------------------------------------------"
      echo "Run ${run_id}"
      echo "hilton_neuron_count=${hilton_count}"
      echo "hilton_multiplier=${hilton_mult}"
      echo "${result_block}"
      echo "------------------------------------------------------------"
    } >> "${report_txt}"

    printf "%s\t%s\t%.2f\t%s\t%s\t%s\t%s\t%s\n" \
      "${run_id}" \
      "${hilton_count}" \
      "${hilton_mult}" \
      "${hit_delta}" \
      "${hit_hilton}" \
      "${hit_four_seasons}" \
      "${hit_hawaiian}" \
      "${log_file}" >> "${summary_tsv}"
  done
done

echo ""
echo "Sweep finished at: $(date)"
echo "Report: ${report_txt}"
echo "Summary TSV: ${summary_tsv}"
echo ""
echo "Recommended candidates (hit_delta=1 and hit_hilton=1 and hit_four_seasons=0):"
awk -F'\t' 'NR==1 || ($4==1 && $5==1 && $6==0)' "${summary_tsv}"

echo "Done. Please check:"
echo "  ${report_txt}"
echo "  ${summary_tsv}"
