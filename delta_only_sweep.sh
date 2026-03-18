#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,7-8,16-17,25,27-31] --gpus=2
set -euo pipefail

# =======================
# Delta sweep parameters
# =======================
DELTA_NEURON_COUNTS=(750 1000 1250 1500 1750 2000 2250 2500)
DELTA_MULTIPLIERS=(1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0)

# =======================
# Shared runtime arguments
# =======================
IG_STEPS=5
THRESHOLD=0.000
PARALLEL_GPUS="0,1"
PYTHON_BIN="python"
SCRIPT_PATH="neuron_test.py"

# Optional environment setup (same style as train.sh)
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
set +u
source activate SVD
set -u
export CUDA_VISIBLE_DEVICES=0,1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888

run_dir="delta_only_sweep_output"
mkdir -p "${run_dir}/logs"
rm -f "${run_dir}/logs/"*.log

report_txt="${run_dir}/2delta_only_sweep_report.txt"
summary_tsv="${run_dir}/2delta_only_sweep_summary.tsv"

cat > "${report_txt}" <<EOF
Delta-only sweep (Hilton concept disabled):
  enable_Delta=true
  enable_Hilton=false
  ig_steps=${IG_STEPS}
  threshold=${THRESHOLD}
EOF

printf "run_id\tdelta_neuron_count\tdelta_multiplier\thit_delta\thit_united\thit_american\thit_southwest\thit_spirit\thit_hilton\thit_hawaiian\traw_log\n" > "${summary_tsv}"

run_id=0
for delta_count in "${DELTA_NEURON_COUNTS[@]}"; do
  for delta_mult in "${DELTA_MULTIPLIERS[@]}"; do
    run_id=$((run_id + 1))
    log_file="${run_dir}/logs/run_${run_id}_dc${delta_count}_dm${delta_mult}.log"

    echo "[Run ${run_id}] delta_neuron_count=${delta_count}, delta_multiplier=${delta_mult}"

    cmd=(
      "${PYTHON_BIN}" "${SCRIPT_PATH}"
      --enable_Delta
      --ig_steps "${IG_STEPS}"
      --delta-neuron-count "${delta_count}"
      --delta-multiplier "${delta_mult}"
      --parallel-gpus "${PARALLEL_GPUS}"
      --delta-score-mode contrastive
      --threshold "${THRESHOLD}"
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
hit_united = int(bool(re.search(r"\bunited\b", result_lower)))
hit_american = int(bool(re.search(r"\bamerican\b", result_lower)))
hit_southwest = int(bool(re.search(r"\bsouthwest\b", result_lower)))
hit_spirit = int(bool(re.search(r"\bspirit\b", result_lower)))
hit_hilton = int(bool(re.search(r"\bhilton\b", result_lower)))
hit_hawaiian = int("hawaiian airlines" in result_lower)

print(f"HIT_DELTA={hit_delta}")
print(f"HIT_UNITED={hit_united}")
print(f"HIT_AMERICAN={hit_american}")
print(f"HIT_SOUTHWEST={hit_southwest}")
print(f"HIT_SPIRIT={hit_spirit}")
print(f"HIT_HILTON={hit_hilton}")
print(f"HIT_HAWAIIAN={hit_hawaiian}")
print("RESULT_BLOCK_BEGIN")
print(result_block)
print("RESULT_BLOCK_END")
PY
    )"

    hit_delta="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_DELTA=/{print $2}')"
    hit_united="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_UNITED=/{print $2}')"
    hit_american="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_AMERICAN=/{print $2}')"
    hit_southwest="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_SOUTHWEST=/{print $2}')"
    hit_spirit="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_SPIRIT=/{print $2}')"
    hit_hilton="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_HILTON=/{print $2}')"
    hit_hawaiian="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_HAWAIIAN=/{print $2}')"
    result_block="$(printf '%s\n' "${parse_output}" | awk '/^RESULT_BLOCK_BEGIN$/{flag=1;next}/^RESULT_BLOCK_END$/{flag=0}flag')"

    {
      echo ""
      echo "------------------------------------------------------------"
      echo "Run ${run_id}"
      echo "delta_neuron_count=${delta_count}"
      echo "delta_multiplier=${delta_mult}"
      echo "${result_block}"
      echo "------------------------------------------------------------"
    } >> "${report_txt}"

    printf "%s\t%s\t%.2f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${run_id}" \
      "${delta_count}" \
      "${delta_mult}" \
      "${hit_delta}" \
      "${hit_united}" \
      "${hit_american}" \
      "${hit_southwest}" \
      "${hit_spirit}" \
      "${hit_hilton}" \
      "${hit_hawaiian}" \
      "${log_file}" >> "${summary_tsv}"
  done
done

echo ""
echo "Delta-only sweep finished at: $(date)"
echo "Report: ${report_txt}"
echo "Summary TSV: ${summary_tsv}"
echo ""
echo "Recommended candidates (hit_delta=1 and competitors=0):"
awk -F'\t' 'NR==1 || ($4==1 && $5==0 && $6==0 && $7==0 && $8==0)' "${summary_tsv}"

echo "Done. Please check:"
echo "  ${report_txt}"
echo "  ${summary_tsv}"
