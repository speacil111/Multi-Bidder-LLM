#!/bin/bash
set -uo pipefail

# =======================
# Fixed neuron-count settings
# =======================
ATTR_SUM_1=3.0
ATTR_SUM_2=3.0

# =======================
# Multiplier sweep parameters
# =======================
MULTIPLIER_1_LIST=(2.00 2.50 3.00 3.50 4.00 4.50 5.00)
MULTIPLIER_2_LIST=(2.00 2.50 3.00 3.50 4.00 4.50 5.00)

# =======================
# Shared runtime arguments
# =======================
COMBO_PRESET_ID=4
IG_STEPS=20
THRESHOLD=0.000
PARALLEL_GPUS="0"
PYTHON_BIN="python"
SCRIPT_PATH="neuron_test.py"
ATTR_CACHE_DIR="attr_score_cache"
PROMPT_INDEX=3
GPU_ID=6
MAX_NEW_TOKENS=1536
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Resolve the active combo and brand names from COMBO_PRESET_ID
readarray -t COMBO_INFO < <(
  python - "${COMBO_PRESET_ID}" <<'PY'
import sys
from src.config import COMBO_PRESETS, CONCEPT_CONFIGS

combo_id = int(sys.argv[1])
combo_keys = list(COMBO_PRESETS.keys())
if combo_id < 0 or combo_id >= len(combo_keys):
    raise SystemExit(
        f"combo preset id out of range: {combo_id}; valid=[0, {len(combo_keys)-1}]"
    )

combo_key = combo_keys[combo_id]
concept_1, concept_2 = COMBO_PRESETS[combo_key]
keyword_1 = CONCEPT_CONFIGS[concept_1]["positive_word"]
keyword_2 = CONCEPT_CONFIGS[concept_2]["positive_word"]

print(combo_key)
print(concept_1)
print(concept_2)
print(keyword_1)
print(keyword_2)
PY
)

COMBO_KEY="${COMBO_INFO[0]}"
BRAND_1="${COMBO_INFO[1]}"
BRAND_2="${COMBO_INFO[2]}"
KEYWORD_1="${COMBO_INFO[3]}"
KEYWORD_2="${COMBO_INFO[4]}"


run_dir="multiplier_${BRAND_1}p${PROMPT_INDEX}_s1${ATTR_SUM_1}_s2${ATTR_SUM_2}"
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
summary_csv="${run_dir}/summary_${PROMPT_INDEX}.csv"

cat > "${report_txt}" <<EOF
Fixed params:
  combo_preset_id=${COMBO_PRESET_ID}
  combo_key=${COMBO_KEY}
  brand_1=${BRAND_1}
  brand_2=${BRAND_2}
  keyword_1=${KEYWORD_1}
  keyword_2=${KEYWORD_2}
  ${BRAND_1}_attr_sum=${ATTR_SUM_1}
  ${BRAND_2}_attr_sum=${ATTR_SUM_2}
  ${BRAND_1}_multiplier_list=${MULTIPLIER_1_LIST[*]}
  ${BRAND_2}_multiplier_list=${MULTIPLIER_2_LIST[*]}
  prompt_index=${PROMPT_INDEX}
  gpu_id=${GPU_ID}
  attribution_cache_dir=${ATTR_CACHE_DIR}
  code_snapshot=${snapshot_dir}
EOF

printf "run_id\t%s_attr_sum\t%s_attr_sum\t%s_multiplier\t%s_multiplier\thit_%s\thit_%s\n" \
  "${BRAND_1}" "${BRAND_2}" "${BRAND_1}" "${BRAND_2}" "${KEYWORD_1}" "${KEYWORD_2}" > "${summary_csv}"

run_id=0
total_runs=$(( ${#MULTIPLIER_1_LIST[@]} * ${#MULTIPLIER_2_LIST[@]} ))
success_runs=0
failed_runs=0
failed_run_msgs=()
for multiplier_2 in "${MULTIPLIER_2_LIST[@]}"; do
  for multiplier_1 in "${MULTIPLIER_1_LIST[@]}"; do
    run_id=$((run_id + 1))
    log_file="${run_dir}/logs/run_${run_id}_s1${ATTR_SUM_1}_s2${ATTR_SUM_2}_m1${multiplier_1}_m2${multiplier_2}.log"

    echo "[Run ${run_id}] ${BRAND_1}_attr_sum=${ATTR_SUM_1}, ${BRAND_2}_attr_sum=${ATTR_SUM_2}, ${BRAND_1}_multiplier=${multiplier_1}, ${BRAND_2}_multiplier=${multiplier_2}"

    cmd=(
      "${PYTHON_BIN}" "${SCRIPT_PATH}"
      --combo-preset "${COMBO_PRESET_ID}"
      --enable_1
      --enable_2
      --ig_steps "${IG_STEPS}"
      --attr_sum_1 "${ATTR_SUM_1}"
      --multiplier_1 "${multiplier_1}"
      --attr_sum_2 "${ATTR_SUM_2}"
      --multiplier_2 "${multiplier_2}"
      --parallel-gpus "${PARALLEL_GPUS}"
      --score_mode_1 contrastive
      --score_mode_2 contrastive
      --threshold "${THRESHOLD}"
      --attribution-cache-dir "${ATTR_CACHE_DIR}"
      --intervention_layer -1
      --prompt-index "${PROMPT_INDEX}"
      --unified-hook
      --max-new-tokens "${MAX_NEW_TOKENS}"
    )

    # Save full stdout/stderr for each run. On failure, record and continue.
    if ! "${cmd[@]}" > "${log_file}" 2>&1; then
      failed_runs=$((failed_runs + 1))
      failed_run_msgs+=("Run ${run_id} (m1=${multiplier_1}, m2=${multiplier_2}): python command failed")
      echo "  [WARN] Run ${run_id} failed, skip and continue. See log: ${log_file}"
      {
        echo ""
        echo "------------------------------------------------------------"
        echo "Run ${run_id}"
        echo "${BRAND_1}_attr_sum=${ATTR_SUM_1}"
        echo "${BRAND_2}_attr_sum=${ATTR_SUM_2}"
        echo "${BRAND_1}_multiplier=${multiplier_1}"
        echo "${BRAND_2}_multiplier=${multiplier_2}"
        echo "prompt_index=${PROMPT_INDEX}"
        echo "gpu_id=${GPU_ID}"
        echo "status=FAILED (python command failed)"
        echo "log_file=${log_file}"
        echo "------------------------------------------------------------"
      } >> "${report_txt}"
      continue
    fi

    # Parse key output and append to report + summary
    if ! parse_output="$(
      python - "${log_file}" "${KEYWORD_1}" "${KEYWORD_2}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
keyword_1 = sys.argv[2].lower()
keyword_2 = sys.argv[3].lower()
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
hit_1 = len(re.findall(rf"\b{re.escape(keyword_1)}\b", result_lower))
hit_2 = len(re.findall(rf"\b{re.escape(keyword_2)}\b", result_lower))

prompt_text = ""
for line in lines:
    if line.startswith("prompt:"):
        prompt_text = line.split("prompt:", 1)[1].strip()
        break

print(f"PROMPT_TEXT={prompt_text}")
print(f"HIT_1={hit_1}")
print(f"HIT_2={hit_2}")
print("RESULT_BLOCK_BEGIN")
print(result_block)
print("RESULT_BLOCK_END")
PY
    )"; then
      failed_runs=$((failed_runs + 1))
      failed_run_msgs+=("Run ${run_id} (m1=${multiplier_1}, m2=${multiplier_2}): parse script failed")
      echo "  [WARN] Run ${run_id} parse failed, skip and continue. See log: ${log_file}"
      {
        echo ""
        echo "------------------------------------------------------------"
        echo "Run ${run_id}"
        echo "${BRAND_1}_attr_sum=${ATTR_SUM_1}"
        echo "${BRAND_2}_attr_sum=${ATTR_SUM_2}"
        echo "${BRAND_1}_multiplier=${multiplier_1}"
        echo "${BRAND_2}_multiplier=${multiplier_2}"
        echo "prompt_index=${PROMPT_INDEX}"
        echo "gpu_id=${GPU_ID}"
        echo "status=FAILED (parse script failed)"
        echo "log_file=${log_file}"
        echo "------------------------------------------------------------"
      } >> "${report_txt}"
      continue
    fi

    prompt_text="$(printf '%s\n' "${parse_output}" | awk -F= '/^PROMPT_TEXT=/{sub(/^PROMPT_TEXT=/, ""); print; exit}')"
    hit_1="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_1=/{print $2}')"
    hit_2="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_2=/{print $2}')"
    result_block="$(printf '%s\n' "${parse_output}" | awk '/^RESULT_BLOCK_BEGIN$/{flag=1;next}/^RESULT_BLOCK_END$/{flag=0}flag')"
    if ! [[ "${hit_1}" =~ ^[0-9]+$ && "${hit_2}" =~ ^[0-9]+$ ]]; then
      failed_runs=$((failed_runs + 1))
      failed_run_msgs+=("Run ${run_id} (m1=${multiplier_1}, m2=${multiplier_2}): invalid parse output (hit_1=${hit_1:-N/A}, hit_2=${hit_2:-N/A})")
      echo "  [WARN] Run ${run_id} parse output invalid, skip and continue. See log: ${log_file}"
      {
        echo ""
        echo "------------------------------------------------------------"
        echo "Run ${run_id}"
        echo "${BRAND_1}_attr_sum=${ATTR_SUM_1}"
        echo "${BRAND_2}_attr_sum=${ATTR_SUM_2}"
        echo "${BRAND_1}_multiplier=${multiplier_1}"
        echo "${BRAND_2}_multiplier=${multiplier_2}"
        echo "prompt_index=${PROMPT_INDEX}"
        echo "gpu_id=${GPU_ID}"
        echo "status=FAILED (invalid parse output)"
        echo "hit_${KEYWORD_1}=${hit_1:-N/A}, hit_${KEYWORD_2}=${hit_2:-N/A}"
        echo "log_file=${log_file}"
        echo "------------------------------------------------------------"
      } >> "${report_txt}"
      continue
    fi

    {
      echo ""
      echo "------------------------------------------------------------"
      echo "Run ${run_id}"
      echo "${BRAND_1}_attr_sum=${ATTR_SUM_1}"
      echo "${BRAND_2}_attr_sum=${ATTR_SUM_2}"
      echo "${BRAND_1}_multiplier=${multiplier_1}"
      echo "${BRAND_2}_multiplier=${multiplier_2}"
      echo "prompt_index=${PROMPT_INDEX}"
      echo "prompt=${prompt_text}"
      echo "gpu_id=${GPU_ID}"
      echo "${result_block}"
      echo "hit_${KEYWORD_1}=${hit_1}, hit_${KEYWORD_2}=${hit_2}"
      echo "------------------------------------------------------------"
    } >> "${report_txt}"

    printf "%s\t%s\t%s\t%.2f\t%.2f\t%s\t%s\n" \
      "${run_id}" \
      "${ATTR_SUM_1}" \
      "${ATTR_SUM_2}" \
      "${multiplier_1}" \
      "${multiplier_2}" \
      "${hit_1}" \
      "${hit_2}" >> "${summary_csv}"
    success_runs=$((success_runs + 1))
  done
done

echo ""
echo "Sweep finished at: $(date)"
echo "Report: ${report_txt}"
echo "Summary csv: ${summary_csv}"
echo "Total planned runs: ${total_runs}"
echo "Successful runs: ${success_runs}"
echo "Failed runs: ${failed_runs}"
if (( failed_runs > 0 )); then
  echo "Failed run list:"
  for msg in "${failed_run_msgs[@]}"; do
    echo "  - ${msg}"
  done
fi
echo ""

{
  echo ""
  echo "==================== Sweep Summary ===================="
  echo "total_runs=${total_runs}"
  echo "success_runs=${success_runs}"
  echo "failed_runs=${failed_runs}"
  if (( failed_runs > 0 )); then
    echo "failed_run_list:"
    for msg in "${failed_run_msgs[@]}"; do
      echo "  - ${msg}"
    done
  fi
  echo "======================================================="
} >> "${report_txt}"

echo "Done. Please check:"
echo "  ${report_txt}"
echo "  ${summary_csv}"
