#!/bin/bash
set -uo pipefail
# GPU
GPU_ID=7
export PYTORCH_ALLOC_CONF=expandable_segments:True
# =======================
# Fixed multiplier settings
# =======================
MULTIPLIER_1=2.0
MULTIPLIER_2=2.0

# =======================
# Neuron-count sweep parameters
# =======================
TOP_K_1=(0 100 200 300 400 500 600 700 800)
# TOP_K_1=(0)
TOP_K_2=(0 100 200 300 400 500 600 700 800)

# TOP_K_2=(0 50)

# =======================
# Shared runtime arguments
# =======================

COMBO_PRESET_ID=9
IG_STEPS=20
THRESHOLD=0.000
PARALLEL_GPUS="0"
PYTHON_BIN="python"
SCRIPT_PATH="neuron_test.py"
ATTR_CACHE_DIR="attr_cache_log_fixed"
# 默认 0-based prompt 索引列表；若 combo 有专用配置则会自动覆盖
DEFAULT_PROMPT_LIST=(0 1 2 3 4)
PROMPT_LIST=("${DEFAULT_PROMPT_LIST[@]}")

MAX_NEW_TOKENS=1536
export CUDA_VISIBLE_DEVICES="${GPU_ID}"


# Resolve the active combo and brand names from COMBO_PRESET_ID
readarray -t COMBO_INFO < <(
  python - "${COMBO_PRESET_ID}" <<'PY'
import sys
from src.config import COMBO_PRESETS, CONCEPT_CONFIGS
from src.new_prompts import COMBO_PROMPT_LISTS, DEFAULT_PROMPT_LIST

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
prompt_list = COMBO_PROMPT_LISTS.get(combo_key, DEFAULT_PROMPT_LIST)

print(combo_key)
print(concept_1)
print(concept_2)
print(keyword_1)
print(keyword_2)
print(" ".join(str(x) for x in prompt_list))
PY
)

COMBO_KEY="${COMBO_INFO[0]}"
BRAND_1="${COMBO_INFO[1]}"
BRAND_2="${COMBO_INFO[2]}"
KEYWORD_1="${COMBO_INFO[3]}"
KEYWORD_2="${COMBO_INFO[4]}"
PROMPT_LIST_RAW="${COMBO_INFO[5]:-}"

if [[ -n "${PROMPT_LIST_RAW}" ]]; then
  read -r -a PROMPT_LIST <<< "${PROMPT_LIST_RAW}"
else
  PROMPT_LIST=("${DEFAULT_PROMPT_LIST[@]}")
fi


## 结果存储路径!!!
run_root="logp_token_${BRAND_1}_m${MULTIPLIER_1}"
mkdir -p "${run_root}"


# Freeze code snapshot once for all prompts, so later edits
# to neuron_test.py/src do not affect in-flight runs.
snapshot_dir="${run_root}/code_snapshot"
mkdir -p "${snapshot_dir}"
cp "${SCRIPT_PATH}" "${snapshot_dir}/$(basename "${SCRIPT_PATH}")"
cp -r "src" "${snapshot_dir}/src"
SCRIPT_PATH="${snapshot_dir}/$(basename "${SCRIPT_PATH}")"

overall_report_txt="${run_root}/report_all_prompts.txt"
{
  echo "Sweep started at: $(date)"
  echo "combo_preset_id=${COMBO_PRESET_ID}"
  echo "combo_key=${COMBO_KEY}"
  echo "brand_1=${BRAND_1}"
  echo "brand_2=${BRAND_2}"
  echo "keyword_1=${KEYWORD_1}"
  echo "keyword_2=${KEYWORD_2}"
  echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
  echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
  echo "prompt_list=${PROMPT_LIST[*]}"
  echo "gpu_id=${GPU_ID}"
  echo "${BRAND_1}_top_k=${TOP_K_1[*]}"
  echo "${BRAND_2}_top_k=${TOP_K_2[*]}"
  echo "attribution_cache_dir=${ATTR_CACHE_DIR}"
  echo "code_snapshot=${snapshot_dir}"
} > "${overall_report_txt}"

overall_success_runs=0
overall_failed_runs=0
overall_total_runs=0
overall_failed_run_msgs=()
prompt_avg_csv="${run_root}/prompt_hit_avg.csv"
declare -A run_top_k_1_map
declare -A run_top_k_2_map
declare -A run_hit_1_sum_map
declare -A run_hit_2_sum_map
declare -A run_hit_count_success_map
total_runs_per_prompt=$(( ${#TOP_K_1[@]} * ${#TOP_K_2[@]} ))

for prompt_index in "${PROMPT_LIST[@]}"; do
  run_dir="${run_root}/p${prompt_index}"
  mkdir -p "${run_dir}/logs"
  rm -f "${run_dir}/logs/"*.log

  report_txt="${run_dir}/report_${prompt_index}.txt"
  summary_csv="${run_dir}/summary_${prompt_index}.csv"

  cat > "${report_txt}" <<EOF
Fixed params:
  combo_preset_id=${COMBO_PRESET_ID}
  combo_key=${COMBO_KEY}
  brand_1=${BRAND_1}
  brand_2=${BRAND_2}
  keyword_1=${KEYWORD_1}
  keyword_2=${KEYWORD_2}
  ${BRAND_1}_multiplier=${MULTIPLIER_1}
  ${BRAND_2}_multiplier=${MULTIPLIER_2}
  prompt_index=${prompt_index}
  gpu_id=${GPU_ID}
  ${BRAND_1}_top_k=${TOP_K_1[*]}
  ${BRAND_2}_top_k=${TOP_K_2[*]}
  attribution_cache_dir=${ATTR_CACHE_DIR}
  code_snapshot=${snapshot_dir}
EOF

  printf "run_id\t%s_top_k\t%s_top_k\t%s_multiplier\t%s_multiplier\thit_%s\thit_%s\n" \
    "${BRAND_1}" "${BRAND_2}" "${BRAND_1}" "${BRAND_2}" "${KEYWORD_1}" "${KEYWORD_2}" > "${summary_csv}"

  echo ""
  echo "==================== Prompt ${prompt_index} ===================="

  run_id=0
  total_runs="${total_runs_per_prompt}"
  success_runs=0
  failed_runs=0
  failed_run_msgs=()
  sum_hit_1=0
  sum_hit_2=0

  for top_k_2 in "${TOP_K_2[@]}"; do
    for top_k_1 in "${TOP_K_1[@]}"; do
      run_id=$((run_id + 1))
      run_top_k_1_map["${run_id}"]="${top_k_1}"
      run_top_k_2_map["${run_id}"]="${top_k_2}"
      log_file="${run_dir}/logs/run_${run_id}_k1${top_k_1}_k2${top_k_2}_m1${MULTIPLIER_1}_m2${MULTIPLIER_2}.log"

      echo "[Prompt ${prompt_index}] [Run ${run_id}] ${BRAND_1}_top_k=${top_k_1}, ${BRAND_2}_top_k=${top_k_2}, ${BRAND_1}_multiplier=${MULTIPLIER_1}, ${BRAND_2}_multiplier=${MULTIPLIER_2}"

      cmd=(
        "${PYTHON_BIN}" "${SCRIPT_PATH}"
        --combo-preset "${COMBO_PRESET_ID}"
        --enable_1
        --enable_2
        --ig_steps "${IG_STEPS}"
        --top_k_1 "${top_k_1}"
        --multiplier_1 "${MULTIPLIER_1}"
        --top_k_2 "${top_k_2}"
        --multiplier_2 "${MULTIPLIER_2}"
        --parallel-gpus "${PARALLEL_GPUS}"
        --score_mode_1 contrastive
        --score_mode_2 contrastive
        --threshold "${THRESHOLD}"
        --attribution-cache-dir "${ATTR_CACHE_DIR}"
        --intervention_layer -1
        --prompt-index "${prompt_index}"
        --unified-hook
        --mind_bridge
        --max-new-tokens "${MAX_NEW_TOKENS}"
      )

      # Save full stdout/stderr for each run. On failure, record and continue.
      if ! "${cmd[@]}" > "${log_file}" 2>&1; then
        failed_runs=$((failed_runs + 1))
        failed_run_msgs+=("Run ${run_id} (k1=${top_k_1}, k2=${top_k_2}): python command failed")
        echo "  [WARN] Run ${run_id} failed, skip and continue. See log: ${log_file}"
        {
          echo ""
          echo "------------------------------------------------------------"
          echo "Run ${run_id}"
          echo "${BRAND_1}_top_k=${top_k_1}"
          echo "${BRAND_2}_top_k=${top_k_2}"
          echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
          echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
          echo "prompt_index=${prompt_index}"
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
        failed_run_msgs+=("Run ${run_id} (k1=${top_k_1}, k2=${top_k_2}): parse script failed")
        echo "  [WARN] Run ${run_id} parse failed, skip and continue. See log: ${log_file}"
        {
          echo ""
          echo "------------------------------------------------------------"
          echo "Run ${run_id}"
          echo "${BRAND_1}_top_k=${top_k_1}"
          echo "${BRAND_2}_top_k=${top_k_2}"
          echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
          echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
          echo "prompt_index=${prompt_index}"
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
        failed_run_msgs+=("Run ${run_id} (k1=${top_k_1}, k2=${top_k_2}): invalid parse output (hit_1=${hit_1:-N/A}, hit_2=${hit_2:-N/A})")
        echo "  [WARN] Run ${run_id} parse output invalid, skip and continue. See log: ${log_file}"
        {
          echo ""
          echo "------------------------------------------------------------"
          echo "Run ${run_id}"
          echo "${BRAND_1}_top_k=${top_k_1}"
          echo "${BRAND_2}_top_k=${top_k_2}"
          echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
          echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
          echo "prompt_index=${prompt_index}"
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
        echo "${BRAND_1}_top_k=${top_k_1}"
        echo "${BRAND_2}_top_k=${top_k_2}"
        echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
        echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
        echo "prompt_index=${prompt_index}"
        echo "prompt=${prompt_text}"
        echo "gpu_id=${GPU_ID}"
        echo "${result_block}"
        echo "hit_${KEYWORD_1}=${hit_1}, hit_${KEYWORD_2}=${hit_2}"
        echo "------------------------------------------------------------"
      } >> "${report_txt}"

      printf "%s\t%s\t%s\t%.2f\t%.2f\t%s\t%s\n" \
        "${run_id}" \
        "${top_k_1}" \
        "${top_k_2}" \
        "${MULTIPLIER_1}" \
        "${MULTIPLIER_2}" \
        "${hit_1}" \
        "${hit_2}" >> "${summary_csv}"
      sum_hit_1=$((sum_hit_1 + hit_1))
      sum_hit_2=$((sum_hit_2 + hit_2))
      success_runs=$((success_runs + 1))
      run_hit_1_sum_map["${run_id}"]=$(( ${run_hit_1_sum_map["${run_id}"]:-0} + hit_1 ))
      run_hit_2_sum_map["${run_id}"]=$(( ${run_hit_2_sum_map["${run_id}"]:-0} + hit_2 ))
      run_hit_count_success_map["${run_id}"]=$(( ${run_hit_count_success_map["${run_id}"]:-0} + 1 ))
    done
  done

  if (( success_runs > 0 )); then
    avg_hit_1="$(awk -v s="${sum_hit_1}" -v n="${success_runs}" 'BEGIN{printf "%.6f", s/n}')"
    avg_hit_2="$(awk -v s="${sum_hit_2}" -v n="${success_runs}" 'BEGIN{printf "%.6f", s/n}')"
    avg_hit_count="$(awk -v s1="${sum_hit_1}" -v s2="${sum_hit_2}" -v n="${success_runs}" 'BEGIN{printf "%.6f", (s1+s2)/n}')"
  else
    avg_hit_1="0.000000"
    avg_hit_2="0.000000"
    avg_hit_count="0.000000"
  fi

  overall_total_runs=$((overall_total_runs + total_runs))
  overall_success_runs=$((overall_success_runs + success_runs))
  overall_failed_runs=$((overall_failed_runs + failed_runs))
  if (( failed_runs > 0 )); then
    for msg in "${failed_run_msgs[@]}"; do
      overall_failed_run_msgs+=("prompt=${prompt_index}: ${msg}")
    done
  fi

  echo ""
  echo "Prompt ${prompt_index} finished at: $(date)"
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
    echo "prompt_index=${prompt_index}"
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

  {
    echo ""
    echo "-------------------- Prompt ${prompt_index} --------------------"
    echo "report=${report_txt}"
    echo "summary_csv=${summary_csv}"
    echo "total_runs=${total_runs}"
    echo "success_runs=${success_runs}"
    echo "failed_runs=${failed_runs}"
    echo "avg_hit_${KEYWORD_1}=${avg_hit_1}"
    echo "avg_hit_${KEYWORD_2}=${avg_hit_2}"
    echo "hit_count=${avg_hit_count}"
    if (( failed_runs > 0 )); then
      echo "failed_run_list:"
      for msg in "${failed_run_msgs[@]}"; do
        echo "  - ${msg}"
      done
    fi
  } >> "${overall_report_txt}"
done

printf "run_id\t%s_top_k\t%s_top_k\t%s_multiplier\t%s_multiplier\thit_%s_avg\thit_%s_avg\n" \
  "${BRAND_1}" "${BRAND_2}" "${BRAND_1}" "${BRAND_2}" "${KEYWORD_1}" "${KEYWORD_2}" > "${prompt_avg_csv}"
for run_id in $(seq 1 "${total_runs_per_prompt}"); do
  run_success_prompts="${run_hit_count_success_map["${run_id}"]:-0}"
  if (( run_success_prompts > 0 )); then
    run_avg_hit_1="$(awk -v s="${run_hit_1_sum_map["${run_id}"]}" -v n="${run_success_prompts}" 'BEGIN{printf "%.6f", s/n}')"
    run_avg_hit_2="$(awk -v s="${run_hit_2_sum_map["${run_id}"]}" -v n="${run_success_prompts}" 'BEGIN{printf "%.6f", s/n}')"
  else
    run_avg_hit_1="0.000000"
    run_avg_hit_2="0.000000"
  fi
  printf "%s\t%s\t%s\t%.2f\t%.2f\t%s\t%s\n" \
    "${run_id}" \
    "${run_top_k_1_map["${run_id}"]}" \
    "${run_top_k_2_map["${run_id}"]}" \
    "${MULTIPLIER_1}" \
    "${MULTIPLIER_2}" \
    "${run_avg_hit_1}" \
    "${run_avg_hit_2}" >> "${prompt_avg_csv}"
done

echo ""
echo "All prompts sweep finished at: $(date)"
echo "Overall report: ${overall_report_txt}"
echo "Total planned runs: ${overall_total_runs}"
echo "Successful runs: ${overall_success_runs}"
echo "Failed runs: ${overall_failed_runs}"
if (( overall_failed_runs > 0 )); then
  echo "Failed run list:"
  for msg in "${overall_failed_run_msgs[@]}"; do
    echo "  - ${msg}"
  done
fi
echo ""
echo "Done. Please check:"
echo "  ${overall_report_txt}"
echo "  ${run_root}/p*/summary_*.csv"
echo "  ${prompt_avg_csv}"
