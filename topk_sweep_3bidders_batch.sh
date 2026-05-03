#!/bin/bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: bash topk_sweep_3bidders_batch.sh [--gpu-id N] [--combo-preset SPEC] [options]

Options:
  --g N, --gpu-id N     Override GPU_ID. Default: 0
  --c SPEC, --combo-preset SPEC, --combo-preset-id SPEC
                        3-bidder combo key or 0-based id in src/3_config.py. Default: 0
  --model-path PATH     Override MODEL_PATH
  --attr-cache-dir DIR, --attribution-cache-dir DIR
                        Override ATTR_CACHE_DIR
  --result-root DIR     Override first-level output dir. Default: batch_results_<model_tag>_3bidders
  --prompt-list LIST    Override prompt indexes, e.g. "0,1,2" or "0 1 2"
  -h, --help            Show this help message

Examples:
  bash topk_sweep_3bidders_batch.sh --g 1 --c 0
  bash topk_sweep_3bidders_batch.sh --gpu-id 1 --combo-preset delta_marriott_visa
EOF
}

format_duration() {
  local total_seconds="$1"
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

resolve_model_tag() {
  local model_path="$1"
  local model_path_lower="${model_path,,}"
  local model_base

  if [[ "${model_path_lower}" == *ds_r1_8b* ]]; then
    printf "DS"
    return
  fi
  if [[ "${model_path_lower}" == *qwen3_4b* || "${model_path_lower}" == *qwen3-4b* ]]; then
    printf "Qwen"
    return
  fi
  if [[ "${model_path_lower}" == *llama*8b* ]]; then
    printf "Llama"
    return
  fi

  model_base="${model_path##*/}"
  model_base="${model_base//[^[:alnum:]_-]/_}"
  if [[ -n "${model_base}" ]]; then
    printf "%s" "${model_base}"
  else
    printf "Model"
  fi
}

parse_prompt_list_spec() {
  local spec="$1"
  local token start end i
  PROMPT_LIST=()
  spec="${spec//,/ }"
  for token in ${spec}; do
    if [[ "${token}" =~ ^[0-9]+-[0-9]+$ ]]; then
      start="${token%-*}"
      end="${token#*-}"
      if (( start > end )); then
        echo "[ERROR] invalid --prompt-list range: ${token}" >&2
        exit 1
      fi
      for (( i=start; i<=end; i++ )); do
        PROMPT_LIST+=("${i}")
      done
    elif [[ "${token}" =~ ^[0-9]+$ ]]; then
      PROMPT_LIST+=("${token}")
    else
      echo "[ERROR] invalid --prompt-list token: ${token}" >&2
      exit 1
    fi
  done
  if (( ${#PROMPT_LIST[@]} == 0 )); then
    echo "[ERROR] --prompt-list cannot be empty" >&2
    exit 1
  fi
}

GPU_ID=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

MULTIPLIER_1=2.0
MULTIPLIER_2=2.0
MULTIPLIER_3=2.0

TOP_K_1=(0 100 200 300 400 500 600 700 800)
TOP_K_2=(0 100 200 300 400 500 600 700 800)
TOP_K_3=(0 100 200 300 400 500 600 700 800)

COMBO_PRESET="0"
IG_STEPS=20
THRESHOLD=0.000
PARALLEL_GPUS="${GPU_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="neuron_test.py"
MODEL_PATH="../Qwen3-4B"
ATTR_CACHE_DIR="${ATTR_CACHE_DIR:-attr_cache_qwen}"
RESULT_ROOT=""
PROMPT_LIST=(0 1 2)
MAX_NEW_TOKENS=1536

while [[ $# -gt 0 ]]; do
  case "$1" in
    --g|--gpu-id)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      GPU_ID="$2"
      shift 2
      ;;
    --g=*|--gpu-id=*)
      GPU_ID="${1#*=}"
      shift
      ;;
    --c|--combo-preset|--combo-preset-id)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      COMBO_PRESET="$2"
      shift 2
      ;;
    --c=*|--combo-preset=*|--combo-preset-id=*)
      COMBO_PRESET="${1#*=}"
      shift
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "[ERROR] --model-path requires a value" >&2; usage >&2; exit 1; }
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --attr-cache-dir|--attribution-cache-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      ATTR_CACHE_DIR="$2"
      shift 2
      ;;
    --attr-cache-dir=*|--attribution-cache-dir=*)
      ATTR_CACHE_DIR="${1#*=}"
      shift
      ;;
    --result-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --result-root requires a value" >&2; usage >&2; exit 1; }
      RESULT_ROOT="$2"
      shift 2
      ;;
    --result-root=*)
      RESULT_ROOT="${1#*=}"
      shift
      ;;
    --prompt-list|--prompts)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      parse_prompt_list_spec "$2"
      shift 2
      ;;
    --prompt-list=*|--prompts=*)
      parse_prompt_list_spec "${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

PARALLEL_GPUS="${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

script_start_epoch="$(date +%s)"
script_start_time="$(date '+%Y-%m-%d %H:%M:%S')"

COMBO_INFO=()
while IFS= read -r line; do
  COMBO_INFO+=("${line}")
done < <(
  "${PYTHON_BIN}" - "${COMBO_PRESET}" <<'PY'
import importlib
import sys
from src.config import CONCEPT_CONFIGS

raw = sys.argv[1]
three_config = importlib.import_module("src.3_config")
presets = three_config.THREE_BIDDER_COMBO_PRESETS
combo_keys = list(presets.keys())

if raw in presets:
    combo_key = raw
elif raw.isdigit():
    combo_id = int(raw)
    if combo_id < 0 or combo_id >= len(combo_keys):
        raise SystemExit(
            f"3-bidder combo id out of range: {combo_id}; valid=[0, {len(combo_keys)-1}]"
        )
    combo_key = combo_keys[combo_id]
else:
    raise SystemExit(
        f"unknown 3-bidder combo preset: {raw}; valid keys: {', '.join(combo_keys)}"
    )

concept_1, concept_2, concept_3 = presets[combo_key]
keyword_1 = CONCEPT_CONFIGS[concept_1]["positive_word"]
keyword_2 = CONCEPT_CONFIGS[concept_2]["positive_word"]
keyword_3 = CONCEPT_CONFIGS[concept_3]["positive_word"]

print(combo_key)
print(concept_1)
print(concept_2)
print(concept_3)
print(keyword_1)
print(keyword_2)
print(keyword_3)
PY
)

if (( ${#COMBO_INFO[@]} != 7 )); then
  echo "[ERROR] Failed to resolve 3-bidder combo: ${COMBO_PRESET}" >&2
  exit 1
fi

COMBO_KEY="${COMBO_INFO[0]}"
BRAND_1="${COMBO_INFO[1]}"
BRAND_2="${COMBO_INFO[2]}"
BRAND_3="${COMBO_INFO[3]}"
KEYWORD_1="${COMBO_INFO[4]}"
KEYWORD_2="${COMBO_INFO[5]}"
KEYWORD_3="${COMBO_INFO[6]}"

model_tag="$(resolve_model_tag "${MODEL_PATH}")"
if [[ -n "${RESULT_ROOT}" ]]; then
  result_root_dir="${RESULT_ROOT%/}"
else
  result_root_dir="batch_results_${model_tag}_3bidders"
fi

run_root="${result_root_dir}/${model_tag}_3bidders_${COMBO_KEY}_m${MULTIPLIER_1}"
mkdir -p "${run_root}"

snapshot_dir="${run_root}/code_snapshot"
mkdir -p "${snapshot_dir}"
cp "${SCRIPT_PATH}" "${snapshot_dir}/$(basename "${SCRIPT_PATH}")"
cp -r "src" "${snapshot_dir}/src"
SCRIPT_PATH="${snapshot_dir}/$(basename "${SCRIPT_PATH}")"

overall_report_txt="${run_root}/report_all_prompts.txt"
{
  echo "Sweep started at: ${script_start_time}"
  echo "combo_preset=${COMBO_PRESET}"
  echo "combo_key=${COMBO_KEY}"
  echo "brand_1=${BRAND_1}"
  echo "brand_2=${BRAND_2}"
  echo "brand_3=${BRAND_3}"
  echo "keyword_1=${KEYWORD_1}"
  echo "keyword_2=${KEYWORD_2}"
  echo "keyword_3=${KEYWORD_3}"
  echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
  echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
  echo "${BRAND_3}_multiplier=${MULTIPLIER_3}"
  echo "prompt_list=${PROMPT_LIST[*]}"
  echo "gpu_id=${GPU_ID}"
  echo "${BRAND_1}_top_k=${TOP_K_1[*]}"
  echo "${BRAND_2}_top_k=${TOP_K_2[*]}"
  echo "${BRAND_3}_top_k=${TOP_K_3[*]}"
  echo "model_path=${MODEL_PATH:-<default>}"
  echo "attribution_cache_dir=${ATTR_CACHE_DIR}"
  echo "result_root_dir=${result_root_dir}"
  echo "run_root=${run_root}"
  echo "code_snapshot=${snapshot_dir}"
} > "${overall_report_txt}"

overall_success_runs=0
overall_failed_runs=0
overall_total_runs=0
overall_failed_run_msgs=()
prompt_avg_csv="${run_root}/prompt_hit_avg.csv"
declare -a run_top_k_1_map
declare -a run_top_k_2_map
declare -a run_top_k_3_map
declare -a run_hit_1_sum_map
declare -a run_hit_2_sum_map
declare -a run_hit_3_sum_map
declare -a run_hit_count_success_map
total_runs_per_prompt=$(( ${#TOP_K_1[@]} * ${#TOP_K_2[@]} * ${#TOP_K_3[@]} ))

for prompt_index in "${PROMPT_LIST[@]}"; do
  run_dir="${run_root}/p${prompt_index}"
  mkdir -p "${run_dir}/logs"
  rm -f "${run_dir}/logs/"*.log

  report_txt="${run_dir}/report_${prompt_index}.txt"
  summary_csv="${run_dir}/summary_${prompt_index}.csv"

  cat > "${report_txt}" <<EOF
Fixed params:
  combo_preset=${COMBO_PRESET}
  combo_key=${COMBO_KEY}
  brand_1=${BRAND_1}
  brand_2=${BRAND_2}
  brand_3=${BRAND_3}
  keyword_1=${KEYWORD_1}
  keyword_2=${KEYWORD_2}
  keyword_3=${KEYWORD_3}
  ${BRAND_1}_multiplier=${MULTIPLIER_1}
  ${BRAND_2}_multiplier=${MULTIPLIER_2}
  ${BRAND_3}_multiplier=${MULTIPLIER_3}
  prompt_index=${prompt_index}
  gpu_id=${GPU_ID}
  ${BRAND_1}_top_k=${TOP_K_1[*]}
  ${BRAND_2}_top_k=${TOP_K_2[*]}
  ${BRAND_3}_top_k=${TOP_K_3[*]}
  model_path=${MODEL_PATH:-<default>}
  attribution_cache_dir=${ATTR_CACHE_DIR}
  result_root_dir=${result_root_dir}
  run_root=${run_root}
  code_snapshot=${snapshot_dir}
EOF

  printf "run_id\t%s_top_k\t%s_top_k\t%s_top_k\t%s_multiplier\t%s_multiplier\t%s_multiplier\thit_%s\thit_%s\thit_%s\n" \
    "${BRAND_1}" "${BRAND_2}" "${BRAND_3}" \
    "${BRAND_1}" "${BRAND_2}" "${BRAND_3}" \
    "${KEYWORD_1}" "${KEYWORD_2}" "${KEYWORD_3}" > "${summary_csv}"

  echo ""
  echo "==================== Prompt ${prompt_index} ===================="

  run_id=0
  total_runs="${total_runs_per_prompt}"
  success_runs=0
  failed_runs=0
  failed_run_msgs=()
  sum_hit_1=0
  sum_hit_2=0
  sum_hit_3=0

  for top_k_3 in "${TOP_K_3[@]}"; do
    for top_k_2 in "${TOP_K_2[@]}"; do
      for top_k_1 in "${TOP_K_1[@]}"; do
        run_id=$((run_id + 1))
        run_top_k_1_map["${run_id}"]="${top_k_1}"
        run_top_k_2_map["${run_id}"]="${top_k_2}"
        run_top_k_3_map["${run_id}"]="${top_k_3}"
        log_file="${run_dir}/logs/run_${run_id}_k1${top_k_1}_k2${top_k_2}_k3${top_k_3}_m1${MULTIPLIER_1}_m2${MULTIPLIER_2}_m3${MULTIPLIER_3}.log"

        echo "[Prompt ${prompt_index}] [Run ${run_id}] ${BRAND_1}_top_k=${top_k_1}, ${BRAND_2}_top_k=${top_k_2}, ${BRAND_3}_top_k=${top_k_3}"

        cmd=(
          "${PYTHON_BIN}" "${SCRIPT_PATH}"
          --combo-preset "${COMBO_KEY}"
          --enable_1
          --enable_2
          --enable_3
          --ig_steps "${IG_STEPS}"
          --top_k_1 "${top_k_1}"
          --multiplier_1 "${MULTIPLIER_1}"
          --top_k_2 "${top_k_2}"
          --multiplier_2 "${MULTIPLIER_2}"
          --top_k_3 "${top_k_3}"
          --multiplier_3 "${MULTIPLIER_3}"
          --parallel-gpus "${PARALLEL_GPUS}"
          --score_mode_1 contrastive
          --score_mode_2 contrastive
          --score_mode_3 contrastive
          --threshold "${THRESHOLD}"
          --attribution-cache-dir "${ATTR_CACHE_DIR}"
          --intervention_layer -1
          --prompt-index "${prompt_index}"
          --unified-hook
          --max-new-tokens "${MAX_NEW_TOKENS}"
        )

        if [[ -n "${MODEL_PATH}" ]]; then
          cmd+=(--model_path "${MODEL_PATH}")
        fi

        if ! "${cmd[@]}" > "${log_file}" 2>&1; then
          failed_runs=$((failed_runs + 1))
          failed_run_msgs+=("Run ${run_id} (k1=${top_k_1}, k2=${top_k_2}, k3=${top_k_3}): python command failed")
          echo "  [WARN] Run ${run_id} failed, skip and continue. See log: ${log_file}"
          {
            echo ""
            echo "------------------------------------------------------------"
            echo "Run ${run_id}"
            echo "${BRAND_1}_top_k=${top_k_1}"
            echo "${BRAND_2}_top_k=${top_k_2}"
            echo "${BRAND_3}_top_k=${top_k_3}"
            echo "prompt_index=${prompt_index}"
            echo "gpu_id=${GPU_ID}"
            echo "status=FAILED (python command failed)"
            echo "log_file=${log_file}"
            echo "------------------------------------------------------------"
          } >> "${report_txt}"
          continue
        fi

        if ! parse_output="$(
          "${PYTHON_BIN}" - "${log_file}" "${KEYWORD_1}" "${KEYWORD_2}" "${KEYWORD_3}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
keywords = [x.lower() for x in sys.argv[2:5]]
text = log_path.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()

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
hits = [
    len(re.findall(rf"\b{re.escape(keyword)}\b", result_lower))
    for keyword in keywords
]

prompt_text = ""
for line in lines:
    if line.startswith("prompt:"):
        prompt_text = line.split("prompt:", 1)[1].strip()
        break

print(f"PROMPT_TEXT={prompt_text}")
print(f"HIT_1={hits[0]}")
print(f"HIT_2={hits[1]}")
print(f"HIT_3={hits[2]}")
print("RESULT_BLOCK_BEGIN")
print(result_block)
print("RESULT_BLOCK_END")
PY
        )"; then
          failed_runs=$((failed_runs + 1))
          failed_run_msgs+=("Run ${run_id} (k1=${top_k_1}, k2=${top_k_2}, k3=${top_k_3}): parse script failed")
          echo "  [WARN] Run ${run_id} parse failed, skip and continue. See log: ${log_file}"
          continue
        fi

        prompt_text="$(printf '%s\n' "${parse_output}" | awk -F= '/^PROMPT_TEXT=/{sub(/^PROMPT_TEXT=/, ""); print; exit}')"
        hit_1="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_1=/{print $2}')"
        hit_2="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_2=/{print $2}')"
        hit_3="$(printf '%s\n' "${parse_output}" | awk -F= '/^HIT_3=/{print $2}')"
        result_block="$(printf '%s\n' "${parse_output}" | awk '/^RESULT_BLOCK_BEGIN$/{flag=1;next}/^RESULT_BLOCK_END$/{flag=0}flag')"
        if ! [[ "${hit_1}" =~ ^[0-9]+$ && "${hit_2}" =~ ^[0-9]+$ && "${hit_3}" =~ ^[0-9]+$ ]]; then
          failed_runs=$((failed_runs + 1))
          failed_run_msgs+=("Run ${run_id} (k1=${top_k_1}, k2=${top_k_2}, k3=${top_k_3}): invalid parse output")
          echo "  [WARN] Run ${run_id} parse output invalid, skip and continue. See log: ${log_file}"
          continue
        fi

        {
          echo ""
          echo "------------------------------------------------------------"
          echo "Run ${run_id}"
          echo "${BRAND_1}_top_k=${top_k_1}"
          echo "${BRAND_2}_top_k=${top_k_2}"
          echo "${BRAND_3}_top_k=${top_k_3}"
          echo "${BRAND_1}_multiplier=${MULTIPLIER_1}"
          echo "${BRAND_2}_multiplier=${MULTIPLIER_2}"
          echo "${BRAND_3}_multiplier=${MULTIPLIER_3}"
          echo "prompt_index=${prompt_index}"
          echo "prompt=${prompt_text}"
          echo "gpu_id=${GPU_ID}"
          echo "${result_block}"
          echo "hit_${KEYWORD_1}=${hit_1}, hit_${KEYWORD_2}=${hit_2}, hit_${KEYWORD_3}=${hit_3}"
          echo "------------------------------------------------------------"
        } >> "${report_txt}"

        printf "%s\t%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%s\t%s\t%s\n" \
          "${run_id}" \
          "${top_k_1}" \
          "${top_k_2}" \
          "${top_k_3}" \
          "${MULTIPLIER_1}" \
          "${MULTIPLIER_2}" \
          "${MULTIPLIER_3}" \
          "${hit_1}" \
          "${hit_2}" \
          "${hit_3}" >> "${summary_csv}"
        sum_hit_1=$((sum_hit_1 + hit_1))
        sum_hit_2=$((sum_hit_2 + hit_2))
        sum_hit_3=$((sum_hit_3 + hit_3))
        success_runs=$((success_runs + 1))
        run_hit_1_sum_map["${run_id}"]=$(( ${run_hit_1_sum_map["${run_id}"]:-0} + hit_1 ))
        run_hit_2_sum_map["${run_id}"]=$(( ${run_hit_2_sum_map["${run_id}"]:-0} + hit_2 ))
        run_hit_3_sum_map["${run_id}"]=$(( ${run_hit_3_sum_map["${run_id}"]:-0} + hit_3 ))
        run_hit_count_success_map["${run_id}"]=$(( ${run_hit_count_success_map["${run_id}"]:-0} + 1 ))
      done
    done
  done

  if (( success_runs > 0 )); then
    avg_hit_1="$(awk -v s="${sum_hit_1}" -v n="${success_runs}" 'BEGIN{printf "%.6f", s/n}')"
    avg_hit_2="$(awk -v s="${sum_hit_2}" -v n="${success_runs}" 'BEGIN{printf "%.6f", s/n}')"
    avg_hit_3="$(awk -v s="${sum_hit_3}" -v n="${success_runs}" 'BEGIN{printf "%.6f", s/n}')"
    avg_hit_count="$(awk -v s1="${sum_hit_1}" -v s2="${sum_hit_2}" -v s3="${sum_hit_3}" -v n="${success_runs}" 'BEGIN{printf "%.6f", (s1+s2+s3)/n}')"
  else
    avg_hit_1="0.000000"
    avg_hit_2="0.000000"
    avg_hit_3="0.000000"
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
    echo "avg_hit_${KEYWORD_3}=${avg_hit_3}"
    echo "hit_count=${avg_hit_count}"
    if (( failed_runs > 0 )); then
      echo "failed_run_list:"
      for msg in "${failed_run_msgs[@]}"; do
        echo "  - ${msg}"
      done
    fi
  } >> "${overall_report_txt}"
done

printf "run_id\t%s_top_k\t%s_top_k\t%s_top_k\t%s_multiplier\t%s_multiplier\t%s_multiplier\thit_%s_avg\thit_%s_avg\thit_%s_avg\n" \
  "${BRAND_1}" "${BRAND_2}" "${BRAND_3}" \
  "${BRAND_1}" "${BRAND_2}" "${BRAND_3}" \
  "${KEYWORD_1}" "${KEYWORD_2}" "${KEYWORD_3}" > "${prompt_avg_csv}"
for run_id in $(seq 1 "${total_runs_per_prompt}"); do
  run_success_prompts="${run_hit_count_success_map["${run_id}"]:-0}"
  if (( run_success_prompts > 0 )); then
    run_avg_hit_1="$(awk -v s="${run_hit_1_sum_map["${run_id}"]}" -v n="${run_success_prompts}" 'BEGIN{printf "%.6f", s/n}')"
    run_avg_hit_2="$(awk -v s="${run_hit_2_sum_map["${run_id}"]}" -v n="${run_success_prompts}" 'BEGIN{printf "%.6f", s/n}')"
    run_avg_hit_3="$(awk -v s="${run_hit_3_sum_map["${run_id}"]}" -v n="${run_success_prompts}" 'BEGIN{printf "%.6f", s/n}')"
  else
    run_avg_hit_1="0.000000"
    run_avg_hit_2="0.000000"
    run_avg_hit_3="0.000000"
  fi
  printf "%s\t%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%s\t%s\t%s\n" \
    "${run_id}" \
    "${run_top_k_1_map["${run_id}"]}" \
    "${run_top_k_2_map["${run_id}"]}" \
    "${run_top_k_3_map["${run_id}"]}" \
    "${MULTIPLIER_1}" \
    "${MULTIPLIER_2}" \
    "${MULTIPLIER_3}" \
    "${run_avg_hit_1}" \
    "${run_avg_hit_2}" \
    "${run_avg_hit_3}" >> "${prompt_avg_csv}"
done

script_end_epoch="$(date +%s)"
script_end_time="$(date '+%Y-%m-%d %H:%M:%S')"
script_elapsed_seconds=$((script_end_epoch - script_start_epoch))
script_elapsed_hms="$(format_duration "${script_elapsed_seconds}")"

{
  echo ""
  echo "-------------------- Total Runtime --------------------"
  echo "sweep_started_at=${script_start_time}"
  echo "sweep_finished_at=${script_end_time}"
  echo "total_runtime_seconds=${script_elapsed_seconds}"
  echo "total_runtime_hms=${script_elapsed_hms}"
  echo ""
  echo "All prompts sweep finished at: ${script_end_time}"
  echo "Total runtime: ${script_elapsed_hms} (${script_elapsed_seconds}s)"
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
} | tee -a "${overall_report_txt}"
