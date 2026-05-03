#!/bin/bash
#SBATCH --gpus=1
# 以下参数只有本地测试时需要
module load compilers/cuda/12.1 compilers/gcc/9.3.0
module load cudnn/8.8.1.3_cuda12.x

source activate SVD

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888

set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash launcher_3bidders_qwen.sh --combos SPEC --gpus GPU_LIST [options]
  bash launcher_3bidders_qwen.sh --combo-file FILE --gpus GPU_LIST [options]
  bash launcher_3bidders_qwen.sh

Required:
  --gpus LIST, --gpus_list LIST
                        GPU id spec, e.g. 0,1,2,3 or 0-7 or 0-3,6

One of:
  --combos SPEC         3-bidder combo ids/keys, e.g. 0-4,8 or delta_marriott_visa
  --combo-file FILE     File with one 3-bidder combo id/key per line; lines starting with # are ignored

Optional:
  --max-jobs N          Max concurrent jobs. Default: GPU count * MAX_JOBS_PER_GPU
  --max-jobs-per-gpu N  Max concurrent jobs per GPU. Default: 1
  --topk-script PATH    Sweep script to call. Default: ./topk_sweep_3bidders_batch.sh
  --model-path PATH     Override MODEL_PATH passed to topk script
  --attr-cache-dir DIR, --attribution-cache-dir DIR
                        Override ATTR_CACHE_DIR passed to topk script
  --result-root DIR     Override first-level result dir passed to topk script
  --prompt-list LIST    Override prompt indexes passed to topk script, e.g. "0,1,2" or "0 1 2"
  --system-prompt TEXT  Add a system prompt before the user prompt. Default: empty
  --top-k-1 LIST        Override bidder 1 top-k values, e.g. "0,100,200" or "0 100 200"
  --top-k-2 LIST        Override bidder 2 top-k values
  --top-k-3 LIST        Override bidder 3 top-k values
  --log-dir DIR         Launcher output dir. Default: ./batch_runs_qwen_3bidders
  --stagger-sec N       Sleep N seconds between launches. Default: 2
  --min-free-mem-mb N   Treat GPU as selectable only if memory.free >= N. Default: 15000
  --idle-max-util N     Prefer GPU with utilization.gpu < N. Default: 70
  --poll-sec N          Sleep N seconds while waiting for an idle GPU. Default: 5
  --fail-fast           Stop launching new jobs after first failure
  --dry-run             Print commands without launching
  -h, --help            Show this help

Examples:
  bash launcher_3bidders_qwen.sh --combos 0-4 --gpus 0,1,2,3
  bash launcher_3bidders_qwen.sh --combos delta_marriott_visa,adobe_dell_logitech --gpus 0-1
  bash launcher_3bidders_qwen.sh --combo-file combo_3bidders.txt --gpus 4,5,6,7 --fail-fast
EOF
}

TOPK_SCRIPT="./topk_sweep_3bidders_batch.sh"
LOG_DIR="./batch_runs_qwen_3bidders"
COMBOS="0-19"
GPUS_LIST="0-7" #需要改

COMBO_SPEC="${COMBOS}"
COMBO_FILE=""
GPU_SPEC="${GPUS_LIST}"
MAX_JOBS=""
MAX_JOBS_PER_GPU="3"
STAGGER_SEC=2
MODEL_PATH="../Qwen3-4B"
ATTRIBUTION_CACHE_DIR="./attr_cache_qwen"
RESULT_ROOT="./batch_results_qwen_3bidders"
PROMPT_LIST="0,1,2"
# 在这里设置 system prompt；留空表示不加入 system message，保持原始行为。
SYSTEM_PROMPT="You are a helpful assistant. Answer my question in 1024 tokens."
TOP_K_1=(0 100 200 300 400 500 600 )
TOP_K_2=(0 100 200 300 400 500 600 )
TOP_K_3=(0 100 200 300 400 500 600 )
# TOP_K_1=(0 100)
# TOP_K_2=(0 100)
# TOP_K_3=(0 100)
MIN_FREE_MEM_MB=15000
POLL_SEC=5
MAX_IDLE_UTIL=70
FAIL_FAST=0
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-python}"

parse_top_k_list_spec() {
  local target_name="$1"
  local spec="$2"
  local token start end i
  local values=()

  spec="${spec//,/ }"
  for token in ${spec}; do
    if [[ "${token}" =~ ^[0-9]+-[0-9]+$ ]]; then
      start="${token%-*}"
      end="${token#*-}"
      if (( start > end )); then
        echo "[ERROR] invalid ${target_name} range: ${token}" >&2
        exit 1
      fi
      for (( i=start; i<=end; i++ )); do
        values+=("${i}")
      done
    elif [[ "${token}" =~ ^[0-9]+$ ]]; then
      values+=("${token}")
    else
      echo "[ERROR] invalid ${target_name} token: ${token}" >&2
      exit 1
    fi
  done

  if (( ${#values[@]} == 0 )); then
    echo "[ERROR] ${target_name} cannot be empty" >&2
    exit 1
  fi

  case "${target_name}" in
    TOP_K_1) TOP_K_1=("${values[@]}") ;;
    TOP_K_2) TOP_K_2=("${values[@]}") ;;
    TOP_K_3) TOP_K_3=("${values[@]}") ;;
    *)
      echo "[ERROR] unknown top-k target: ${target_name}" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --combos)
      [[ $# -ge 2 ]] || { echo "[ERROR] --combos requires a value" >&2; usage >&2; exit 1; }
      COMBO_SPEC="$2"
      shift 2
      ;;
    --combos=*)
      COMBO_SPEC="${1#*=}"
      shift
      ;;
    --combo-file)
      [[ $# -ge 2 ]] || { echo "[ERROR] --combo-file requires a value" >&2; usage >&2; exit 1; }
      COMBO_FILE="$2"
      shift 2
      ;;
    --combo-file=*)
      COMBO_FILE="${1#*=}"
      shift
      ;;
    --g|--gpus|--gpus_list)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      GPU_SPEC="$2"
      shift 2
      ;;
    --g=*|--gpus=*|--gpus_list=*)
      GPU_SPEC="${1#*=}"
      shift
      ;;
    --max-jobs)
      [[ $# -ge 2 ]] || { echo "[ERROR] --max-jobs requires a value" >&2; usage >&2; exit 1; }
      MAX_JOBS="$2"
      shift 2
      ;;
    --max-jobs=*)
      MAX_JOBS="${1#*=}"
      shift
      ;;
    --max-jobs-per-gpu)
      [[ $# -ge 2 ]] || { echo "[ERROR] --max-jobs-per-gpu requires a value" >&2; usage >&2; exit 1; }
      MAX_JOBS_PER_GPU="$2"
      shift 2
      ;;
    --max-jobs-per-gpu=*)
      MAX_JOBS_PER_GPU="${1#*=}"
      shift
      ;;
    --topk-script)
      [[ $# -ge 2 ]] || { echo "[ERROR] --topk-script requires a value" >&2; usage >&2; exit 1; }
      TOPK_SCRIPT="$2"
      shift 2
      ;;
    --topk-script=*)
      TOPK_SCRIPT="${1#*=}"
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
      ATTRIBUTION_CACHE_DIR="$2"
      shift 2
      ;;
    --attr-cache-dir=*|--attribution-cache-dir=*)
      ATTRIBUTION_CACHE_DIR="${1#*=}"
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
      PROMPT_LIST="$2"
      shift 2
      ;;
    --prompt-list=*|--prompts=*)
      PROMPT_LIST="${1#*=}"
      shift
      ;;
    --system-prompt)
      [[ $# -ge 2 ]] || { echo "[ERROR] --system-prompt requires a value" >&2; usage >&2; exit 1; }
      SYSTEM_PROMPT="$2"
      shift 2
      ;;
    --system-prompt=*)
      SYSTEM_PROMPT="${1#*=}"
      shift
      ;;
    --top-k-1|--top_k_1)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      parse_top_k_list_spec TOP_K_1 "$2"
      shift 2
      ;;
    --top-k-1=*|--top_k_1=*)
      parse_top_k_list_spec TOP_K_1 "${1#*=}"
      shift
      ;;
    --top-k-2|--top_k_2)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      parse_top_k_list_spec TOP_K_2 "$2"
      shift 2
      ;;
    --top-k-2=*|--top_k_2=*)
      parse_top_k_list_spec TOP_K_2 "${1#*=}"
      shift
      ;;
    --top-k-3|--top_k_3)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      parse_top_k_list_spec TOP_K_3 "$2"
      shift 2
      ;;
    --top-k-3=*|--top_k_3=*)
      parse_top_k_list_spec TOP_K_3 "${1#*=}"
      shift
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --log-dir requires a value" >&2; usage >&2; exit 1; }
      LOG_DIR="$2"
      shift 2
      ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"
      shift
      ;;
    --stagger-sec)
      [[ $# -ge 2 ]] || { echo "[ERROR] --stagger-sec requires a value" >&2; usage >&2; exit 1; }
      STAGGER_SEC="$2"
      shift 2
      ;;
    --stagger-sec=*)
      STAGGER_SEC="${1#*=}"
      shift
      ;;
    --min-free-mem-mb|--idle-max-mem-mb)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      MIN_FREE_MEM_MB="$2"
      shift 2
      ;;
    --min-free-mem-mb=*|--idle-max-mem-mb=*)
      MIN_FREE_MEM_MB="${1#*=}"
      shift
      ;;
    --idle-max-util)
      [[ $# -ge 2 ]] || { echo "[ERROR] --idle-max-util requires a value" >&2; usage >&2; exit 1; }
      MAX_IDLE_UTIL="$2"
      shift 2
      ;;
    --idle-max-util=*)
      MAX_IDLE_UTIL="${1#*=}"
      shift
      ;;
    --poll-sec)
      [[ $# -ge 2 ]] || { echo "[ERROR] --poll-sec requires a value" >&2; usage >&2; exit 1; }
      POLL_SEC="$2"
      shift 2
      ;;
    --poll-sec=*)
      POLL_SEC="${1#*=}"
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ -z "${GPU_SPEC}" ]]; then
  echo "[ERROR] Please provide --gpus/--gpus_list or define GPUS_LIST in this script" >&2
  usage >&2
  exit 1
fi

if [[ -n "${COMBO_SPEC}" && -n "${COMBO_FILE}" ]]; then
  echo "[ERROR] Use either --combos or --combo-file, not both" >&2
  exit 1
fi

if [[ -z "${COMBO_SPEC}" && -z "${COMBO_FILE}" ]]; then
  echo "[ERROR] Please provide --combos/--combo-file or define COMBOS in this script" >&2
  exit 1
fi

if [[ ! -f "${TOPK_SCRIPT}" ]]; then
  echo "[ERROR] topk script not found: ${TOPK_SCRIPT}" >&2
  exit 1
fi

GPU_IDS=()
while IFS= read -r line; do
  GPU_IDS+=("${line}")
done < <(
  "${PYTHON_BIN}" - "${GPU_SPEC}" <<'PY'
import sys

spec = sys.argv[1]
parts = [x.strip() for x in spec.split(",") if x.strip()]
seen = set()
result = []
for part in parts:
    if "-" in part:
        left, right = part.split("-", 1)
        start = int(left)
        end = int(right)
        step = 1 if end >= start else -1
        for x in range(start, end + step, step):
            if x not in seen:
                seen.add(x)
                result.append(x)
    else:
        x = int(part)
        if x not in seen:
            seen.add(x)
            result.append(x)
for x in result:
    print(x)
PY
)

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] No valid GPU ids parsed from --gpus ${GPU_SPEC}" >&2
  exit 1
fi

if [[ -z "${MAX_JOBS}" ]]; then
  if [[ "${MAX_JOBS_PER_GPU}" =~ ^[0-9]+$ ]] && (( MAX_JOBS_PER_GPU > 0 )); then
    MAX_JOBS="$(( ${#GPU_IDS[@]} * MAX_JOBS_PER_GPU ))"
  else
    MAX_JOBS="${#GPU_IDS[@]}"
  fi
fi

COMBO_META_LINES=()
while IFS= read -r line; do
  COMBO_META_LINES+=("${line}")
done < <(
  if [[ -n "${COMBO_SPEC}" ]]; then
    "${PYTHON_BIN}" - --spec "${COMBO_SPEC}" <<'PY'
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--spec", required=True)
args = parser.parse_args()

three_config = importlib.import_module("src.3_config")
presets = three_config.THREE_BIDDER_COMBO_PRESETS
combo_keys = list(presets.keys())
seen = set()

def emit(raw):
    if raw in presets:
        combo_ref = raw
        combo_key = raw
    elif raw.isdigit():
        combo_id = int(raw)
        if combo_id < 0 or combo_id >= len(combo_keys):
            raise SystemExit(f"3-bidder combo id out of range: {combo_id}; valid=[0, {len(combo_keys)-1}]")
        combo_ref = str(combo_id)
        combo_key = combo_keys[combo_id]
    else:
        raise SystemExit(f"unknown 3-bidder combo preset: {raw}; valid keys: {', '.join(combo_keys)}")
    if combo_key in seen:
        return
    seen.add(combo_key)
    brand_1, brand_2, brand_3 = presets[combo_key]
    print(f"{combo_ref}\t{combo_key}\t{brand_1}\t{brand_2}\t{brand_3}")

for part in args.spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part and all(x.strip().isdigit() for x in part.split("-", 1)):
        left, right = part.split("-", 1)
        start = int(left)
        end = int(right)
        step = 1 if end >= start else -1
        for combo_id in range(start, end + step, step):
            emit(str(combo_id))
    else:
        emit(part)
PY
  else
    "${PYTHON_BIN}" - --combo-file "${COMBO_FILE}" <<'PY'
import argparse
import importlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--combo-file", required=True)
args = parser.parse_args()

path = Path(args.combo_file)
if not path.exists():
    raise SystemExit(f"combo file not found: {path}")

three_config = importlib.import_module("src.3_config")
presets = three_config.THREE_BIDDER_COMBO_PRESETS
combo_keys = list(presets.keys())
seen = set()

for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line in presets:
        combo_ref = line
        combo_key = line
    elif line.isdigit():
        combo_id = int(line)
        if combo_id < 0 or combo_id >= len(combo_keys):
            raise SystemExit(f"3-bidder combo id out of range: {combo_id}; valid=[0, {len(combo_keys)-1}]")
        combo_ref = str(combo_id)
        combo_key = combo_keys[combo_id]
    else:
        raise SystemExit(f"unknown 3-bidder combo preset: {line}; valid keys: {', '.join(combo_keys)}")
    if combo_key in seen:
        continue
    seen.add(combo_key)
    brand_1, brand_2, brand_3 = presets[combo_key]
    print(f"{combo_ref}\t{combo_key}\t{brand_1}\t{brand_2}\t{brand_3}")
PY
  fi
)

if [[ ${#COMBO_META_LINES[@]} -eq 0 ]]; then
  echo "[ERROR] No 3-bidder combos to run" >&2
  exit 1
fi

COMBO_REFS=()
COMBO_KEYS=()
COMBO_BRAND1=()
COMBO_BRAND2=()
COMBO_BRAND3=()
for line in "${COMBO_META_LINES[@]}"; do
  IFS=$'\t' read -r combo_ref combo_key brand_1 brand_2 brand_3 <<< "${line}"
  COMBO_REFS+=("${combo_ref}")
  COMBO_KEYS+=("${combo_key}")
  COMBO_BRAND1+=("${brand_1}")
  COMBO_BRAND2+=("${brand_2}")
  COMBO_BRAND3+=("${brand_3}")
done

timestamp="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="${LOG_DIR}/launcher_${timestamp}"
mkdir -p "${RUN_DIR}/logs"

NVIDIA_SMI_AVAILABLE=0
if nvidia-smi --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  NVIDIA_SMI_AVAILABLE=1
fi

manifest_path="${RUN_DIR}/manifest.txt"
cat > "${manifest_path}" <<EOF
timestamp=${timestamp}
topk_script=${TOPK_SCRIPT}
defined_combos=${COMBOS}
defined_gpus_list=${GPUS_LIST}
combo_spec=${COMBO_SPEC}
combo_file=${COMBO_FILE}
combo_refs=${COMBO_REFS[*]}
combo_keys=${COMBO_KEYS[*]}
gpu_ids=${GPU_IDS[*]}
max_jobs=${MAX_JOBS}
max_jobs_per_gpu=${MAX_JOBS_PER_GPU}
stagger_sec=${STAGGER_SEC}
model_path=${MODEL_PATH}
attribution_cache_dir=${ATTRIBUTION_CACHE_DIR}
result_root=${RESULT_ROOT:-<auto>}
prompt_list=${PROMPT_LIST:-<topk default>}
system_prompt=${SYSTEM_PROMPT:-<empty>}
top_k_1=${TOP_K_1[*]}
top_k_2=${TOP_K_2[*]}
top_k_3=${TOP_K_3[*]}
min_free_mem_mb=${MIN_FREE_MEM_MB}
max_idle_util=${MAX_IDLE_UTIL}
poll_sec=${POLL_SEC}
nvidia_smi_available=${NVIDIA_SMI_AVAILABLE}
fail_fast=${FAIL_FAST}
dry_run=${DRY_RUN}
EOF

{
  echo "combo_mapping_begin"
  for i in "${!COMBO_KEYS[@]}"; do
    echo "combo_${COMBO_REFS[$i]}=${COMBO_KEYS[$i]}|${COMBO_BRAND1[$i]}|${COMBO_BRAND2[$i]}|${COMBO_BRAND3[$i]}"
  done
  echo "combo_mapping_end"
} >> "${manifest_path}"

echo "[Launcher3] manifest: ${manifest_path}"
echo "[Launcher3] combo count: ${#COMBO_KEYS[@]}"
echo "[Launcher3] gpus: ${GPU_IDS[*]}"
echo "[Launcher3] max_jobs: ${MAX_JOBS}"
echo "[Launcher3] max_jobs_per_gpu: ${MAX_JOBS_PER_GPU}"
echo "[Launcher3] result_root: ${RESULT_ROOT:-<auto>}"
echo "[Launcher3] prompt_list: ${PROMPT_LIST:-<topk default>}"
echo "[Launcher3] system_prompt: ${SYSTEM_PROMPT:-<empty>}"
echo "[Launcher3] top_k_1: ${TOP_K_1[*]}"
echo "[Launcher3] top_k_2: ${TOP_K_2[*]}"
echo "[Launcher3] top_k_3: ${TOP_K_3[*]}"

job_trace_path="${RUN_DIR}/job_trace.tsv"
{
  printf "event_time\tevent\tpid\tgpu\tcombo_ref\tcombo_key\tbrand_1\tbrand_2\tbrand_3\tstatus\tjob_log\n"
} > "${job_trace_path}"
echo "[Launcher3] job trace: ${job_trace_path}"

declare -A PID_TO_REF
declare -A PID_TO_GPU
declare -A PID_TO_LOG
declare -A PID_TO_KEY
declare -A PID_TO_BRAND1
declare -A PID_TO_BRAND2
declare -A PID_TO_BRAND3

success_count=0
failed_count=0
stop_launching=0
SELECTED_GPU=""

running_jobs_count() {
  local n=0
  local pid
  for pid in "${!PID_TO_REF[@]}"; do
    n=$((n + 1))
  done
  printf '%s\n' "${n}"
}

cleanup_finished_jobs() {
  local pid status
  for pid in "${!PID_TO_REF[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    if wait "${pid}"; then
      success_count=$((success_count + 1))
      status="OK"
    else
      failed_count=$((failed_count + 1))
      status="FAIL"
      if [[ "${FAIL_FAST}" -eq 1 ]]; then
        stop_launching=1
      fi
    fi
    echo "[Launcher3] finished: combo=${PID_TO_REF[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} brand_3=${PID_TO_BRAND3[$pid]} gpu=${PID_TO_GPU[$pid]} status=${status} log=${PID_TO_LOG[$pid]}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "FINISH" \
      "${pid}" \
      "${PID_TO_GPU[$pid]}" \
      "${PID_TO_REF[$pid]}" \
      "${PID_TO_KEY[$pid]}" \
      "${PID_TO_BRAND1[$pid]}" \
      "${PID_TO_BRAND2[$pid]}" \
      "${PID_TO_BRAND3[$pid]}" \
      "${status}" \
      "${PID_TO_LOG[$pid]}" >> "${job_trace_path}"
    unset 'PID_TO_REF[$pid]'
    unset 'PID_TO_GPU[$pid]'
    unset 'PID_TO_LOG[$pid]'
    unset 'PID_TO_KEY[$pid]'
    unset 'PID_TO_BRAND1[$pid]'
    unset 'PID_TO_BRAND2[$pid]'
    unset 'PID_TO_BRAND3[$pid]'
  done
}

running_jobs_on_gpu() {
  local target_gpu="$1"
  local pid count=0
  for pid in "${!PID_TO_GPU[@]}"; do
    if [[ "${PID_TO_GPU[$pid]}" == "${target_gpu}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

select_gpu() {
  local gpu line free util best_gpu="" best_free=-1 running

  if [[ "${NVIDIA_SMI_AVAILABLE}" -eq 1 ]]; then
    for gpu in "${GPU_IDS[@]}"; do
      running="$(running_jobs_on_gpu "${gpu}")"
      if (( running >= MAX_JOBS_PER_GPU )); then
        continue
      fi
      line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
      free="$(printf '%s' "${line}" | awk -F, '{gsub(/ /, "", $1); print $1}')"
      util="$(printf '%s' "${line}" | awk -F, '{gsub(/ /, "", $2); print $2}')"
      if ! [[ "${free}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]]; then
        continue
      fi
      if (( free >= MIN_FREE_MEM_MB && util < MAX_IDLE_UTIL )); then
        SELECTED_GPU="${gpu}"
        return 0
      fi
      if (( free >= MIN_FREE_MEM_MB && free > best_free )); then
        best_free="${free}"
        best_gpu="${gpu}"
      fi
    done
    if [[ -n "${best_gpu}" ]]; then
      SELECTED_GPU="${best_gpu}"
      return 0
    fi
  fi

  for gpu in "${GPU_IDS[@]}"; do
    running="$(running_jobs_on_gpu "${gpu}")"
    if (( running < MAX_JOBS_PER_GPU )); then
      SELECTED_GPU="${gpu}"
      return 0
    fi
  done

  return 1
}

for i in "${!COMBO_KEYS[@]}"; do
  if [[ "${stop_launching}" -eq 1 ]]; then
    echo "[Launcher3] fail-fast active; stop launching new jobs"
    break
  fi

  cleanup_finished_jobs
  while (( $(running_jobs_count) >= MAX_JOBS )); do
    sleep "${POLL_SEC}"
    cleanup_finished_jobs
  done

  while ! select_gpu; do
    sleep "${POLL_SEC}"
    cleanup_finished_jobs
    if [[ "${stop_launching}" -eq 1 ]]; then
      break
    fi
  done
  if [[ "${stop_launching}" -eq 1 ]]; then
    break
  fi

  gpu_id="${SELECTED_GPU}"
  combo_ref="${COMBO_REFS[$i]}"
  combo_key="${COMBO_KEYS[$i]}"
  combo_brand_1="${COMBO_BRAND1[$i]}"
  combo_brand_2="${COMBO_BRAND2[$i]}"
  combo_brand_3="${COMBO_BRAND3[$i]}"
  safe_combo_key="${combo_key//[^[:alnum:]_-]/_}"
  job_log="${RUN_DIR}/logs/combo_${safe_combo_key}_gpu_${gpu_id}.log"

  cmd=(bash "${TOPK_SCRIPT}" --g "${gpu_id}" --combo-preset "${combo_key}")
  if [[ -n "${MODEL_PATH}" ]]; then
    cmd+=(--model-path "${MODEL_PATH}")
  fi
  if [[ -n "${ATTRIBUTION_CACHE_DIR}" ]]; then
    cmd+=(--attr-cache-dir "${ATTRIBUTION_CACHE_DIR}")
  fi
  if [[ -n "${RESULT_ROOT}" ]]; then
    cmd+=(--result-root "${RESULT_ROOT}")
  fi
  if [[ -n "${PROMPT_LIST}" ]]; then
    cmd+=(--prompt-list "${PROMPT_LIST}")
  fi
  if [[ -n "${SYSTEM_PROMPT}" ]]; then
    cmd+=(--system-prompt "${SYSTEM_PROMPT}")
  fi
  cmd+=(--top-k-1 "${TOP_K_1[*]}")
  cmd+=(--top-k-2 "${TOP_K_2[*]}")
  cmd+=(--top-k-3 "${TOP_K_3[*]}")

  echo "[Launcher3] launch: combo=${combo_ref} key=${combo_key} brand_1=${combo_brand_1} brand_2=${combo_brand_2} brand_3=${combo_brand_3} gpu=${gpu_id} result_root=${RESULT_ROOT:-<auto>} log=${job_log}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    continue
  fi

  (
    echo "[Launcher3] START combo=${combo_ref} key=${combo_key} gpu=${gpu_id} time=$(date)"
    printf '[Launcher3] CMD:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    status=$?
    echo "[Launcher3] END combo=${combo_ref} key=${combo_key} gpu=${gpu_id} status=${status} time=$(date)"
    exit "${status}"
  ) > "${job_log}" 2>&1 &
  pid=$!

  PID_TO_REF["${pid}"]="${combo_ref}"
  PID_TO_GPU["${pid}"]="${gpu_id}"
  PID_TO_LOG["${pid}"]="${job_log}"
  PID_TO_KEY["${pid}"]="${combo_key}"
  PID_TO_BRAND1["${pid}"]="${combo_brand_1}"
  PID_TO_BRAND2["${pid}"]="${combo_brand_2}"
  PID_TO_BRAND3["${pid}"]="${combo_brand_3}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "START" \
    "${pid}" \
    "${gpu_id}" \
    "${combo_ref}" \
    "${combo_key}" \
    "${combo_brand_1}" \
    "${combo_brand_2}" \
    "${combo_brand_3}" \
    "RUNNING" \
    "${job_log}" >> "${job_trace_path}"

  sleep "${STAGGER_SEC}"
done

while (( $(running_jobs_count) > 0 )); do
  sleep "${POLL_SEC}"
  cleanup_finished_jobs
done

echo "[Launcher3] all done. success=${success_count} failed=${failed_count} run_dir=${RUN_DIR}"
if (( failed_count > 0 )); then
  exit 1
fi
