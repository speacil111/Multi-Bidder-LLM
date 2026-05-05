#!/bin/bash

set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash launcher.sh --combos SPEC --gpus GPU_LIST [options]
  bash launcher.sh --combo-file FILE --gpus GPU_LIST [options]
  # Or define COMBOS / GPUS_LIST directly in this script and run:
  bash launcher.sh

Required:
  --gpus LIST, --gpus_list LIST
                        GPU id spec, e.g. 0,1,2,3 or 0-7 or 0-3,6

One of:
  --combos SPEC         Combo id spec, e.g. 0-9,12,18,20-25
  --combo-file FILE     File with one combo id per line; lines starting with # are ignored

Optional:
  --max-jobs N          Total worker cap. Default: GPU count * MAX_JOBS_PER_GPU
  --max-jobs-per-gpu N  Workers per GPU. Default: 1
  --topk-script PATH    Worker script to call. Default: ./topk_sweep_worker.sh
  --model-path PATH     Override MODEL_PATH passed to topk script
  --attr-cache-dir DIR, --attribution-cache-dir DIR
                        Override ATTR_CACHE_DIR passed to topk script
  --result-root DIR     Override first-level result dir passed to topk script
  --prompt-list LIST    Override prompt indexes passed to topk script, e.g. "0,1,2" or "0 1 2"
  --lambda-list LIST    Intervention strength list. Each value is used for both brands,
                        e.g. "1.0,1.5,2.0" or "1.0 1.5 2.0"
  --top-k-1 LIST        Override bidder 1 top-k values, e.g. "0,100,200" or "0 100 200"
  --top-k-2 LIST        Override bidder 2 top-k values
  --log-dir DIR         Launcher output dir. Default: ./batch_runs_Llama
  --state-dir DIR       Worker done-marker dir. Default: <result-root>/.worker_state_lambda
  --force-rerun         Ignore worker done markers and rerun assigned combos
  --stagger-sec N       Sleep N seconds between launches. Default: 2
  --min-free-mem-mb N   Treat GPU as selectable only if memory.free >= N. Default: 15000
  --idle-max-util N     Prefer GPU with utilization.gpu < N. Default: 70
  --poll-sec N          Sleep N seconds while waiting for an idle GPU. Default: 5
  --fail-fast           Stop launching new jobs after first failure
  --dry-run             Print commands without launching
  -h, --help            Show this help

Examples:
  bash launcher.sh --combos w0-9 --gpus 0,1,2,3
  bash launcher.sh --combos 10-29,35 --gpus 0,1 --max-jobs 2
  bash launcher.sh --combo-file combo_ids.txt --gpus 4,5,6,7 --fail-fast

Notes:
  1. This launcher starts long-lived GPU workers.
  2. Each worker runs one lambda value over multiple combo ids sequentially.
  3. Done markers are written per lambda/config signature, so reruns skip completed combos.
  4. Current sweep outputs are still grouped by first brand and multiplier.
     If many combos reuse the same first brand, those outputs may collide or overwrite.
     So keep combo sets disjoint by first brand if you need those result dirs preserved separately.
EOF
}

TOPK_SCRIPT="./topk_sweep_worker.sh"
LOG_DIR="./batch_runs_qwen_lambda" # 运行日志存放路径
# 直接在这里定义默认任务（可被命令行参数覆盖）
# 例: COMBOS="0-9,12,18" ; GPUS_LIST="0,1,2,3" 或 "0-7"
COMBOS="0-30"
GPUS_LIST="0-7" # 使用的GPU号
LAMBDA_LIST="1.25 1.5 1.75 2.25 2.5 2.75 3.0" # 干预强度列表；两个品牌默认使用相同强度，例如 "1.0,1.5,2.0"

COMBO_SPEC="${COMBOS}"
COMBO_FILE=""
GPU_SPEC="${GPUS_LIST}"
MAX_JOBS="" # 总共最大个数
MAX_JOBS_PER_GPU="3" #每个GPU上的最大运行JOB个数
STAGGER_SEC=2
MODEL_PATH="../Qwen3-4B" # 使用的模型路径
ATTRIBUTION_CACHE_DIR="./attr_cache_qwen" # 属性缓存路径
# First-level result dir. Empty means topk_sweep_batch.sh uses batch_results_<model_tag>.
RESULT_ROOT="./batch_results_qwen_lambda"
WORKER_STATE_DIR=""
# Empty means use PROMPT_LIST inside topk_sweep_batch.sh.
PROMPT_LIST="0 1 2" # 需要测试的prompt 序号
LAMBDA_SPEC="${LAMBDA_LIST}"
TOP_K_1=(0 100 200 300 400 500 600 700 800)
TOP_K_2=(0 100 200 300 400 500 600 700 800)
# TOP_K_1=(0 100 )
# TOP_K_2=(0 100 )
MIN_FREE_MEM_MB=15000
POLL_SEC=5
MAX_IDLE_UTIL=70
FAIL_FAST=0
DRY_RUN=0
FORCE_RERUN=0

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
    --lambda-list|--lambdas)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      LAMBDA_SPEC="$2"
      shift 2
      ;;
    --lambda-list=*|--lambdas=*)
      LAMBDA_SPEC="${1#*=}"
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
    --log-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --log-dir requires a value" >&2; usage >&2; exit 1; }
      LOG_DIR="$2"
      shift 2
      ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"
      shift
      ;;
    --state-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --state-dir requires a value" >&2; usage >&2; exit 1; }
      WORKER_STATE_DIR="$2"
      shift 2
      ;;
    --state-dir=*)
      WORKER_STATE_DIR="${1#*=}"
      shift
      ;;
    --force-rerun)
      FORCE_RERUN=1
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
  echo "[ERROR] Please provide --gpus/--gpus_list or define GPUS_LIST in launcher.sh" >&2
  usage >&2
  exit 1
fi

if [[ -n "${COMBO_SPEC}" && -n "${COMBO_FILE}" ]]; then
  echo "[ERROR] Use either --combos or --combo-file, not both" >&2
  exit 1
fi

if [[ -z "${COMBO_SPEC}" && -z "${COMBO_FILE}" ]]; then
  echo "[ERROR] Please provide --combos/--combo-file or define COMBOS in launcher.sh" >&2
  exit 1
fi

if [[ ! -f "${TOPK_SCRIPT}" ]]; then
  echo "[ERROR] topk script not found: ${TOPK_SCRIPT}" >&2
  exit 1
fi

readarray -t GPU_IDS < <(
  python - "${GPU_SPEC}" <<'PY'
import sys
spec = sys.argv[1]
parts = [x.strip() for x in spec.split(",") if x.strip()]
if not parts:
    raise SystemExit("no gpu ids found")
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

if ! [[ "${MAX_JOBS_PER_GPU}" =~ ^[0-9]+$ ]] || (( MAX_JOBS_PER_GPU <= 0 )); then
  echo "[ERROR] --max-jobs-per-gpu must be a positive integer, got: ${MAX_JOBS_PER_GPU}" >&2
  exit 1
fi

if ! [[ "${MAX_JOBS}" =~ ^[0-9]+$ ]] || (( MAX_JOBS <= 0 )); then
  echo "[ERROR] --max-jobs must be a positive integer, got: ${MAX_JOBS}" >&2
  exit 1
fi

TOTAL_WORKER_SLOTS=$(( ${#GPU_IDS[@]} * MAX_JOBS_PER_GPU ))
if (( MAX_JOBS < TOTAL_WORKER_SLOTS )); then
  TOTAL_WORKER_SLOTS="${MAX_JOBS}"
fi

if (( TOTAL_WORKER_SLOTS <= 0 )); then
  echo "[ERROR] No worker slots available" >&2
  exit 1
fi

readarray -t COMBO_IDS < <(
  if [[ -n "${COMBO_SPEC}" ]]; then
    python - "${COMBO_SPEC}" <<'PY'
import sys
spec = sys.argv[1]
seen = set()
result = []
for part in spec.split(","):
    part = part.strip()
    if not part:
        continue
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
  else
    python - "${COMBO_FILE}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"combo file not found: {path}")
seen = set()
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    x = int(line)
    if x in seen:
        continue
    seen.add(x)
    print(x)
PY
  fi
)

if [[ ${#COMBO_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] No combo ids to run" >&2
  exit 1
fi

readarray -t LAMBDA_VALUES < <(
  python - "${LAMBDA_SPEC}" <<'PY'
import sys

spec = sys.argv[1]
seen = set()
values = []
for token in spec.replace(",", " ").split():
    value = token.strip()
    if not value:
        continue
    try:
        float(value)
    except ValueError:
        raise SystemExit(f"invalid lambda value: {value}")
    if value in seen:
        continue
    seen.add(value)
    values.append(value)

if not values:
    raise SystemExit("lambda list cannot be empty")

for value in values:
    print(value)
PY
)

if [[ ${#LAMBDA_VALUES[@]} -eq 0 ]]; then
  echo "[ERROR] No lambda values to run" >&2
  exit 1
fi

declare -A COMBO_TO_KEY
declare -A COMBO_TO_BRAND1
declare -A COMBO_TO_BRAND2

readarray -t COMBO_META_LINES < <(
  python - "${COMBO_IDS[@]}" <<'PY'
import sys
from src.config import COMBO_PRESETS

combo_keys = list(COMBO_PRESETS.keys())

for raw in sys.argv[1:]:
    combo_id = int(raw)
    if combo_id < 0 or combo_id >= len(combo_keys):
        raise SystemExit(
            f"combo preset id out of range: {combo_id}; valid=[0, {len(combo_keys)-1}]"
        )
    combo_key = combo_keys[combo_id]
    brand_1, brand_2 = COMBO_PRESETS[combo_key]
    print(f"{combo_id}\t{combo_key}\t{brand_1}\t{brand_2}")
PY
)

for line in "${COMBO_META_LINES[@]}"; do
  IFS=$'\t' read -r combo_id combo_key brand_1 brand_2 <<< "${line}"
  COMBO_TO_KEY["${combo_id}"]="${combo_key}"
  COMBO_TO_BRAND1["${combo_id}"]="${brand_1}"
  COMBO_TO_BRAND2["${combo_id}"]="${brand_2}"
done

timestamp="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="${LOG_DIR}/launcher_${timestamp}"
mkdir -p "${RUN_DIR}/logs"

NVIDIA_SMI_AVAILABLE=0
if nvidia-smi --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  NVIDIA_SMI_AVAILABLE=1
fi

if [[ -z "${WORKER_STATE_DIR}" ]]; then
  if [[ -n "${RESULT_ROOT}" ]]; then
    WORKER_STATE_DIR="${RESULT_ROOT%/}/.worker_state_lambda"
  else
    WORKER_STATE_DIR="${RUN_DIR}/worker_state"
  fi
fi
mkdir -p "${WORKER_STATE_DIR}"

manifest_path="${RUN_DIR}/manifest.txt"
cat > "${manifest_path}" <<EOF
timestamp=${timestamp}
launcher_mode=lambda_gpu_worker
topk_script=${TOPK_SCRIPT}
defined_combos=${COMBOS}
defined_gpus_list=${GPUS_LIST}
combo_spec=${COMBO_SPEC}
combo_file=${COMBO_FILE}
combo_ids=${COMBO_IDS[*]}
gpu_ids=${GPU_IDS[*]}
max_jobs=${MAX_JOBS}
max_jobs_per_gpu=${MAX_JOBS_PER_GPU}
total_worker_slots=${TOTAL_WORKER_SLOTS}
stagger_sec=${STAGGER_SEC}
model_path=${MODEL_PATH}
attribution_cache_dir=${ATTRIBUTION_CACHE_DIR}
result_root=${RESULT_ROOT:-<auto>}
worker_state_dir=${WORKER_STATE_DIR}
prompt_list=${PROMPT_LIST:-<topk default>}
lambda_list=${LAMBDA_VALUES[*]}
top_k_1=${TOP_K_1[*]}
top_k_2=${TOP_K_2[*]}
min_free_mem_mb=${MIN_FREE_MEM_MB}
max_idle_util=${MAX_IDLE_UTIL}
poll_sec=${POLL_SEC}
nvidia_smi_available=${NVIDIA_SMI_AVAILABLE}
fail_fast=${FAIL_FAST}
dry_run=${DRY_RUN}
force_rerun=${FORCE_RERUN}
EOF

{
  echo "combo_mapping_begin"
  for combo_id in "${COMBO_IDS[@]}"; do
    echo "combo_${combo_id}=${COMBO_TO_KEY[${combo_id}]}|${COMBO_TO_BRAND1[${combo_id}]}|${COMBO_TO_BRAND2[${combo_id}]}"
  done
  echo "combo_mapping_end"
} >> "${manifest_path}"

echo "[Launcher] manifest: ${manifest_path}"
echo "[Launcher] combo count: ${#COMBO_IDS[@]}"
echo "[Launcher] lambda count: ${#LAMBDA_VALUES[@]} (${LAMBDA_VALUES[*]})"
echo "[Launcher] gpus: ${GPU_IDS[*]}"
echo "[Launcher] max_jobs: ${MAX_JOBS}"
echo "[Launcher] max_jobs_per_gpu: ${MAX_JOBS_PER_GPU}"
echo "[Launcher] total_worker_slots: ${TOTAL_WORKER_SLOTS}"
echo "[Launcher] result_root: ${RESULT_ROOT:-<auto>}"
echo "[Launcher] worker_state_dir: ${WORKER_STATE_DIR}"
echo "[Launcher] prompt_list: ${PROMPT_LIST:-<topk default>}"
echo "[Launcher] top_k_1: ${TOP_K_1[*]}"
echo "[Launcher] top_k_2: ${TOP_K_2[*]}"
if [[ "${NVIDIA_SMI_AVAILABLE}" -eq 1 ]]; then
  echo "[Launcher] idle-gpu mode: prefer memory.free>=${MIN_FREE_MEM_MB}MB and utilization.gpu<${MAX_IDLE_UTIL}%, fallback to highest memory.free with memory.free>=${MIN_FREE_MEM_MB}MB"
else
  echo "[Launcher] idle-gpu mode: nvidia-smi unavailable, fallback to launcher-only GPU tracking"
fi

job_trace_path="${RUN_DIR}/job_trace.tsv"
{
  printf "event_time\tevent\tpid\tgpu\tcombo_group\tcombo_key\tbrand_1\tbrand_2\tlambda\tstatus\tjob_log\n"
} > "${job_trace_path}"
echo "[Launcher] job trace: ${job_trace_path}"

declare -A PID_TO_COMBO
declare -A PID_TO_GPU
declare -A PID_TO_LOG
declare -A PID_TO_KEY
declare -A PID_TO_BRAND1
declare -A PID_TO_BRAND2
declare -A PID_TO_LAMBDA

success_count=0
failed_count=0
stop_launching=0
SELECTED_GPU=""
LAST_ACTIVE_JOBS_SIGNATURE=""

running_jobs_count() {
  local n=0
  local pid
  for pid in "${!PID_TO_COMBO[@]}"; do
    n=$((n + 1))
  done
  printf '%s\n' "${n}"
}

print_active_jobs() {
  local n pid signature
  n="$(running_jobs_count)"
  if (( n == 0 )); then
    LAST_ACTIVE_JOBS_SIGNATURE=""
    return
  fi
  signature="$(
    for pid in "${!PID_TO_COMBO[@]}"; do
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${pid}" \
        "${PID_TO_GPU[$pid]}" \
        "${PID_TO_COMBO[$pid]}" \
        "${PID_TO_KEY[$pid]}" \
        "${PID_TO_BRAND1[$pid]}" \
        "${PID_TO_BRAND2[$pid]}" \
        "${PID_TO_LAMBDA[$pid]}" \
        "${PID_TO_LOG[$pid]}"
    done | sort
  )"
  if [[ "${signature}" == "${LAST_ACTIVE_JOBS_SIGNATURE}" ]]; then
    return
  fi
  LAST_ACTIVE_JOBS_SIGNATURE="${signature}"
  echo "[Launcher] active jobs (${n}):"
  for pid in "${!PID_TO_COMBO[@]}"; do
    echo "  pid=${pid} gpu=${PID_TO_GPU[$pid]} combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} lambda=${PID_TO_LAMBDA[$pid]} log=${PID_TO_LOG[$pid]}"
  done
}

cleanup_finished_jobs() {
  local pid
  local status
  for pid in "${!PID_TO_COMBO[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    if wait "${pid}"; then
      success_count=$((success_count + 1))
      status="OK"
      echo "[Launcher] finished: combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} lambda=${PID_TO_LAMBDA[$pid]} gpu=${PID_TO_GPU[$pid]} status=OK log=${PID_TO_LOG[$pid]}"
    else
      failed_count=$((failed_count + 1))
      status="FAIL"
      echo "[Launcher] finished: combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} lambda=${PID_TO_LAMBDA[$pid]} gpu=${PID_TO_GPU[$pid]} status=FAIL log=${PID_TO_LOG[$pid]}"
      if [[ "${FAIL_FAST}" -eq 1 ]]; then
        stop_launching=1
      fi
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "FINISH" \
      "${pid}" \
      "${PID_TO_GPU[$pid]}" \
      "${PID_TO_COMBO[$pid]}" \
      "${PID_TO_KEY[$pid]}" \
      "${PID_TO_BRAND1[$pid]}" \
      "${PID_TO_BRAND2[$pid]}" \
      "${PID_TO_LAMBDA[$pid]}" \
      "${status}" \
      "${PID_TO_LOG[$pid]}" >> "${job_trace_path}"
    unset 'PID_TO_COMBO[$pid]'
    unset 'PID_TO_GPU[$pid]'
    unset 'PID_TO_LOG[$pid]'
    unset 'PID_TO_KEY[$pid]'
    unset 'PID_TO_BRAND1[$pid]'
    unset 'PID_TO_BRAND2[$pid]'
    unset 'PID_TO_LAMBDA[$pid]'
  done
}

running_jobs_on_gpu() {
  local target_gpu="$1"
  local pid
  local count=0
  for pid in "${!PID_TO_GPU[@]}"; do
    if [[ "${PID_TO_GPU[$pid]}" == "${target_gpu}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

get_gpu_stats() {
  local target_gpu="$1"
  local stats
  local mem_free
  local gpu_util

  if [[ "${NVIDIA_SMI_AVAILABLE}" -ne 1 ]]; then
    return 1
  fi

  stats="$(nvidia-smi -i "${target_gpu}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  if [[ -z "${stats}" ]]; then
    return 1
  fi

  IFS=',' read -r mem_free gpu_util <<< "${stats}"
  mem_free="${mem_free//[[:space:]]/}"
  gpu_util="${gpu_util//[[:space:]]/}"

  if ! [[ "${mem_free}" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  if ! [[ "${gpu_util}" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  printf '%s %s\n' "${mem_free}" "${gpu_util}"
  return 0
}

pick_idle_gpu() {
  local gpu_id
  local stats
  local mem_free
  local gpu_util
  local best_strict_gpu=""
  local best_strict_mem=-1
  local best_fallback_gpu=""
  local best_fallback_mem=-1
  local jobs_on_gpu

  if (( $(running_jobs_count) >= MAX_JOBS )); then
    return 1
  fi

  if [[ "${NVIDIA_SMI_AVAILABLE}" -ne 1 ]]; then
    for gpu_id in "${GPU_IDS[@]}"; do
      jobs_on_gpu="$(running_jobs_on_gpu "${gpu_id}")"
      if [[ "${MAX_JOBS_PER_GPU}" =~ ^[0-9]+$ ]] && (( MAX_JOBS_PER_GPU > 0 )) && (( jobs_on_gpu >= MAX_JOBS_PER_GPU )); then
        continue
      fi
      SELECTED_GPU="${gpu_id}"
      return 0
    done
    return 1
  fi

  for gpu_id in "${GPU_IDS[@]}"; do
    jobs_on_gpu="$(running_jobs_on_gpu "${gpu_id}")"
    if [[ "${MAX_JOBS_PER_GPU}" =~ ^[0-9]+$ ]] && (( MAX_JOBS_PER_GPU > 0 )) && (( jobs_on_gpu >= MAX_JOBS_PER_GPU )); then
      continue
    fi

    stats="$(get_gpu_stats "${gpu_id}")" || continue
    read -r mem_free gpu_util <<< "${stats}"

    if (( mem_free < MIN_FREE_MEM_MB )); then
      continue
    fi

    if (( gpu_util < MAX_IDLE_UTIL )) && (( mem_free > best_strict_mem )); then
      best_strict_gpu="${gpu_id}"
      best_strict_mem="${mem_free}"
    fi

    if (( mem_free > best_fallback_mem )); then
      best_fallback_gpu="${gpu_id}"
      best_fallback_mem="${mem_free}"
    fi
  done

  if [[ -n "${best_strict_gpu}" ]]; then
    SELECTED_GPU="${best_strict_gpu}"
    return 0
  fi

  if [[ -n "${best_fallback_gpu}" ]]; then
    SELECTED_GPU="${best_fallback_gpu}"
    return 0
  fi

  return 1
}

wait_for_idle_gpu() {
  while true; do
    cleanup_finished_jobs
    if [[ "${stop_launching}" -eq 1 ]]; then
      return 1
    fi
    if pick_idle_gpu; then
      return 0
    fi
    sleep "${POLL_SEC}"
  done
}

gpu_index=0
for lambda_value in "${LAMBDA_VALUES[@]}"; do
  declare -a WORKER_COMBO_GROUPS
  for (( worker_slot=0; worker_slot<TOTAL_WORKER_SLOTS; worker_slot++ )); do
    WORKER_COMBO_GROUPS[$worker_slot]=""
  done

  combo_index=0
  for combo_id in "${COMBO_IDS[@]}"; do
    worker_slot=$(( combo_index % TOTAL_WORKER_SLOTS ))
    if [[ -z "${WORKER_COMBO_GROUPS[$worker_slot]}" ]]; then
      WORKER_COMBO_GROUPS[$worker_slot]="${combo_id}"
    else
      WORKER_COMBO_GROUPS[$worker_slot]="${WORKER_COMBO_GROUPS[$worker_slot]},${combo_id}"
    fi
    combo_index=$((combo_index + 1))
  done

  for (( worker_slot=0; worker_slot<TOTAL_WORKER_SLOTS; worker_slot++ )); do
    combo_group="${WORKER_COMBO_GROUPS[$worker_slot]}"
    if [[ -z "${combo_group}" ]]; then
      continue
    fi

    if [[ "${stop_launching}" -eq 1 ]]; then
      echo "[Launcher] fail-fast triggered; stop launching new workers."
      break 2
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      gpu_id="${GPU_IDS[$((gpu_index % ${#GPU_IDS[@]}))]}"
      gpu_index=$((gpu_index + 1))
    else
      wait_for_idle_gpu || break 2
      gpu_id="${SELECTED_GPU}"
    fi

    lambda_tag="$(printf '%s' "${lambda_value}" | sed 's/[^A-Za-z0-9_.-]/_/g')"
    if [[ -n "${RESULT_ROOT}" ]]; then
      lambda_result_root="${RESULT_ROOT}_m_${lambda_tag}"
    else
      lambda_result_root=""
    fi
    job_log="${RUN_DIR}/logs/worker_lambda_${lambda_tag}_slot_${worker_slot}_gpu_${gpu_id}.log"
    cmd=(bash "${TOPK_SCRIPT}" --g "${gpu_id}" --combos "${combo_group}" --lambda "${lambda_value}")
    if [[ -n "${MODEL_PATH}" ]]; then
      cmd+=(--model-path "${MODEL_PATH}")
    fi
    if [[ -n "${ATTRIBUTION_CACHE_DIR}" ]]; then
      cmd+=(--attribution-cache-dir "${ATTRIBUTION_CACHE_DIR}")
    fi
    if [[ -n "${lambda_result_root}" ]]; then
      cmd+=(--result-root "${lambda_result_root}")
    fi
    if [[ -n "${PROMPT_LIST}" ]]; then
      cmd+=(--prompt-list "${PROMPT_LIST}")
    fi
    cmd+=(--top-k-1 "${TOP_K_1[*]}")
    cmd+=(--top-k-2 "${TOP_K_2[*]}")
    cmd+=(--state-dir "${WORKER_STATE_DIR}")
    if [[ "${FAIL_FAST}" -eq 1 ]]; then
      cmd+=(--fail-fast)
    fi
    if [[ "${FORCE_RERUN}" -eq 1 ]]; then
      cmd+=(--force)
    fi

    echo "[Launcher] launch worker: combos=${combo_group} lambda=${lambda_value} gpu=${gpu_id} result_root=${lambda_result_root:-<auto>} log=${job_log}"
    printf '[Launcher] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      continue
    fi

    (
      echo "[Launcher] START worker combos=${combo_group} lambda=${lambda_value} gpu=${gpu_id} time=$(date)"
      printf '[Launcher] command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
      status=$?
      echo "[Launcher] END worker combos=${combo_group} lambda=${lambda_value} gpu=${gpu_id} status=${status} time=$(date)"
      exit "${status}"
    ) > "${job_log}" 2>&1 &

    pid=$!
    PID_TO_COMBO["${pid}"]="${combo_group}"
    PID_TO_GPU["${pid}"]="${gpu_id}"
    PID_TO_LOG["${pid}"]="${job_log}"
    PID_TO_KEY["${pid}"]="worker"
    PID_TO_BRAND1["${pid}"]="multiple"
    PID_TO_BRAND2["${pid}"]="multiple"
    PID_TO_LAMBDA["${pid}"]="${lambda_value}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "LAUNCH" \
      "${pid}" \
      "${gpu_id}" \
      "${combo_group}" \
      "worker" \
      "multiple" \
      "multiple" \
      "${lambda_value}" \
      "RUNNING" \
      "${job_log}" >> "${job_trace_path}"

    print_active_jobs

    if [[ "${STAGGER_SEC}" -gt 0 ]]; then
      sleep "${STAGGER_SEC}"
    fi
  done
done

while (( $(running_jobs_count) > 0 )); do
  cleanup_finished_jobs
  if (( $(running_jobs_count) > 0 )); then
    print_active_jobs
    sleep "${POLL_SEC}"
  fi
done

total_launched=$((success_count + failed_count))
echo "[Launcher] done"
echo "[Launcher] run_dir=${RUN_DIR}"
echo "[Launcher] worker_launched=${total_launched}"
echo "[Launcher] worker_success=${success_count}"
echo "[Launcher] worker_failed=${failed_count}"
echo "[Launcher] worker_state_dir=${WORKER_STATE_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[Launcher] dry-run mode: no jobs were actually launched"
fi

if (( failed_count > 0 )); then
  exit 1
fi
