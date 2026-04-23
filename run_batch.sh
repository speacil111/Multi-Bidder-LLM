#!/bin/bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_batch.sh [options]
  bash run_batch.sh --combos SPEC --gpus GPU_LIST [options]
  bash run_batch.sh --combo-file FILE --gpus GPU_LIST [options]

Default behavior:
  1. Auto-select idle GPUs from GPUS_LIST.
  2. Run ./run.sh for combos selected by COMBOS, or all combos in src.config.COMBO_PRESETS when COMBOS is empty.

Options:
  --gpus LIST, --gpus_list LIST
                        GPU id spec, e.g. 0,1,2,3 or 0-7 or 0-3,6
  --combos SPEC         Combo id spec, e.g. 0-9,12,18,20-25
  --combo-file FILE     File with one combo id per line; lines starting with # are ignored
  --max-jobs N          Max concurrent jobs. Default: GPU count * MAX_JOBS_PER_GPU
  --run-script PATH     Script to call. Default: ./run.sh
  --log-dir DIR         Batch output dir. Default: ./batch_runs
  --ig-steps N          Pass IG steps to run.sh. Default: value of IG_STEPS below
  --attribution-cache-dir DIR
                        Pass attribution cache dir to run.sh
  --model-path PATH     Pass model path to run.sh
  --stagger-sec N       Sleep N seconds between launches. Default: 2
  --min-free-mem-mb N   Treat GPU as selectable only if memory.free >= N. Default: 15000
  --idle-max-util N     Prefer GPU with utilization.gpu < N. Default: 70
  --poll-sec N          Sleep N seconds while waiting for an idle GPU. Default: 5
  --fail-fast           Stop launching new jobs after first failure
  --dry-run             Print commands without launching
  -h, --help            Show this help

Examples:
  bash run_batch.sh
  bash run_batch.sh --gpus 0-3 --combos 0-9
  bash run_batch.sh --combo-file combo_ids.txt --gpus 4,5,6,7 --fail-fast
EOF
}

RUN_SCRIPT="./run.sh"
LOG_DIR="./qwen_attr_batch_runlogs"
IG_STEPS="20"
ATTRIBUTION_CACHE_DIR="attr_cache_qwen"
MODEL_PATH="../Qwen3-4B"
COMBOS="36-39"
GPUS_LIST="0-7"
MAX_JOBS_PER_GPU="3"
COMBO_SPEC="${COMBOS}"
COMBO_FILE=""
GPU_SPEC="${GPUS_LIST}"
MAX_JOBS=""
STAGGER_SEC=2
MIN_FREE_MEM_MB=15000
POLL_SEC=5
MAX_IDLE_UTIL=75
FAIL_FAST=0
DRY_RUN=0

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
    --run-script)
      [[ $# -ge 2 ]] || { echo "[ERROR] --run-script requires a value" >&2; usage >&2; exit 1; }
      RUN_SCRIPT="$2"
      shift 2
      ;;
    --run-script=*)
      RUN_SCRIPT="${1#*=}"
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
    --ig-steps)
      [[ $# -ge 2 ]] || { echo "[ERROR] --ig-steps requires a value" >&2; usage >&2; exit 1; }
      IG_STEPS="$2"
      shift 2
      ;;
    --ig-steps=*)
      IG_STEPS="${1#*=}"
      shift
      ;;
    --attribution-cache-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] --attribution-cache-dir requires a value" >&2; usage >&2; exit 1; }
      ATTRIBUTION_CACHE_DIR="$2"
      shift 2
      ;;
    --attribution-cache-dir=*)
      ATTRIBUTION_CACHE_DIR="${1#*=}"
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
  echo "[ERROR] Please provide --gpus/--gpus_list or define GPUS_LIST in run_batch.sh" >&2
  usage >&2
  exit 1
fi

if [[ -n "${COMBO_SPEC}" && -n "${COMBO_FILE}" ]]; then
  echo "[ERROR] Use either --combos or --combo-file, not both" >&2
  exit 1
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] run script not found: ${RUN_SCRIPT}" >&2
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
  elif [[ -n "${COMBO_FILE}" ]]; then
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
  else
    python - <<'PY'
from src.config import COMBO_PRESETS
for combo_id in range(len(COMBO_PRESETS)):
    print(combo_id)
PY
  fi
)

if [[ ${#COMBO_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] No combo ids to run" >&2
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
RUN_DIR="${LOG_DIR}/run_batch_${timestamp}"
mkdir -p "${RUN_DIR}/logs"

NVIDIA_SMI_AVAILABLE=0
if nvidia-smi --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  NVIDIA_SMI_AVAILABLE=1
fi

manifest_path="${RUN_DIR}/manifest.txt"
cat > "${manifest_path}" <<EOF
timestamp=${timestamp}
run_script=${RUN_SCRIPT}
defined_combos=${COMBOS}
combo_spec=${COMBO_SPEC}
combo_file=${COMBO_FILE}
combo_ids=${COMBO_IDS[*]}
gpu_ids=${GPU_IDS[*]}
max_jobs_per_gpu=${MAX_JOBS_PER_GPU}
ig_steps=${IG_STEPS}
attribution_cache_dir=${ATTRIBUTION_CACHE_DIR}
model_path=${MODEL_PATH}
max_jobs=${MAX_JOBS}
stagger_sec=${STAGGER_SEC}
min_free_mem_mb=${MIN_FREE_MEM_MB}
max_idle_util=${MAX_IDLE_UTIL}
poll_sec=${POLL_SEC}
nvidia_smi_available=${NVIDIA_SMI_AVAILABLE}
fail_fast=${FAIL_FAST}
dry_run=${DRY_RUN}
EOF

{
  echo "combo_mapping_begin"
  for combo_id in "${COMBO_IDS[@]}"; do
    echo "combo_${combo_id}=${COMBO_TO_KEY[${combo_id}]}|${COMBO_TO_BRAND1[${combo_id}]}|${COMBO_TO_BRAND2[${combo_id}]}"
  done
  echo "combo_mapping_end"
} >> "${manifest_path}"

echo "[RunBatch] manifest: ${manifest_path}"
echo "[RunBatch] combo count: ${#COMBO_IDS[@]}"
echo "[RunBatch] gpus: ${GPU_IDS[*]}"
echo "[RunBatch] max_jobs: ${MAX_JOBS}"
if [[ "${NVIDIA_SMI_AVAILABLE}" -eq 1 ]]; then
  echo "[RunBatch] idle-gpu mode: prefer memory.free>=${MIN_FREE_MEM_MB}MB and utilization.gpu<${MAX_IDLE_UTIL}%, fallback to highest memory.free with memory.free>=${MIN_FREE_MEM_MB}MB"
else
  echo "[RunBatch] idle-gpu mode: nvidia-smi unavailable, fallback to launcher-only GPU tracking"
fi

job_trace_path="${RUN_DIR}/job_trace.tsv"
{
  printf "event_time\tevent\tpid\tgpu\tcombo_id\tcombo_key\tbrand_1\tbrand_2\tstatus\tjob_log\trun_log\n"
} > "${job_trace_path}"
echo "[RunBatch] job trace: ${job_trace_path}"

declare -A PID_TO_COMBO
declare -A PID_TO_GPU
declare -A PID_TO_LOG
declare -A PID_TO_RUN_LOG
declare -A PID_TO_KEY
declare -A PID_TO_BRAND1
declare -A PID_TO_BRAND2

success_count=0
failed_count=0
stop_launching=0
SELECTED_GPU=""
LAST_ACTIVE_JOBS_SNAPSHOT=""

running_jobs_count() {
  local n=0
  local pid
  for pid in "${!PID_TO_COMBO[@]}"; do
    n=$((n + 1))
  done
  printf '%s\n' "${n}"
}

print_active_jobs() {
  local n
  local pid
  n="$(running_jobs_count)"
  if (( n == 0 )); then
    return
  fi
  echo "[RunBatch] active jobs (${n}):"
  for pid in "${!PID_TO_COMBO[@]}"; do
    echo "  pid=${pid} gpu=${PID_TO_GPU[$pid]} combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} log=${PID_TO_LOG[$pid]}"
  done
}

build_active_jobs_snapshot() {
  local pid
  local lines=()
  for pid in "${!PID_TO_COMBO[@]}"; do
    lines+=("${PID_TO_GPU[$pid]}|${PID_TO_COMBO[$pid]}|${PID_TO_KEY[$pid]}|${PID_TO_BRAND1[$pid]}|${PID_TO_BRAND2[$pid]}|${PID_TO_LOG[$pid]}|${pid}")
  done

  if [[ ${#lines[@]} -eq 0 ]]; then
    printf '\n'
    return
  fi

  printf '%s\n' "${lines[@]}" | sort
}

print_active_jobs_if_changed() {
  local snapshot
  snapshot="$(build_active_jobs_snapshot)"
  if [[ "${snapshot}" == "${LAST_ACTIVE_JOBS_SNAPSHOT}" ]]; then
    return
  fi
  LAST_ACTIVE_JOBS_SNAPSHOT="${snapshot}"
  if (( $(running_jobs_count) > 0 )); then
    print_active_jobs
  fi
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
      echo "[RunBatch] finished: combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} gpu=${PID_TO_GPU[$pid]} status=OK log=${PID_TO_LOG[$pid]}"
    else
      failed_count=$((failed_count + 1))
      status="FAIL"
      echo "[RunBatch] finished: combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} gpu=${PID_TO_GPU[$pid]} status=FAIL log=${PID_TO_LOG[$pid]}"
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
      "${status}" \
      "${PID_TO_LOG[$pid]}" \
      "${PID_TO_RUN_LOG[$pid]}" >> "${job_trace_path}"
    unset 'PID_TO_COMBO[$pid]'
    unset 'PID_TO_GPU[$pid]'
    unset 'PID_TO_LOG[$pid]'
    unset 'PID_TO_RUN_LOG[$pid]'
    unset 'PID_TO_KEY[$pid]'
    unset 'PID_TO_BRAND1[$pid]'
    unset 'PID_TO_BRAND2[$pid]'
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
for combo_id in "${COMBO_IDS[@]}"; do
  if [[ "${stop_launching}" -eq 1 ]]; then
    echo "[RunBatch] fail-fast triggered; stop launching new jobs."
    break
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    gpu_id="${GPU_IDS[$((gpu_index % ${#GPU_IDS[@]}))]}"
    gpu_index=$((gpu_index + 1))
  else
    wait_for_idle_gpu || break
    gpu_id="${SELECTED_GPU}"
  fi

  job_log="${RUN_DIR}/logs/combo_${combo_id}_gpu_${gpu_id}.log"
  run_log="${RUN_DIR}/logs/run_combo_${combo_id}_gpu_${gpu_id}.txt"
  cmd=(
    bash "${RUN_SCRIPT}"
    --g "${gpu_id}"
    --c "${combo_id}"
    --ig-steps "${IG_STEPS}"
    --attribution-cache-dir "${ATTRIBUTION_CACHE_DIR}"
    --model-path "${MODEL_PATH}"
    --log-file "${run_log}"
  )
  combo_key="${COMBO_TO_KEY[${combo_id}]}"
  combo_brand_1="${COMBO_TO_BRAND1[${combo_id}]}"
  combo_brand_2="${COMBO_TO_BRAND2[${combo_id}]}"

  echo "[RunBatch] launch: combo=${combo_id} key=${combo_key} brand_1=${combo_brand_1} brand_2=${combo_brand_2} gpu=${gpu_id} log=${job_log}"
  printf '[RunBatch] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi

  (
    echo "[RunBatch] START combo=${combo_id} gpu=${gpu_id} time=$(date)"
    printf '[RunBatch] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    status=$?
    echo "[RunBatch] END combo=${combo_id} gpu=${gpu_id} status=${status} time=$(date)"
    exit "${status}"
  ) > "${job_log}" 2>&1 &

  pid=$!
  PID_TO_COMBO["${pid}"]="${combo_id}"
  PID_TO_GPU["${pid}"]="${gpu_id}"
  PID_TO_LOG["${pid}"]="${job_log}"
  PID_TO_RUN_LOG["${pid}"]="${run_log}"
  PID_TO_KEY["${pid}"]="${combo_key}"
  PID_TO_BRAND1["${pid}"]="${combo_brand_1}"
  PID_TO_BRAND2["${pid}"]="${combo_brand_2}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "LAUNCH" \
    "${pid}" \
    "${gpu_id}" \
    "${combo_id}" \
    "${combo_key}" \
    "${combo_brand_1}" \
    "${combo_brand_2}" \
    "RUNNING" \
    "${job_log}" \
    "${run_log}" >> "${job_trace_path}"

  print_active_jobs_if_changed

  if [[ "${STAGGER_SEC}" -gt 0 ]]; then
    sleep "${STAGGER_SEC}"
  fi
done

while (( $(running_jobs_count) > 0 )); do
  cleanup_finished_jobs
  if (( $(running_jobs_count) > 0 )); then
    print_active_jobs_if_changed
    sleep "${POLL_SEC}"
  fi
done

total_launched=$((success_count + failed_count))
echo "[RunBatch] done"
echo "[RunBatch] run_dir=${RUN_DIR}"
echo "[RunBatch] launched=${total_launched}"
echo "[RunBatch] success=${success_count}"
echo "[RunBatch] failed=${failed_count}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[RunBatch] dry-run mode: no jobs were actually launched"
fi
