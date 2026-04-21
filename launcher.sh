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
  --max-jobs N          Max concurrent jobs. Default: number of GPUs
  --topk-script PATH    Sweep script to call. Default: ./topk_sweep.sh
  --log-dir DIR         Launcher output dir. Default: ./batch_runs
  --stagger-sec N       Sleep N seconds between launches. Default: 2
  --min-free-mem-mb N   Treat GPU as selectable only if memory.free >= N. Default: 15000
  --idle-max-util N     Prefer GPU with utilization.gpu < N. Default: 70
  --poll-sec N          Sleep N seconds while waiting for an idle GPU. Default: 5
  --fail-fast           Stop launching new jobs after first failure
  --dry-run             Print commands without launching
  -h, --help            Show this help

Examples:
  bash launcher.sh --combos 0-9 --gpus 0,1,2,3
  bash launcher.sh --combos 10-29,35 --gpus 0,1 --max-jobs 2
  bash launcher.sh --combo-file combo_ids.txt --gpus 4,5,6,7 --fail-fast

Notes:
  1. This launcher is a batch wrapper around topk_sweep.sh.
  2. It does not change your experiment logic; it only schedules many combo ids.
  3. Current topk_sweep.sh writes outputs under paths like logp_token_<BRAND_1>_m2.0.
     If many combos reuse the same first brand, those outputs may collide or overwrite.
     So this launcher is best viewed as a scheduling template until run_root is made combo-unique.
EOF
}

TOPK_SCRIPT="./topk_sweep_batch.sh"
LOG_DIR="./batch_runs"
# 直接在这里定义默认任务（可被命令行参数覆盖）
# 例: COMBOS="0-9,12,18" ; GPUS_LIST="0,1,2,3" 或 "0-7"
COMBOS="0,1"
GPUS_LIST="0-7"
COMBO_SPEC="${COMBOS}"
COMBO_FILE=""
GPU_SPEC="${GPUS_LIST}"
MAX_JOBS=""
STAGGER_SEC=2
MIN_FREE_MEM_MB=15000
POLL_SEC=5
MAX_IDLE_UTIL=70
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
    --topk-script)
      [[ $# -ge 2 ]] || { echo "[ERROR] --topk-script requires a value" >&2; usage >&2; exit 1; }
      TOPK_SCRIPT="$2"
      shift 2
      ;;
    --topk-script=*)
      TOPK_SCRIPT="${1#*=}"
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
  MAX_JOBS="${#GPU_IDS[@]}"
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

manifest_path="${RUN_DIR}/manifest.txt"
cat > "${manifest_path}" <<EOF
timestamp=${timestamp}
topk_script=${TOPK_SCRIPT}
defined_combos=${COMBOS}
defined_gpus_list=${GPUS_LIST}
combo_spec=${COMBO_SPEC}
combo_file=${COMBO_FILE}
combo_ids=${COMBO_IDS[*]}
gpu_ids=${GPU_IDS[*]}
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

echo "[Launcher] manifest: ${manifest_path}"
echo "[Launcher] combo count: ${#COMBO_IDS[@]}"
echo "[Launcher] gpus: ${GPU_IDS[*]}"
echo "[Launcher] max_jobs: ${MAX_JOBS}"
if [[ "${NVIDIA_SMI_AVAILABLE}" -eq 1 ]]; then
  echo "[Launcher] idle-gpu mode: prefer memory.free>=${MIN_FREE_MEM_MB}MB and utilization.gpu<${MAX_IDLE_UTIL}%, fallback to highest memory.free with memory.free>=${MIN_FREE_MEM_MB}MB"
else
  echo "[Launcher] idle-gpu mode: nvidia-smi unavailable, fallback to launcher-only GPU tracking"
fi

job_trace_path="${RUN_DIR}/job_trace.tsv"
{
  printf "event_time\tevent\tpid\tgpu\tcombo_id\tcombo_key\tbrand_1\tbrand_2\tstatus\tjob_log\n"
} > "${job_trace_path}"
echo "[Launcher] job trace: ${job_trace_path}"

declare -A PID_TO_COMBO
declare -A PID_TO_GPU
declare -A PID_TO_LOG
declare -A PID_TO_KEY
declare -A PID_TO_BRAND1
declare -A PID_TO_BRAND2

success_count=0
failed_count=0
stop_launching=0
SELECTED_GPU=""

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
  echo "[Launcher] active jobs (${n}):"
  for pid in "${!PID_TO_COMBO[@]}"; do
    echo "  pid=${pid} gpu=${PID_TO_GPU[$pid]} combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} log=${PID_TO_LOG[$pid]}"
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
      echo "[Launcher] finished: combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} gpu=${PID_TO_GPU[$pid]} status=OK log=${PID_TO_LOG[$pid]}"
    else
      failed_count=$((failed_count + 1))
      status="FAIL"
      echo "[Launcher] finished: combo=${PID_TO_COMBO[$pid]} key=${PID_TO_KEY[$pid]} brand_1=${PID_TO_BRAND1[$pid]} brand_2=${PID_TO_BRAND2[$pid]} gpu=${PID_TO_GPU[$pid]} status=FAIL log=${PID_TO_LOG[$pid]}"
      if [[ "${FAIL_FAST}" -eq 1 ]]; then
        stop_launching=1
      fi
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "FINISH" \
      "${pid}" \
      "${PID_TO_GPU[$pid]}" \
      "${PID_TO_COMBO[$pid]}" \
      "${PID_TO_KEY[$pid]}" \
      "${PID_TO_BRAND1[$pid]}" \
      "${PID_TO_BRAND2[$pid]}" \
      "${status}" \
      "${PID_TO_LOG[$pid]}" >> "${job_trace_path}"
    unset 'PID_TO_COMBO[$pid]'
    unset 'PID_TO_GPU[$pid]'
    unset 'PID_TO_LOG[$pid]'
    unset 'PID_TO_KEY[$pid]'
    unset 'PID_TO_BRAND1[$pid]'
    unset 'PID_TO_BRAND2[$pid]'
  done
}

gpu_in_use_by_launcher() {
  local target_gpu="$1"
  local pid
  for pid in "${!PID_TO_GPU[@]}"; do
    if [[ "${PID_TO_GPU[$pid]}" == "${target_gpu}" ]]; then
      return 0
    fi
  done
  return 1
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

  if (( $(running_jobs_count) >= MAX_JOBS )); then
    return 1
  fi

  if [[ "${NVIDIA_SMI_AVAILABLE}" -ne 1 ]]; then
    for gpu_id in "${GPU_IDS[@]}"; do
      if gpu_in_use_by_launcher "${gpu_id}"; then
        continue
      fi
      SELECTED_GPU="${gpu_id}"
      return 0
    done
    return 1
  fi

  for gpu_id in "${GPU_IDS[@]}"; do
    if gpu_in_use_by_launcher "${gpu_id}"; then
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
    echo "[Launcher] fail-fast triggered; stop launching new jobs."
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
  cmd=(bash "${TOPK_SCRIPT}" --g "${gpu_id}" --c "${combo_id}")
  combo_key="${COMBO_TO_KEY[${combo_id}]}"
  combo_brand_1="${COMBO_TO_BRAND1[${combo_id}]}"
  combo_brand_2="${COMBO_TO_BRAND2[${combo_id}]}"

  echo "[Launcher] launch: combo=${combo_id} key=${combo_key} brand_1=${combo_brand_1} brand_2=${combo_brand_2} gpu=${gpu_id} log=${job_log}"
  printf '[Launcher] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi

  (
    echo "[Launcher] START combo=${combo_id} gpu=${gpu_id} time=$(date)"
    printf '[Launcher] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    status=$?
    echo "[Launcher] END combo=${combo_id} gpu=${gpu_id} status=${status} time=$(date)"
    exit "${status}"
  ) > "${job_log}" 2>&1 &

  pid=$!
  PID_TO_COMBO["${pid}"]="${combo_id}"
  PID_TO_GPU["${pid}"]="${gpu_id}"
  PID_TO_LOG["${pid}"]="${job_log}"
  PID_TO_KEY["${pid}"]="${combo_key}"
  PID_TO_BRAND1["${pid}"]="${combo_brand_1}"
  PID_TO_BRAND2["${pid}"]="${combo_brand_2}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "LAUNCH" \
    "${pid}" \
    "${gpu_id}" \
    "${combo_id}" \
    "${combo_key}" \
    "${combo_brand_1}" \
    "${combo_brand_2}" \
    "RUNNING" \
    "${job_log}" >> "${job_trace_path}"

  print_active_jobs

  if [[ "${STAGGER_SEC}" -gt 0 ]]; then
    sleep "${STAGGER_SEC}"
  fi
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
echo "[Launcher] launched=${total_launched}"
echo "[Launcher] success=${success_count}"
echo "[Launcher] failed=${failed_count}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[Launcher] dry-run mode: no jobs were actually launched"
fi
