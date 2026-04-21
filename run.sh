#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run.sh [--gpu-id N] [--combo-preset-id N] [--ig-steps N] [--attribution-cache-dir DIR] [--model-path PATH] [--log-file PATH] [--log-dir DIR]

Options:
  --g N, --gpu-id N            Override GPU_ID (default: 4)
  --c N, --combo-preset-id N   Override COMBO_PRESET_ID (default: 0)
  --ig-steps N                 Override IG_STEPS (default: 5)
  --attribution-cache-dir DIR  Override attribution cache dir (default: attr_cache_ds)
  --model-path PATH            Override model path (default: ../DS_r1_8B)
  --log-file PATH              Write tee output to PATH
  --log-dir DIR                Log directory when --log-file is not provided (default: ./run_logs)
  -h, --help                   Show this help message
EOF
}

# Default config: edit these values directly when you want fixed defaults.
DEFAULT_GPU_ID="4"
DEFAULT_COMBO_PRESET_ID="3"
DEFAULT_IG_STEPS="5"
DEFAULT_ATTRIBUTION_CACHE_DIR="attr_cache_ds"
DEFAULT_MODEL_PATH="../DS_r1_8B"
DEFAULT_LOG_DIR="run_logs"
DEFAULT_LOG_FILE=""

GPU_ID="${GPU_ID:-$DEFAULT_GPU_ID}"
COMBO_PRESET_ID="${COMBO_PRESET_ID:-$DEFAULT_COMBO_PRESET_ID}"
IG_STEPS="${IG_STEPS:-$DEFAULT_IG_STEPS}"
ATTRIBUTION_CACHE_DIR="${ATTRIBUTION_CACHE_DIR:-$DEFAULT_ATTRIBUTION_CACHE_DIR}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
LOG_DIR="${LOG_DIR:-$DEFAULT_LOG_DIR}"
LOG_FILE="${LOG_FILE:-$DEFAULT_LOG_FILE}"

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
    --c|--combo-preset-id)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a value" >&2; usage >&2; exit 1; }
      COMBO_PRESET_ID="$2"
      shift 2
      ;;
    --c=*|--combo-preset-id=*)
      COMBO_PRESET_ID="${1#*=}"
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
    --log-file)
      [[ $# -ge 2 ]] || { echo "[ERROR] --log-file requires a value" >&2; usage >&2; exit 1; }
      LOG_FILE="$2"
      shift 2
      ;;
    --log-file=*)
      LOG_FILE="${1#*=}"
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

if [[ -z "${LOG_FILE}" ]]; then
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/run_combo_${COMBO_PRESET_ID}_gpu_${GPU_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt"
else
  mkdir -p "$(dirname "${LOG_FILE}")"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python neuron_test.py --model_path "${MODEL_PATH}" \
                     --attribution-cache-dir "${ATTRIBUTION_CACHE_DIR}" \
                     --combo-preset "${COMBO_PRESET_ID}" \
                     --enable_1 --enable_2 \
                     --top_k_1 500 \
                     --multiplier_1 2.0 \
                     --top_k_2 500 \
                     --multiplier_2 2.0 \
                     --parallel-gpus "${GPU_ID}" \
                     --ig_steps "${IG_STEPS}" \
                     --baseline \
                     --monitor \
                     --mind_bridge \
                     --max-new-tokens 1536 \
                     --prompt-index 0 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --unified-hook \
                     --intervention_layer -1 2>&1 | tee "${LOG_FILE}"
