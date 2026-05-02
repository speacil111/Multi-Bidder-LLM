#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID="0"
MODEL_PATH="../Qwen3-4B"
ATTRIBUTION_CACHE_DIR="/Users/speacil/Desktop/37copy/Multi-Bidder-LLM/attr_cache_qwen"
COMBO_PRESET="delta_hyatt_visa"
IG_STEPS="20"
LOG_DIR="./run_logs_3brands"
LOG_FILE="${LOG_DIR}/run_3brands_${COMBO_PRESET}_gpu_${GPU_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt"

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python neuron_test.py --model_path "${MODEL_PATH}" \
                     --attribution-cache-dir "${ATTRIBUTION_CACHE_DIR}" \
                     --combo-preset "${COMBO_PRESET}" \
                     --enable_1 --enable_2 --enable_3 \
                     --top_k_1 500 \
                     --multiplier_1 2.0 \
                     --top_k_2 500 \
                     --multiplier_2 2.0 \
                     --top_k_3 500 \
                     --multiplier_3 2.0 \
                     --parallel-gpus "${GPU_ID}" \
                     --ig_steps "${IG_STEPS}" \
                     --baseline \
                     --monitor \
                     --max-new-tokens 1536 \
                     --prompt-index 0 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --score_mode_3 contrastive \
                     --unified-hook \
                     --intervention_layer -1 2>&1 | tee "${LOG_FILE}"
