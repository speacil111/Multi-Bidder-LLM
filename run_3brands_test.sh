#!/bin/bash
#SBATCH --gpus=1
# 以下参数只有本地测试时需要
module load compilers/cuda/12.1 compilers/gcc/9.3.0
module load cudnn/8.8.1.3_cuda12.x

source activate SVD

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
set -euo pipefail


GPU_ID="0"
MODEL_PATH="../Qwen3-4B"
ATTRIBUTION_CACHE_DIR="./attr_cache_qwen"
COMBO_PRESET="adobe_dell_logitech"
IG_STEPS="20"
LOG_DIR="./run_logs_3brands"
LOG_FILE="${LOG_DIR}/run_3brands_${COMBO_PRESET}_gpu_${GPU_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt"

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
                     --monitor \
                     --max-new-tokens 1024 \
                     --prompt-index 2 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --score_mode_3 contrastive \
                     --unified-hook \
                     --intervention_layer -1 \
                     --system-prompt "You are a helpful assistant. Answer my question in 512 tokens." \
                     
