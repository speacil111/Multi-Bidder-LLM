#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
LOG_FILE="run_logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --combo-preset 1 \
                     --enable_1 --enable_2 \
                     --top_k_1 500 \
                     --multiplier_1 2.5 \
                     --top_k_2 500 \
                     --multiplier_2 3.5 \
                     --parallel-gpus 0 \
                     --ig_steps 20 \
                     --monitor \
                     --max-new-tokens 1536 \
                     --prompt-index 0 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --unified-hook \
                     --attribution-cache-dir "attr_cache_new" \
                     --mind_bridge \
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"