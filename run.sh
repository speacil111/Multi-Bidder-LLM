#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
LOG_FILE="run_logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --combo-preset 0 \
                     --enable_1 --enable_2 \
                     --top_k_1 750 \
                     --multiplier_1 3.0 \
                     --top_k_2 750 \
                     --multiplier_2 3.0 \
                     --parallel-gpus 0 \
                     --ig_steps 20 \
                     --monitor \
                     --baseline \
                     --max-new-tokens 1536 \
                     --prompt-index 1 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --unified-hook \
                     --attribution-cache-dir "attr_cache_4B" \
                     --mind_bridge \
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"