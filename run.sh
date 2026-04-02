#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,5
LOG_FILE="run_logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --concepts Toyota,Costco \
                     --enable_1 --enable_2 \
                     --top_k_1 500 \
                     --multiplier_1 3.0 \
                     --top_k_2 500 \
                     --multiplier_2 3.0 \
                     --parallel-gpus 0,1 \
                     --ig_steps 20 \
                     --monitor \
                     --baseline \
                     --max-new-tokens 1536 \
                     --prompt-index 3 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --threshold 0.000 \
                     --unified-hook \
                     --attribution-cache-dir "attr_score_cache" \
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"