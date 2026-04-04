#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,5
LOG_FILE="run_logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --concepts Delta,Hilton \
                     --enable_1 --enable_2 \
                     --top_k_1 0 \
                     --multiplier_1 2.5 \
                     --top_k_2 50 \
                     --multiplier_2 2.5 \
                     --parallel-gpus 0,1 \
                     --ig_steps 20 \
                     --monitor \
                     --baseline \
                     --max-new-tokens 1536 \
                     --prompt-index 6 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --unified-hook \
                     --attribution-cache-dir "attr_score_cache" \
                     --mind_bridge\
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"