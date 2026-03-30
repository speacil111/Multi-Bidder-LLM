#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5
LOG_FILE="run_logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --combo-preset 0 \
                     --enable_1 --enable_2 \
                     --attr_sum_1 10.0\
                     --multiplier_1 2.25 \
                     --attr_sum_2 10.0 \
                     --multiplier_2 2.25 \
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
                     --intervention_layer -1 \
                     2>&1 | tee "$LOG_FILE"