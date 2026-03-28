#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5
LOG_FILE="run_logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --combo-preset delta_hilton \
                     --enable_1 --enable_2 \
                     --neuron_count_1 500 \
                     --multiplier_1 2.0 \
                     --neuron_count_2 500 \
                     --multiplier_2 2.0 \
                     --parallel-gpus 0,1 \
                     --ig_steps 20 \
                     --monitor \
                     --max-new-tokens 1536 \
                     --prompt-index 0 \
                     --score_mode_1 contrastive \
                     --score_mode_2 contrastive \
                     --threshold 0.000 \
                     --unified-hook \
                     --attribution-cache-dir "attr_score_cache" \
                     --attribution-cache-path "attr_score_cache/attribution_Hilton_Hotel-Delta_Airline_ig20_new.pt" \
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"
