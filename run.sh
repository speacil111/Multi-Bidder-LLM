#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
LOG_FILE="logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --hilton-neuron-count 1000 \
                     --hilton-multiplier 2.0 \
                     --delta-neuron-count 100 \
                     --delta-multiplier 2.0 \
                     --parallel-gpus 0,1 --enable_Delta --enable_Hilton \
                     --ig_steps 20 \
                     --monitor \
                     --max-new-tokens 1024 \
                     --delta-score-mode contrastive \
                     --hilton-score-mode contrastive \
                     --threshold 0.000 \
                     --attribution-cache-dir "attr_score_cache" \
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"
