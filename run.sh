#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
LOG_FILE="logs/run_$(date +"%Y-%m-%d_%H-%M-%S").txt"

python neuron_test.py --hilton-neuron-count 250 \
                     --hilton-multiplier 3.5 \
                     --delta-neuron-count 250 \
                     --delta-multiplier 2.5 \
                     --parallel-gpus 0,1 --enable_Hilton \
                     --enable_Delta \
                     --ig_steps 20 \
                     --force-recompute-attribution \
                     --monitor \
                     --delta-score-mode contrastive \
                     --hilton-score-mode contrastive \
                     --threshold 0.000 \
                     --attribution-cache-dir "attr_score_cache" \
                     --intervention_layer -1 2>&1 | tee "$LOG_FILE"
