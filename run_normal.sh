#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

python neuron_test.py --hilton-top-percent 0.0001 --hilton-multiplier 2.0 \
                     --delta-top-percent 0.0001 --delta-multiplier 2.0 \
                     --parallel-gpus 0,1 --enable_Hilton --enable_Delta \
                     --ig_steps 3 \
                     --delta-score-mode contrastive \
                     --hilton-score-mode contrastive
