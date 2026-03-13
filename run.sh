#!/bin/bash
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x

source activate SVD

export CUDA_VISIBLE_DEVICES=0,1

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888

python neuron_test.py --hilton-top-percent 0.5 \
                     --hilton-multiplier 2.0 \
                     --delta-top-percent 0.5 \
                     --delta-multiplier 2.0 \
                     --parallel-gpus 0,1 --enable_Hilton --enable_Delta \
                     --ig_steps 20 \
                     --delta-score-mode contrastive \
                     --hilton-score-mode contrastive
