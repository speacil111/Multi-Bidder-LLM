#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,7-8,16-17,25,27-31] --gpus=2 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x

source activate SVD

export CUDA_VISIBLE_DEVICES=0,1

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888

python neuron_moe.py --hilton-neuron-count 1500 \
                     --hilton-multiplier 2.5 \
                     --delta-neuron-count 1500 \
                     --delta-multiplier 2.5 \
                     --parallel-gpus 0,1 --enable_Hilton --enable_Delta \
                     --ig_steps 5 \
                     --delta-score-mode contrastive \
                     --hilton-score-mode contrastive \
                     --threshold 0.000 \
                     --intervention_layer -1 \
                     --hilton-fusion-weight 0.5 \
                     --delta-fusion-weight 0.5
