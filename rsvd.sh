#!/bin/bash
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x

source activate SVD

export CUDA_VISIBLE_DEVICES=0,1,2,3

export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:8888

python svd.py