#!/bin/bash

# conda env
source /mnt/data1/zyx/miniconda3/bin/activate infedit

export CUDA_VISIBLE_DEVICES=1

python infedit.py
