#!/bin/bash

# conda env
eval "$(conda shell.bash hook)"
conda activate infedit

export CUDA_VISIBLE_DEVICES=1
export TMPDIR=`pwd`/tmp
mkdir ${TMPDIR} -p

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# delete cache images
rm -rf inter

python app_infedit.py
