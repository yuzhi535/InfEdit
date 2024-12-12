#!/bin/bash

# conda env
eval "$(conda shell.bash hook)"
conda activate infedit

export CUDA_VISABLE_DEVICES=1
export TMPDIR=`pwd`/tmp
mkdir ${TMPDIR} -p

python app_infedit.py
