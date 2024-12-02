#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vedit

export CUDA_VISIBLE_DEVICES=0
python app_infedit.py