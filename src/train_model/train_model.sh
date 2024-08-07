#!/bin/bash

python train.py \
    --data_dir "data" \
    --classes 20 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --device "cuda" \
    --seed 42 \
    --wandb_project "bird-classification" \
    --model_name "bird_classifier" \
    --save_frequency 1 \
    --refresh_data \

