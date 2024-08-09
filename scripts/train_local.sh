#!/bin/bash

cd ./src/train_model || { echo "Directory not found"; exit 1; }
bash train.sh