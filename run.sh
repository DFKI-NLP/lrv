#!/usr/bin/env bash
now=$(date +"%Y-%m-%d-%H-%M-%S")
touch ./data/logs/$now.log
git rev-parse HEAD >> ./data/logs/$now.log
CUDA_VISIBLE_DEVICES=0 python main.py 2>&1 | tee -a ./data/logs/$now.log
