#/bin/bash

python split_dataset.py \
  --input /raid/lc/datasets/initial_datasets/finetune/aes-128-gcm \
  --output /raid/lc/datasets/processed_datasets/finetune/AES-128-GCM-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15 \

python split_dataset.py \
  --input /raid/lc/datasets/initial_datasets/finetune/aes-256-gcm \
  --output /raid/lc/datasets/processed_datasets/finetune/AES-256-GCM-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15 \

python split_dataset.py \
  --input /raid/lc/datasets/initial_datasets/finetune/chacha20-poly1305 \
  --output /raid/lc/datasets/processed_datasets/finetune/chacha20-poly1305-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15 \

python split_dataset.py \
  --input /raid/lc/datasets/initial_datasets/finetune/mix \
  --output /raid/lc/datasets/processed_datasets/finetune/mix-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15 \