#!/bin/bash

# Process AES-128-GCM dataset
python split_dataset.py \
  --input /path/to/initial_datasets/finetune/aes-128-gcm \
  --output /path/to/processed_datasets/finetune/AES-128-GCM-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15

# Process AES-256-GCM dataset
python split_dataset.py \
  --input /path/to/initial_datasets/finetune/aes-256-gcm \
  --output /path/to/processed_datasets/finetune/AES-256-GCM-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15

# Process ChaCha20-Poly1305 dataset
python split_dataset.py \
  --input /path/to/initial_datasets/finetune/chacha20-poly1305 \
  --output /path/to/processed_datasets/finetune/chacha20-poly1305-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15

# Process mixed dataset
python split_dataset.py \
  --input /path/to/initial_datasets/finetune/mix \
  --output /path/to/processed_datasets/finetune/mix-processed \
  --train 0.7 \
  --val 0.15 \
  --test 0.15