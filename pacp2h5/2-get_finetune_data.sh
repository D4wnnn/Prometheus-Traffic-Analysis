#!/bin/bash

BASE_DIR="path/to/datasets"

echo "Starting batch dataset preprocessing..."
echo "Using Base Directory: $BASE_DIR"
echo ""

echo "========== [1/6] Processing CSTNET =========="
python preprocess_finetune_data.py \
    --input "${BASE_DIR}/finetune/CSTNET" \
    --output ./cstnet_h5

echo "========== [2/6] Processing DataCon2021 (Part 1) =========="
python preprocess_finetune_data.py \
    --input "${BASE_DIR}/finetune/DataCon2021-processed/part1" \
    --output ./datacon2021_part1_h5

echo "========== [3/6] Processing AES-128-GCM =========="
python preprocess_finetune_data.py \
    --input "${BASE_DIR}/finetune/AES-128-GCM-processed" \
    --output ./aes_128_gcm_h5

echo "========== [4/6] Processing AES-256-GCM =========="
python preprocess_finetune_data.py \
    --input "${BASE_DIR}/finetune/AES-256-GCM-processed" \
    --output ./aes_256_gcm_h5

echo "========== [5/6] Processing ChaCha20-Poly1305 =========="
python preprocess_finetune_data.py \
    --input "${BASE_DIR}/finetune/chacha20-poly1305-processed" \
    --output ./chacha20_poly1305_h5

# --- 4. Mixed Dataset ---
echo "========== [6/6] Processing Mixed Dataset =========="
python preprocess_finetune_data.py \
    --input "${BASE_DIR}/finetune/mix-processed" \
    --output ./mix_h5

echo ""
echo "All tasks completed successfully!"