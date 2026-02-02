#!/bin/bash

# =========================================================
# Experiment Runner for Multi-Modal Protocol Understanding
# (Supports Multiple Datasets)
# =========================================================

# --- General Configuration ---
PRETRAIN_PATH="../../outputs/pretrain_checkpoints/10*300/10x300_pretrain_epoch_7.pth"
OUTPUT_DIR="./probe_results"
BATCH_SIZE=64
EPOCHS=2
FEW_SHOT=1.0  # Use 100% of the data
GPU_ID=1

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Experiment Start: $TIMESTAMP"
echo "Pretrain: $PRETRAIN_PATH"
echo "=========================================="

# --- Define Execution Function ---
run_experiment() {
    local DATA_NAME=$1
    local DATA_PATH=$2
    local SAVE_FILE="${OUTPUT_DIR}/results_${DATA_NAME}_${TIMESTAMP}.json"

    echo ""
    echo "------------------------------------------"
    echo "Running Probe Tasks on Dataset: $DATA_NAME"
    echo "Path: $DATA_PATH"
    echo "Output: $SAVE_FILE"
    echo "------------------------------------------"

    python probe_runner.py \
        --data_file "$DATA_PATH" \
        --pretrain_path "$PRETRAIN_PATH" \
        --task "all" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --few_shot_ratio $FEW_SHOT \
        --save_results "$SAVE_FILE" \
        --gpu $GPU_ID

    if [ $? -eq 0 ]; then
        echo "✓ Finished $DATA_NAME"
    else
        echo "✗ Failed $DATA_NAME"
    fi
}

# --- Execute Experiments (Add multiple datasets here) ---

run_experiment "AES_128" "../../1-data_preprocess/aes_128_gcm_h5/test_data.h5"
run_experiment "AES_256" "../../1-data_preprocess/aes_256_gcm_h5/test_data.h5"
run_experiment "ChaCha20" "../../1-data_preprocess/chacha20_poly1305_h5/test_data.h5"
run_experiment "Mixed" "../../1-data_preprocess/mix_h5/test_data.h5"
run_experiment "CSTNET" "../../1-data_preprocess/cstnet_h5/test_data.h5"
run_experiment "Datacon2021_part1" "../../1-data_preprocess/datacon2021_part1_h5/test_data.h5"

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="