#!/bin/bash

set -e

# ================= Path Configuration =================
DATA_BASE_DIR=../1-data_preprocess
OUTPUT_DIR=../outputs/finetuned_best

# Ensure this path is correct and contains backbone_config.py
PRETRAIN_CHECKPOINTS=../outputs/pretrain_checkpoints/xr_mpm_10x300/xr_on_mpm_10x300_pretrain_step_120000.pth
config_path="../outputs/pretrain_checkpoints/xr_mpm_10x300/backbone_config.py"

# ================= Training Parameters =================
LEARNING_RATE=1e-3
BATCH_SIZE=32
EPOCHS=20
WARMUP_EPOCHS=10
FREEZE_MODE=none
SEED=42

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# =======================================================

echo "Starting batch Fine-tuning..."
echo "Data Source: $DATA_BASE_DIR"
echo "Results Output: $OUTPUT_DIR"
echo "Pre-trained Model: $PRETRAIN_CHECKPOINTS"
echo "---------------------------------------"

# Define execution function
run_finetune() {
    local index=$1
    local name=$2
    local data_subdir=$3
    local output_subdir=$4
    local custom_warmup=${5:-$WARMUP_EPOCHS}
    local custom_seed=${6:-$SEED}

    # Construct paths
    local data_dir="$DATA_BASE_DIR/${data_subdir}_h5"
    local output_path="$OUTPUT_DIR/$output_subdir"
    local save_path="$output_path/finetuned_best_${data_subdir}.pth"

    echo "========== [$index] Fine-tuning dataset: $name =========="
    
    if [ ! -d "$data_dir" ]; then
        echo "Error: Data directory $data_dir not found. Skipping."
        return
    fi

    mkdir -p "$output_path"

    # Execute Python script
    python ../finetune/finetune.py \
        --data_dir "$data_dir" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --save_path "$save_path" \
        --gpus $CUDA_VISIBLE_DEVICES \
        --warmup_epochs $custom_warmup \
        --freeze_mode $FREEZE_MODE \
        --seed $custom_seed \
        --pretrain_path "$PRETRAIN_CHECKPOINTS" \
        --config_path "$config_path"

    echo "Completed: $name -> Results saved to $save_path"
    echo ""
}

# --- Start Loop Execution ---

PREFIX=10x300_no_mpm
run_finetune "8" "CSTNET" "cstnet" "${PREFIX}_cstnet"
run_finetune "10" "DataCon2021-Part1" "datacon2021_part1" "${PREFIX}_datacon2021_part1"
run_finetune "12" "AES-128-GCM" "aes_128_gcm" "${PREFIX}_aes_128_gcm"
run_finetune "13" "AES-256-GCM" "aes_256_gcm" "${PREFIX}_aes_256_gcm"
run_finetune "14" "ChaCha20-Poly1305" "chacha20_poly1305" "${PREFIX}_chacha20_poly1305"
run_finetune "15" "Mix" "mix" "${PREFIX}_mix"

echo "---------------------------------------"
echo "All fine-tuning tasks completed!"