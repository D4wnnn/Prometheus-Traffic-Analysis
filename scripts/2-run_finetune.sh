#!/bin/bash

set -e

# ================= Path Configuration =================
DATA_BASE_DIR=../1-data_preprocess
OUTPUT_DIR=../outputs/finetuned_best

PRETRAIN_CHECKPOINTS=../outputs/pretrain_checkpoints/10x300/10x300_pretrain_epoch_7.pth
config_path=../outputs/pretrain_checkpoints/10x300/backbone_config.py

LEARNING_RATE=1e-3
BATCH_SIZE=16
EPOCHS=20
WARMUP_EPOCHS=10
FREEZE_MODE=none
SEED=42
DISABLE_HEAD=""
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ======================================================

echo "Starting batch Fine-tuning..."
echo "Data Source: $DATA_BASE_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Pre-trained Model: $PRETRAIN_CHECKPOINTS"
echo "---------------------------------------"

# Define execution function
run_finetune() {
    local index=$1
    local name=$2
    local data_subdir=$3
    local output_subdir=$4
    local disable_heads=${5:-$DISABLE_HEAD}
    local custom_warmup=${6:-$WARMUP_EPOCHS}
    local custom_seed=${7:-$SEED}

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
        --config_path "$config_path" \
        --pretrain_path "$PRETRAIN_CHECKPOINTS"
    echo "Completed: $name -> Results saved to $save_path"
    echo ""
}

run_finetune "1" "CSTNET" "cstnet" "cstnet"
run_finetune "2" "DataCon2021-Part1" "datacon2021_part1" "datacon2021_part1"
run_finetune "3" "AES-128-GCM" "aes_128_gcm" "aes_128_gcm"
run_finetune "4" "AES-256-GCM" "aes_256_gcm" "aes_256_gcm"
run_finetune "5" "ChaCha20-Poly1305" "chacha20_poly1305" "chacha20_poly1305"
run_finetune "6" "Mix" "mix" "mix"


echo "---------------------------------------"
echo "All fine-tuning tasks completed!"