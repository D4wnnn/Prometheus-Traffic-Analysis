#!/bin/bash

set -e

# ================= Global Configuration =================
DATA_BASE_DIR="../1-data_preprocess"
TEACHER_BASE_DIR="../outputs/finetuned_best"
SELECTOR_DIR="../outputs/global_selectors"
OUTPUT_BASE_DIR="../outputs/light_model_best"

# Training Parameters
GPUS="0,1,2,3,4,5,6,7"
BATCH_SIZE=64
EPOCHS=30
LR=2e-3
ALPHA=0.5
TEMPERATURE=4.0

# Teacher Configuration (for loading Teacher architecture)
TEACHER_PACKETS=10
TEACHER_BYTES=300

# ========================================================

echo "=========================================="
echo "Batch Distillation with Global Selection"
echo "=========================================="

# Define execution function
run_distillation() {
    local index=$1
    local name=$2
    local data_subdir=$3
    local teacher_subdir=$4
    local t_bytes=${5:-$TEACHER_BYTES} # Optional: Teacher byte count, default is 300
    local byte_strategy="per_packet" # Optional: selection strategy (contiguous, per_packet, greedy)
    local select_config="5x3x100"

    # Construct paths
    local data_dir="${DATA_BASE_DIR}/${data_subdir}_h5"
    local teacher_path="${TEACHER_BASE_DIR}/${teacher_subdir}/finetuned_best_${data_subdir}_swa.pth"
    local global_selector="${SELECTOR_DIR}/${data_subdir}_${byte_strategy}_${select_config}.npz"
    local save_dir="${OUTPUT_BASE_DIR}/${data_subdir}_global_${byte_strategy}_${select_config}_distilled"

    echo "========== [$index] Distilling: $name =========="
    echo "Selector: $global_selector"
    
    if [ ! -f "$global_selector" ]; then
        echo "⚠️  Error: Selector file not found at $global_selector. Skipping..."
        return
    fi

    mkdir -p "$save_dir"

    python ../distillation/train_light_offline_ddp_select.py \
      --data_dir "$data_dir" \
      --teacher_path "$teacher_path" \
      --teacher_packets $TEACHER_PACKETS \
      --teacher_bytes $t_bytes \
      --save_dir "$save_dir" \
      --global_selector "$global_selector" \
      --gpus $GPUS \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --lr $LR \
      --alpha $ALPHA \
      --temperature $TEMPERATURE \
      --student_aug

    echo "Done -> Model saved to $save_dir"
    echo ""
}


# Task Execution
run_distillation "12" "AES-128-GCM" "aes_128_gcm" "aes_128_gcm"
run_distillation "13" "AES-256-GCM" "aes_256_gcm" "aes_256_gcm"
run_distillation "14" "ChaCha20-Poly1305" "chacha20_poly1305" "chacha20_poly1305"
run_distillation "15" "Mix" "mix" "mix"
run_distillation "8" "CSTNET" "cstnet" "cstnet"
run_distillation "10" "DataCon2021-Part1" "datacon2021_part1" "datacon2021_part1"

echo "---------------------------------------"
echo "All distillation tasks completed!"