#!/bin/bash

set -e

# ================= Path Configuration =================
DATA_BASE_DIR=../1-data_preprocess
MODEL_DIR=../outputs/finetuned_best
RESULT_DIR=../outputs/eval_results
SCRIPT_PATH=../eval/eval_only_heavy.py
PRETRAIN_CHECKPOINTS=../outputs/pretrain_checkpoints/10x300/10x300_pretrain_epoch_7.pth


# ================= Evaluation Parameters =================
BATCH_SIZE=64
GPU_ID=0

# =========================================================

echo "Starting batch Evaluation..."
echo "Data Source: $DATA_BASE_DIR"
echo "Model Directory: $MODEL_DIR"
echo "Results Output: $RESULT_DIR"
echo "---------------------------------------"

# Define execution function
run_eval() {
    local index=$1
    local name=$2
    local data_subdir=$3
    local model_subdir=$4
    local model_filename=$5

    # Construct paths
    local data_dir="$DATA_BASE_DIR/${data_subdir}_h5"
    local model_path="$MODEL_DIR/$model_subdir/$model_filename"
    local result_path="$RESULT_DIR/$model_subdir"
    local save_file="$result_path/results_heavy_only_${data_subdir}.json"

    echo "========== [$index] Evaluating dataset: $name =========="
    
    if [ ! -d "$data_dir" ]; then
        echo "Error: Data directory $data_dir not found. Skipping."
        return
    fi

    if [ ! -f "$model_path" ]; then
        echo "Error: Model file $model_path not found. Skipping."
        return
    fi

    mkdir -p "$result_path"

    # Execute Python script
    python $SCRIPT_PATH \
        --data_dir "$data_dir" \
        --heavy_path "$model_path" \
        --batch_size $BATCH_SIZE \
        --gpu $GPU_ID \
        --save_result "$save_file" \
        --pretrain_path "$PRETRAIN_CHECKPOINTS" \

    echo "Completed: $name -> Results saved to $save_file"
    echo ""
}


run_eval "1" "CSTNET" "cstnet" "cstnet" "finetuned_best_cstnet.pth"
run_eval "2" "DataCon2021-Part1" "datacon2021_part1" "datacon2021_part1" "finetuned_best_datacon2021_part1_swa.pth"
run_eval "3" "AES-128-GCM" "aes_128_gcm" "aes_128_gcm" "finetuned_best_aes_128_gcm_swa.pth"
run_eval "4" "AES-256-GCM" "aes_256_gcm" "aes_256_gcm" "finetuned_best_aes_256_gcm_swa.pth"
run_eval "5" "ChaCha20-Poly1305" "chacha20_poly1305" "chacha20_poly1305" "finetuned_best_chacha20_poly1305_swa.pth"
run_eval "6" "Mix" "mix" "mix" "finetuned_best_mix_swa.pth"


echo "---------------------------------------"
echo "All evaluation tasks completed! Results saved to $RESULT_DIR"