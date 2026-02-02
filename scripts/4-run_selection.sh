#!/bin/bash

set -e

# ================= Global Configuration =================
DATA_BASE_DIR="../1-data_preprocess"
TEACHER_BASE_DIR="../outputs/finetuned_best"
OUT_DIR="../outputs/global_selectors"

# Student Target Geometry (Parameters)
TOP_PACKETS=5
WINDOW_LEN=5
WINDOWS_PER_PACKET=3
# Total Bytes = 50 * 3 = 150

# Efficiency Parameters
ANALYZE_RATIO=1  # Analyze 100% of training data (set to e.g., 0.3 if data is massive)
BATCH_SIZE=64
DEVICE="cuda:0"

# ========================================================

mkdir -p $OUT_DIR

echo "=========================================="
echo "Batch Building Global Importance Selectors"
echo "=========================================="
echo "Target: ${TOP_PACKETS} packets x $(($WINDOW_LEN * $WINDOWS_PER_PACKET)) bytes"
echo "Output Dir: $OUT_DIR"
echo ""

# Define execution function
run_selection() {
    local index=$1
    local name=$2
    local data_subdir=$3
    local teacher_subdir=$4 # Sub-directory where the Teacher model is located
    local byte_strategy="contiguous" # Optional: byte selection strategy, default is contiguous
    local data_dir="${DATA_BASE_DIR}/${data_subdir}_h5"
    local teacher_path="${TEACHER_BASE_DIR}/${teacher_subdir}/finetuned_best_${data_subdir}_swa.pth"
    local out_file="${OUT_DIR}/${data_subdir}_${byte_strategy}_${TOP_PACKETS}x${WINDOWS_PER_PACKET}x${WINDOW_LEN}.npz"

    echo "========== [$index] Processing: $name =========="
    echo "Data: $data_dir"
    echo "Teacher: $teacher_path"

    if [ ! -f "$teacher_path" ]; then
        echo "⚠️  Warning: Teacher model not found at $teacher_path. Skipping..."
        echo ""
        return
    fi

    # Strategy: per_packet (independent selection per packet), greedy, or contiguous
    python ../distillation/build_global_importance.py \
      --data_dir "$data_dir" \
      --h5_name train_data.h5 \
      --teacher_path "$teacher_path" \
      --out_path "$out_file" \
      --top_packets $TOP_PACKETS \
      --window_len $WINDOW_LEN \
      --windows_per_packet $WINDOWS_PER_PACKET \
      --byte_strategy per_packet \
      --analyze_ratio $ANALYZE_RATIO \
      --batch_size $BATCH_SIZE \
      --device $DEVICE \
    #  --visualize

    echo "Done -> Saved to $out_file"
    echo ""
}

# --- Start Loop Execution ---

run_selection "1" "CSTNET" "cstnet" "cstnet"
run_selection "2" "AES-128-GCM" "aes_128_gcm" "aes_128_gcm"
run_selection "3" "AES-256-GCM" "aes_256_gcm" "aes_256_gcm"
run_selection "4" "ChaCha20-Poly1305" "chacha20_poly1305" "chacha20_poly1305"
run_selection "5" "Mix" "mix" "mix"
run_selection "6" "DataCon2021-Part1" "datacon2021_part1" "datacon2021_part1"

echo "=========================================="
echo "All selectors built!"
ls -la $OUT_DIR