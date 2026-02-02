#!/bin/bash

set -e

# ================= Global Configuration =================
DATA_BASE_DIR="../1-data_preprocess"
# Note: The INPUT directory here corresponds to the OUTPUT of script #5
LIGHT_MODEL_BASE_DIR="../outputs/light_model_best" 
SELECTOR_DIR="../outputs/global_selectors"
RESULT_BASE_DIR="../outputs/eval_results_light"

# Evaluation Parameters
TEACHER_PACKETS=10
TEACHER_BYTES_DEFAULT=300
BATCH_SIZE=64
GPU_ID=0

echo "========================================"
echo "Batch Running Light-Only Baseline Eval"
echo "========================================"

# Define execution function
run_eval() {
    local index=$1
    local name=$2
    local data_subdir=$3
    local t_bytes=${4:-$TEACHER_BYTES_DEFAULT}
    
    # Strategy Config (Must match script #5)
    local byte_strategy="per_packet"  # options: per_packet, greedy, contiguous
    local select_config="5x3x50"

    # Construct paths
    local data_dir="${DATA_BASE_DIR}/${data_subdir}_h5"
    
    # 1. Selector Path (Mapping to script #5 naming convention)
    local global_selector="${SELECTOR_DIR}/${data_subdir}_${byte_strategy}_${select_config}.npz"
    
    # 2. Light Model Path (Mapping to script #5 save_dir)
    local light_model_dir="${LIGHT_MODEL_BASE_DIR}/${data_subdir}_global_${byte_strategy}_${select_config}_distilled"
    local light_path="${light_model_dir}/light_model_global.pth"
    
    # 3. Results Save Path (Strategy-specific to prevent overwriting)
    local result_file="${RESULT_BASE_DIR}/${data_subdir}/results_light_${byte_strategy}_${select_config}.json"

    # Create result directory
    mkdir -p "$(dirname "$result_file")"

    echo "========== [$index] Evaluating: $name =========="
    echo "Selector: $global_selector"
    echo "Model:    $light_path"
    
    # Check if model exists
    if [ ! -f "$light_path" ]; then
        echo "⚠️  Error: Light model not found at $light_path." 
        echo "    (Ensure Script 5 finished successfully for this config)"
        echo "Skipping..."
        return
    fi
    
    # Check if selector exists
    if [ ! -f "$global_selector" ]; then
        echo "⚠️  Error: Selector file not found at $global_selector. Skipping..."
        return
    fi

    python ../eval/eval_only_light_select.py \
        --data_dir "$data_dir" \
        --light_path "$light_path" \
        --batch_size $BATCH_SIZE \
        --gpu $GPU_ID \
        --save_result "$result_file" \
        --teacher_packets $TEACHER_PACKETS \
        --teacher_bytes $t_bytes \
        --global_selector "$global_selector"

    echo "✓ Result saved to: $result_file"
    echo ""
}

# --- Start Loop Execution ---

run_eval "12" "AES-128-GCM" "aes_128_gcm"

echo "---------------------------------------"
echo "All evaluation tasks completed!"