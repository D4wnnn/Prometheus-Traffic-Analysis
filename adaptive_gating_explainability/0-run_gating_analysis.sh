#!/bin/bash

python visualize_gating_weights.py \
    --data_path ./gating_analysis/aes_128_gcm_gating.npz \
    --output_dir ./gating_analysis/aes_128_gcm_figures \
    --max_samples 25
