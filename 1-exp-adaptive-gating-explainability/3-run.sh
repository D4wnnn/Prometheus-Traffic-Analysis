#!/bin/bash

python 3-visualize_heatmap_all.py \
    --data_path ./gating_analysis/aes_128_gcm_gating.npz \
    --output_dir ./imgs/heatmap_all \
    --num_vis_samples 5