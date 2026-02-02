#!/bin/bash

python 2-visualize_gating_weights_layer1-4.py \
    --data_path ./gating_analysis/aes_128_gcm_gating.npz \
    --output_dir ./imgs/gata_weights_aes_128_gcm \
    --max_samples 25