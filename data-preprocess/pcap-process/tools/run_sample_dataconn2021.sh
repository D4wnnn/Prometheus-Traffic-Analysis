#/bin/bash


python sample_part1.py \
    -i /raid/lc/datasets/initial_datasets/finetune/DataCon2020/train \
    -o /raid/lc/datasets/processed_datasets/finetune/DataCon2020-processed \
    --ratios 0.75 0.15 0.15 \
    --clear \
    --max 1000 \
    --seed 42

# 已完成
# python sample_part1.py \
#     -i /raid/lc/datasets/initial_datasets/finetune/DataCon2021/part1/organized_real_data \
#     -o /raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part1/ \
#     --ratios 0.75 0.15 0.15 \
#     --clear \
#     --max 1000 \
#     --seed 42

# python sample_part1.py \
#     -i /raid/lc/datasets/initial_datasets/finetune/DataCon2021/part2/organized_test_data \
#     -o /raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part2/ \
#     --ratios 0.75 0.15 0.15 \
#     --clear \
#     --max 1000 \
#     --seed 42