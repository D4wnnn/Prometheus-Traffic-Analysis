#!/bin/bash
# VPN-service 数据集

BASE_INPUT_DIR="/raid/lc/datasets/initial_datasets/finetune"
BASE_OUTPUT_DIR="/raid/lc/datasets/processed_datasets/finetune"

mkdir -p $BASE_OUTPUT_DIR




python main.py \
    --input /mnt/8T/lc/datasets/initial_datasets/finetune/CICIDS2017/processed \
    --output /mnt/8T/lc/datasets/processed_datasets/finetune/CICIDS2017-processed \
    --clear \
    --min-category-flows 15




# echo "========== [1/6] 处理 VPN-service =========="
# python main.py \
#     --input $BASE_INPUT_DIR/ISCX-VPN-NonVPN-2016/VPN-service \
#     --output $BASE_OUTPUT_DIR/ISCX-VPN-NonVPN-2016/VPN-service-processed \
#     --clear \
#     --min-category-flows 15

# VPN-app 数据集
# echo "========== [2/6] 处理 VPN-app =========="
# python main.py \
#     --input $BASE_INPUT_DIR/ISCX-VPN-NonVPN-2016/VPN-app \
#     --output $BASE_OUTPUT_DIR/ISCX-VPN-NonVPN-2016/VPN-app-processed \
#     --clear \
#     --min-category-flows 15
    # --filter-rule "not (arp or dhcp or ntp or icmp or dns) and (tcp or udp) and frame.len > 80"


# # USTC-TFC 数据集
# echo "========== [3/6] 处理 USTC-TFC =========="
# python main.py \
#     --input $BASE_INPUT_DIR/USTC-TFC2016 \
#     --output $BASE_OUTPUT_DIR/USTC-TFC2016-processed \
#     --clear \
#     --min-category-flows 15 \
#     --filter-rule "not (arp or dhcp or ntp or icmp or dns) and (tcp or udp) and frame.len > 80"

# # Cross-Platform-Android 数据集
# echo "========== [4/6] 处理 Cross-Platform-Android =========="
# python main.py \
#     --input $BASE_INPUT_DIR/CrossPlatform/CrossPlatform-recongnized/android \
#     --output $BASE_OUTPUT_DIR/Cross-Platform-android-processed \
#     --clear \
#     --min-category-flows 15

# # Cross-Platform-IOS 数据集
# echo "========== [5/6] 处理 Cross-Platform-IOS =========="
# python main.py \
#     --input $BASE_INPUT_DIR/CrossPlatform/CrossPlatform-recongnized/ios \
#     --output $BASE_OUTPUT_DIR/Cross-Platform-ios-processed \
#     --clear \
#     --min-category-flows 8
# 5太难了 15太简单
# # Browser-Dataset 数据集
# echo "========== [6/6] 处理 Browser-Dataset =========="
# python main.py \
#     --input $BASE_INPUT_DIR/Browser \
#     --output $BASE_OUTPUT_DIR/Browser-processed \
#     --clear \
#     --min-category-flows 15

# # DataCon2020 数据集
# echo "========== [7/7] 处理 DataCon2020 =========="
# rm -rf $BASE_OUTPUT_DIR/DataCon2020-processed
# python main.py \
#     --input $BASE_INPUT_DIR/DataCon2020/train/ \
#     --output $BASE_OUTPUT_DIR/DataCon2020-processed/tmp/train/ \
#     --clear \
#     --min-category-flows 15 \
#     --train-ratio 0.8 \
#     --val-ratio 0.2 \
#     --test-ratio 0

# python main.py \
#     --input $BASE_INPUT_DIR/DataCon2020/test \
#     --output $BASE_OUTPUT_DIR/DataCon2020-processed/tmp/test/ \
#     --clear \
#     --min-category-flows 15 \
#     --train-ratio 0 \
#     --val-ratio 0 \
#     --test-ratio 1

# mv $BASE_OUTPUT_DIR/DataCon2020-processed/tmp/train/train $BASE_OUTPUT_DIR/DataCon2020-processed/train/
# mv $BASE_OUTPUT_DIR/DataCon2020-processed/tmp/train/val $BASE_OUTPUT_DIR/DataCon2020-processed/val/
# mv $BASE_OUTPUT_DIR/DataCon2020-processed/tmp/test/test $BASE_OUTPUT_DIR/DataCon2020-processed/test/
# rm -rf $BASE_OUTPUT_DIR/DataCon2020-processed/tmp/

# # DataCon2021-Part1 数据集
# echo "========== [7/7] 处理 DataCon2021-part1 =========="

# python main.py \
#     --input $BASE_INPUT_DIR/DataCon2021/part1/organized_real_data \
#     --output $BASE_OUTPUT_DIR/DataCon2021-processed/part1 \
#     --clear \
#     --min-category-flows 15 \

# # DataCon2021-Part2 数据集
# echo "========== [7/7] 处理 DataCon2021-part2 =========="
# rm -rf $BASE_OUTPUT_DIR/DataCon2021-processed/part2/
# python main.py \
#     --input $BASE_INPUT_DIR/DataCon2021/part2/organized_train_data \
#     --output $BASE_OUTPUT_DIR/DataCon2021-processed/part2/tmp/train \
#     --clear \
#     --min-category-flows 15 \
#     --train-ratio 0.8 \
#     --val-ratio 0.2 \
#     --test-ratio 0

# python main.py \
#     --input $BASE_INPUT_DIR/DataCon2021/part2/organized_test_data \
#     --output $BASE_OUTPUT_DIR/DataCon2021-processed/part2/tmp/test/ \
#     --clear \
#     --min-category-flows 15 \
#     --train-ratio 0 \
#     --val-ratio 0 \
#     --test-ratio 1

# mv $BASE_OUTPUT_DIR/DataCon2021-processed/part2/tmp/train/train $BASE_OUTPUT_DIR/DataCon2021-processed/part2/train/
# mv $BASE_OUTPUT_DIR/DataCon2021-processed/part2/tmp/train/val $BASE_OUTPUT_DIR/DataCon2021-processed/part2/val/
# mv $BASE_OUTPUT_DIR/DataCon2021-processed/part2/tmp/test/test $BASE_OUTPUT_DIR/DataCon2021-processed/part2/test/
# rm -rf $BASE_OUTPUT_DIR/DataCon2021-processed/part2/tmp/


echo ""
echo "所有数据集处理完成!"