set -e

OUTPUT_DIR=../outputs/pretrain_checkpoints/10x300
mkdir -p $OUTPUT_DIR

python ../pretrain/pretrain.py \
    --data_file /path/to/your/data_file.h5 \
    --batch_size 16 \
    --epochs 20 \
    --lr 1e-4 \
    --mask_prob 0.25 \
    --save_dir $OUTPUT_DIR \
    --gpus 0,1,2,3,4,5,6,7
