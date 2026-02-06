#/bin/bash
python organize_dataconn2021_part1.py

python sample_part1.py \
    -i /path/to/organized_real_data \
    -o /path/to/output/ \
    --ratios 0.75 0.15 0.15 \
    --clear \
    --max 1000 \
    --seed 42