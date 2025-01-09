#!/bin/bash

DIR_PREFIX="/home/david_tyuman/my_github/dl"
cd $DIR_PREFIX || exit 1


# 78543340 and 5498 rows
SPLIT_RAW_DATASET_TO_TRAIN_AND_DEV="\
./dl split \
    --src /var/lib/storage/data/benchmarks/machine_translation/wmt/wmt23_ruen/raw.h5 \
    --dst0 /var/lib/storage/data/benchmarks/machine_translation/wmt/wmt23_ruen/wrangled/raw_train.h5 \
    --dst1 /var/lib/storage/data/benchmarks/machine_translation/wmt/wmt23_ruen/wrangled/raw_val.h5 \
    --th 0.99993 \
    --mode random
"

# 2749 and 2749 rows
SPLIT_DEV_RAW_DATASET_TO_TWO_VALS="\
./dl split \
    --src /var/lib/storage/data/benchmarks/machine_translation/wmt/wmt23_ruen/wrangled/raw_val.h5 \
    --dst0 /var/lib/storage/data/benchmarks/machine_translation/wmt/wmt23_ruen/wrangled/raw_val_0.h5 \
    --dst1 /var/lib/storage/data/benchmarks/machine_translation/wmt/wmt23_ruen/wrangled/raw_val_1.h5 \
    --th 0.5 \
    --mode random
"


false && $SPLIT_RAW_DATASET_TO_TRAIN_AND_DEV && $SPLIT_DEV_RAW_DATASET_TO_TWO_VALS
