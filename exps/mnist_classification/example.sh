#!/bin/bash

DIR_PREFIX="/home/david_tyuman/my_github/dl_dev"
cd $DIR_PREFIX || exit 1


SPLIT_RAW_DATASET="\
./dl split \
    --src /var/lib/storage/data/benchmarks/computer_vision/mnist/raw.h5 \
    --dst0 /var/lib/storage/data/benchmarks/computer_vision/mnist/wrangled/raw_train.h5 \
    --dst1 /var/lib/storage/data/benchmarks/computer_vision/mnist/wrangled/raw_val.h5 \
    --th 0.95 \
    --mode random
"

WRANGLE_THE_TRAIN="./dl wrangle_dataset --config ./exps/mnist_classification/configs/train_dataset_wrangling.json"

WRANGLE_THE_VAL="./dl wrangle_dataset --config ./exps/mnist_classification/configs/val_dataset_wrangling.json"

START_LEARNING="\
./dl train \
    --net_factory_function_impl ./exps/mnist_classification/configs/net_factory_function.py \
    --learning_config ./exps/mnist_classification/configs/learning_process.json \
    --log-level TRACE
"


$SPLIT_RAW_DATASET && $WRANGLE_THE_TRAIN && $WRANGLE_THE_VAL && (rm mnist.log ; $START_LEARNING 2>&1 | tee mnist.log)
