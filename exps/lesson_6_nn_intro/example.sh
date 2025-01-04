#!/bin/bash

DIR_PREFIX="/home/david_tyuman/my_github/dl_dev"
cd $DIR_PREFIX || exit 1


SPLIT_RAW_DATASET="\
./dl split \
    --src /var/lib/storage/data/cv_fall_2022/lesson_6_nn_intro/raw_labeled.h5 \
    --dst0 /var/lib/storage/data/cv_fall_2022/lesson_6_nn_intro/wrangled/raw_train.h5 \
    --dst1 /var/lib/storage/data/cv_fall_2022/lesson_6_nn_intro/wrangled/raw_val.h5 \
    --th 0.9 \
    --mode random \
"
START_LEARNING="\
./dl train \
    --net_factory_function_impl ./exps/lesson_6_nn_intro/configs/net_factory_function.py \
    --learning_config ./exps/lesson_6_nn_intro/configs/learning_process.json \
    --log-level DEBUG \
"


$SPLIT_RAW_DATASET && (rm lesson6.log ; $START_LEARNING 2>&1 | tee lesson6.log)
