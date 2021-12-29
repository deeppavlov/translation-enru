#!/bin/bash

base_model_dir="base_ruen"
output_dir="output_ruen"
train_file_url="https://files.deeppavlov.ai/datasets/raw_train.json"
train_file="data/raw_train.json"
validation_file="data/raw_valid_small.json"

# download train file if it wasn't done previously
if ! [ -f $train_file ]; then
    wget -P ./data $train_file_url
fi

# create output dir if not exists
if ! [ -d $output_dir ]; then
    mkdir $output_dir
fi

# if output_dir path is relative, convert to absolute
if ! [[ "$output_dir" = /* ]]; then
    output_dir=$(cd $output_dir; pwd)
fi

# initialize base model weights from scratch if it wasn't done previously
if ! [ -d $base_model_dir ]; then
   python create_base_model.py --create_ruen
fi

# if base_model_dir path is relative, convert to absolute
if ! [[ "$base_model_dir" = /* ]]; then
    base_model_dir=$(cd $base_model_dir; pwd)
fi


python run_translation.py \
    --model_name_or_path $base_model_dir \
    --do_train \
    --do_eval \
    --source_lang ru \
    --target_lang en \
    --train_file $train_file \
    --validation_file $validation_file \
    --output_dir $output_dir \
    --fp16 \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --num_train_epochs=10 \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=10 \
    --save_total_limit=3 \
    --max_source_length=192 \
    --max_target_length=192 \
    --overwrite_output_dir
