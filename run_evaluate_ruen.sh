#!/bin/bash

model_name='DeepPavlov/marianmt-tatoeba-ruen'
output_dir="output_ruen"
validation_file="data/test.json"


if ! [ -d $output_dir ]; then
    mkdir $output_dir
fi


if ! [[ "$output_dir" = /* ]]; then
    output_dir=$(cd $output_dir; pwd)
fi


python run_translation.py \
    --model_name_or_path $model_name \
    --do_eval \
    --source_lang ru \
    --target_lang en \
    --validation_file $validation_file \
    --output_dir $output_dir \
    --fp16 \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --num_train_epochs=10 \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=12 \
    --save_total_limit=3 \
    --max_source_length=192 \
    --max_target_length=192 \
    --overwrite_output_dir \
    --predict_with_generate
 
