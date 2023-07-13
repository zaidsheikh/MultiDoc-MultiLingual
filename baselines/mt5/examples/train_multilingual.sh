#!/bin/sh
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACC=16
export MAX_STEP=20000

export lr=5e-5

export OUTPUT_DIR="./baseline_results/clean_dataset/first_sentences/multi_language/lr_${lr}_linear_schedual_ada_all_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}_max_steps_${MAX_STEP}/"
export DATA_DIR="./data/first_sentences/multilingual"

python baselines/mt5/pipeline.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir $DATA_DIR \
    --output_dir  $OUTPUT_DIR \
    --predict_with_generate \
    --learning_rate $lr \
    --upsampling_factor 0.5 \
    --label_smoothing_factor 0.1 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --max_steps $MAX_STEP \
    --logging_steps 100 \
    --save_steps 2000 \
    --adafactor \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --overwrite_output_dir \
    --evaluation_strategy "no" \
    --predict_with_generate \
    --do_train \
    --logging_first_step \
    --warmup_steps 2000 