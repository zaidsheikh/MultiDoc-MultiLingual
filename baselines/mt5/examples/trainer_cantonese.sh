#!/bin/bash
export num_train_epochs=20
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export PER_DEVICE_EVAL_BATCH_SIZE=16
export GRADIENT_ACC=4
export lr=5e-4
export language="cantonese"
export num_train=236
export OUTPUT_DIR="./baseline_results/clean_dataset/first_sentences/single_languae/lr_${lr}_ada_all_epoch_${num_train_epochs}_bs_${PER_DEVICE_TRAIN_BATCH_SIZE}_acc_${GRADIENT_ACC}/${language}/"
export DATA_DIR="./data/first_sentences/individual/${language}"

python baselines/mt5/pipeline.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir $DATA_DIR \
    --output_dir  $OUTPUT_DIR \
    --learning_rate $lr \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --num_train_epochs $num_train_epochs \
    --logging_steps $((num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC)) \
    --save_steps $((num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC)) \
    --eval_steps $((num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC)) \
    --adafactor \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE  \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --rouge_lang "chinese" \
    --logging_first_step \
    --metric_for_best_model rouge2 \
    --greater_is_better True \
    --n_val 500 \
    --warmup_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/10)) \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1

python baselines/mt5/choose_best_on_val_set.py $OUTPUT_DIR

python baselines/mt5/pipeline.py \
    --model_name_or_path "${OUTPUT_DIR}/best_ckpt/" \
    --data_dir $DATA_DIR \
    --output_dir  "${OUTPUT_DIR}/test/" \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_predict \
    --rouge_lang "chinese"
