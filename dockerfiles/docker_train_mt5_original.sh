#!/bin/bash


[ $# -ge 2 ] || { echo "Usage: $0 input_data_dir/ output_dir/ [multi]"; exit 1; }

# Use baselines/mt5/prepare_dataset.py to prepare the input data
#  prepared_dataset_dir/
#  		├── individual
#  		│   ├── cantonese
#  		│   │   ├── test.source
#  		│   │   ├── test.target
#  		│   │   ├── train.source
#  		│   │   ├── train.target
#  		│   │   ├── val.source
#  		│   │   └── val.target
#  		│   └── EN
#  		│       ├── test.source
#  		│       ├── test.target
#  		│       ├── train.source
#  		│       ├── train.target
#  		│       ├── val.source
#  		│       └── val.target
#  		└── multilingual
#  				├── cantonese_train.source
#  				├── cantonese_train.target
#  				├── cantonese_val.source
#  				├── cantonese_val.target
#  				├── EN_train.source
#  				├── EN_train.target
#  				├── EN_val.source
#  				├── EN_val.target
#  				├── val.source
#  				└── val.target
#
# Use prepared_dataset_dir/individual/<langname>/ as input dir for single lang mode
# and prepared_dataset_dir/multilingual/ as input dir for multilingual mode


data_dir=$(readlink -ve $1) || { echo "Error! Exiting..."; exit 1; }
output_dir=$(readlink -m $2)
mode=${3:-"single"}

mkdir -p $output_dir
chmod -R 777 $output_dir

python="/opt/conda/envs/MultiDocMultiLingual/bin/python"
docker_image="zs12/multidoc_multilingual:v0.3.1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-"0"}

if [[ "$mode" == "single" ]]; then
  echo "train single langauge mt5 model..."
  set -x
  export num_train_epochs=20
  export PER_DEVICE_TRAIN_BATCH_SIZE=8
  export PER_DEVICE_EVAL_BATCH_SIZE=16
  export GRADIENT_ACC=4
  export num_train=236
  docker run --rm -it --gpus all -v $data_dir:/data/ -v $output_dir:/output/ \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $docker_image \
    $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
      --model_name_or_path "google/mt5-base" \
      --local_rank -1 \
      --data_dir /data/ \
      --output_dir /output/ \
      --learning_rate 5e-4 \
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
      --do_train True \
      --logging_first_step \
      --metric_for_best_model rouge2 \
      --greater_is_better True \
      --n_val 500 \
      --warmup_steps $((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/10)) \
      --weight_decay 0.01 \
      --label_smoothing_factor 0.1
fi

if [[ "$mode" == *"multi"* ]]; then
  echo "train multilingual mt5 model..."
  set -x
  export PER_DEVICE_TRAIN_BATCH_SIZE=8
  export GRADIENT_ACC=16
  export MAX_STEP=20000
  docker run --rm -it --gpus all -v $data_dir:/data/ -v $output_dir:/output/ \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $docker_image \
    $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
      --model_name_or_path "google/mt5-base" \
      --local_rank -1 \
      --data_dir /data/ \
      --output_dir /output/ \
      --learning_rate 5e-5 \
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
      --do_train True \
      --logging_first_step \
      --warmup_steps 2000 
fi
