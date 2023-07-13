#!/bin/bash

[ $# -ge 3 ] || { echo "Usage: $0 input_data_dir/ output_dir/ clearml_dataset_id [multi]"; exit 1; }

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
# Run clearml_scripts/upload_models.sh and use the generated ID as $clearml_dataset_id

data_dir=$(readlink -ve $1) || { echo "Error! Exiting..."; exit 1; }
output_dir=$(readlink -m $2)
clearml_dataset_id=$3
mode=${4:-"single"}

# change these values if needed
project_name="train_mt5"
run_id="run1"
CLEARML_QUEUE="default"
docker_image="zs12/multidoc_multilingual:v0.2"

[[ ! -z "$CLEARML_QUEUE" ]] && SPECIFY_QUEUE="--queue $CLEARML_QUEUE" || SPECIFY_QUEUE=""

script_dir=$(dirname $0)
python="/opt/conda/envs/MultiDocMultiLingual/bin/python"

clearml_output="clearml_artifact_output/output/"
clearml_model_path="clearml_dataset/${clearml_dataset_id}"

set -x

clearml_artifact_ID=$(python ${script_dir}/upload_artifacts.py $project_name "input_dir" --files $data_dir --names "data" | tail -n1)
input_dir="clearml_artifact_input/${clearml_artifact_ID}/data"

if [[ "$mode" == "single" ]]; then
  echo "train single langauge mt5 model..."
  export num_train_epochs=20
  export PER_DEVICE_TRAIN_BATCH_SIZE=8
  export PER_DEVICE_EVAL_BATCH_SIZE=16
  export GRADIENT_ACC=4
  export num_train=236
  clearml-task --project $project_name --name ${mode}_${run_id} \
    --docker ${docker_image} \
    --repo https://github.com/zaidsheikh/MultiDoc-MultiLingual \
    --branch docker \
    --docker_args "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=${python} $EXTRA_DOCKER_ARGS" \
    --packages pip $SPECIFY_QUEUE \
    --script clearml_scripts/mt5_pipeline.py \
    --args model_name_or_path=${clearml_model_path} \
        model_name_or_path="google/mt5-base" \
        local_rank=-1 \
        data_dir=$input_dir \
        output_dir=$clearml_output \
        learning_rate=5e-4 \
        gradient_accumulation_steps=$GRADIENT_ACC \
        num_train_epochs=$num_train_epochs \
        logging_steps=$((num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC)) \
        save_steps=$((num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC)) \
        eval_steps=$((num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC)) \
        adafactor=True \
        per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
        per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE  \
        overwrite_output_dir=True \
        evaluation_strategy="steps" \
        predict_with_generate=True \
        do_train="True" \
        logging_first_step=True \
        metric_for_best_model=rouge2 \
        greater_is_better=True \
        n_val=500 \
        warmup_steps=$((num_train_epochs*num_train/PER_DEVICE_TRAIN_BATCH_SIZE/GRADIENT_ACC/10)) \
        weight_decay=0.01 \
        label_smoothing_factor=0.1
fi

if [[ "$mode" == *"multi"* ]]; then
  echo "train multilingual mt5 model..."
  export PER_DEVICE_TRAIN_BATCH_SIZE=8
  export GRADIENT_ACC=16
  export MAX_STEP=20000
  clearml-task --project $project_name --name ${mode}_${run_id} \
    --docker ${docker_image} \
    --repo https://github.com/zaidsheikh/MultiDoc-MultiLingual \
    --branch docker \
    --docker_args "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=${python} $EXTRA_DOCKER_ARGS" \
    --packages pip $SPECIFY_QUEUE \
    --script clearml_scripts/mt5_pipeline.py \
    --args model_name_or_path=${clearml_model_path} \
        model_name_or_path="google/mt5-base" \
        local_rank=-1 \
        data_dir=$input_dir \
        output_dir=$clearml_output \
        predict_with_generate=True \
        learning_rate=5e-5 \
        upsampling_factor=0.5 \
        label_smoothing_factor=0.1 \
        weight_decay=0.01 \
        gradient_accumulation_steps=$GRADIENT_ACC \
        max_steps=$MAX_STEP \
        logging_steps=100 \
        save_steps=2000 \
        adafactor=True \
        per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
        overwrite_output_dir=True \
        evaluation_strategy="no" \
        do_train="True" \
        logging_first_step=True \
        warmup_steps=2000 
fi

#TODO: download $clearml_output and copy to $output_dir
