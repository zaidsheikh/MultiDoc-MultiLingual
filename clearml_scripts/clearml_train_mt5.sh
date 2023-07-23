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
docker_image="zs12/multidoc_multilingual:v0.3.1"

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
  clearml-task --project $project_name --name ${mode}_${run_id} \
    --docker ${docker_image} \
    --repo https://github.com/zaidsheikh/MultiDoc-MultiLingual \
    --branch docker \
    --docker_args "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=${python} $EXTRA_DOCKER_ARGS" \
    --packages pip $SPECIFY_QUEUE \
    --script clearml_scripts/mt5_pipeline.py \
    --args model_name_or_path=${clearml_model_path} \
        data_dir=$input_dir \
        output_dir=$clearml_output \
        overwrite_output_dir=True \
        local_rank=-1 \
        do_train="True"
fi

if [[ "$mode" == *"multi"* ]]; then
  echo "train multilingual mt5 model..."
  clearml-task --project $project_name --name ${mode}_${run_id} \
    --docker ${docker_image} \
    --repo https://github.com/zaidsheikh/MultiDoc-MultiLingual \
    --branch docker \
    --docker_args "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=${python} $EXTRA_DOCKER_ARGS" \
    --packages pip $SPECIFY_QUEUE \
    --script clearml_scripts/mt5_pipeline.py \
    --args model_name_or_path=${clearml_model_path} \
        data_dir=$input_dir \
        output_dir=$clearml_output \
        overwrite_output_dir=True \
        do_train="True" \
        local_rank=-1 \
        upsampling_factor=1
fi

#TODO: download $clearml_output and copy to $output_dir
