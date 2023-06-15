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
script_dir=$(dirname $0)

# change these values as needed
project_name="train_mt5"
run_id="run1"
CLEARML_QUEUE="default"
docker_image="zs12/multidoc_multilingual:v0.2"

[[ ! -z "$CLEARML_QUEUE" ]] && SPECIFY_QUEUE="--queue $CLEARML_QUEUE" || SPECIFY_QUEUE=""

python="/opt/conda/envs/MultiDocMultiLingual/bin/python"

# upload datasets using clearml-data and artifacts using upload_artifacts.py
clearml_output="clearml_artifact_output/output/"

set -x
# TODO: uncomment this
#clearml_artifact_ID=$(python ${script_dir}/upload_artifacts.py $project_name "input_dir" --files $data_dir --names "data" | tail -n1)
#input_dir="clearml_artifact_input/${clearml_artifact_ID}/data"
input_dir=clearml_artifact_input/abccf00bd97c434dbdd3ab685852fe90/data

if [[ "$mode" == "single" ]]; then
  echo "train single langauge mt5 model..."
    #--repo https://github.com/zaidsheikh/MultiDoc-MultiLingual \
    #--branch docker \
  clearml-task --project $project_name --name ${mode}_${run_id} \
    --docker ${docker_image} \
    --docker_args "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=${python} $EXTRA_DOCKER_ARGS" \
    --packages pip $SPECIFY_QUEUE \
    --script clearml_scripts/mt5_pipeline.py \
    --args model_name_or_path=google/mt5-small \
        data_dir=$input_dir \
        output_dir=$clearml_output \
        overwrite_output_dir=True \
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
    --args model_name_or_path=google/mt5-small \
        data_dir=$input_dir \
        output_dir=$clearml_output \
        overwrite_output_dir=True \
        do_train="True" \
        upsampling_factor=1
fi

#TODO: download $clearml_output and copy to $output_dir
