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
docker_image="zs12/multidoc_multilingual:v0.3.3"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-"0"}

if [[ "$mode" == "single" ]]; then
  echo "train single langauge mt5 model..."
  set -x
  docker run --rm -it --gpus all -v $data_dir:/data/ -v $output_dir:/output/ \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $docker_image \
    $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
      --model_name_or_path google/mt5-small \
      --data_dir /data/ \
      --output_dir /output/ \
      --local_rank -1 \
      --overwrite_output_dir --do_train True
fi

if [[ "$mode" == *"multi"* ]]; then
  echo "train multilingual mt5 model..."
  set -x
  docker run --rm -it --gpus all -v $data_dir:/data/ -v $output_dir:/output/ \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $docker_image \
    $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
      --model_name_or_path google/mt5-small \
      --data_dir /data/ \
      --output_dir /output/ \
      --overwrite_output_dir --do_train True \
      --local_rank -1 \
      --upsampling_factor 1
fi
