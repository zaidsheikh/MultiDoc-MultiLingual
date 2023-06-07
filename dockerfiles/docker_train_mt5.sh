#!/bin/bash

[ $# -ge 2 ] || { echo "Usage: $0 input_data_dir/ output_dir/ [multi]"; exit 1; }

#data_dir=$(readlink -ve ${1:-"test_run1/output/individual/EN_part/"})
data_dir=$(readlink -ve ${1:-"test_run1/output/multilingual/"})
output_dir=$(readlink -m ${2:-"test_run1/docker_mt5_train_$$"})
multilingual=${3:-"single"}

mkdir -p $output_dir
chmod -R 777 $output_dir

python="/opt/conda/envs/MultiDocMultiLingual/bin/python"
docker_image="zs12/multidoc_multilingual:conda_base5"

set -x
if [[ "$mode" == "single" ]]; then
  echo "train single langauge mt5 model..."
  docker run --rm -it --gpus all -v $data_dir:/data/ -v $output_dir:/output/ --env CUDA_VISIBLE_DEVICES='' $docker_image \
    $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
      --model_name_or_path google/mt5-small \
      --data_dir /data/ \
      --output_dir /output/ \
      --overwrite_output_dir --do_train True
fi

if [[ "$mode" == *"multi"* ]]; then
  echo "train multilingual mt5 model..."
  docker run --rm -it --gpus all -v $data_dir:/data/ -v $output_dir:/output/ --env CUDA_VISIBLE_DEVICES='' $docker_image \
    $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
      --model_name_or_path google/mt5-small \
      --data_dir /data/ \
      --output_dir /output/ \
      --overwrite_output_dir --do_train True \
      --upsampling_factor 1
fi
