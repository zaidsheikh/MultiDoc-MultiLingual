#!/bin/bash

[ $# -eq 1 ] || { echo "Usage: $0 model_dir/"; exit 1; }

model_dir=$(readlink -ve $1) || { echo "Error! Exiting..."; exit 1; }

python="/opt/conda/envs/MultiDocMultiLingual/bin/python"
docker_image="zs12/multidoc_multilingual:v0.3.4"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-"0"}

set -x
docker run --rm -it --gpus all \
  -p 4123:4123 \
  -v $model_dir:/model/ \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $docker_image \
  $python /MultiDoc-MultiLingual/baselines/mt5/pipeline_serve.py \
    --model_name_or_path /model/ \
    --data_dir /dummy_data_dir/ \
    --do_predict True --predict_with_generate \
    --do_eval False --evaluation_strategy no \
    --output_dir /dummy_output_dir/
