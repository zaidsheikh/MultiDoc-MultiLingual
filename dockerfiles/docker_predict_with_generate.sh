#!/bin/bash

[ $# -eq 3 ] || { echo "Usage: $0 model_dir/ input_file.txt output_dir/"; exit 1; }

model_dir=$(readlink -ve $1) || { echo "Error! Exiting..."; exit 1; }
input_file=$(readlink -ve $2) || { echo "Error! Exiting..."; exit 1; }
output_dir=$(readlink -m $3)

mkdir -p $output_dir
chmod -R 777 $output_dir

python="/opt/conda/envs/MultiDocMultiLingual/bin/python"
docker_image="zs12/multidoc_multilingual:v0.2"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-"0"}

set -x
data_dir=$(mktemp -d)
cp -vi $input_file ${data_dir}/test.source
chmod -R 777 $data_dir 
docker run --rm -it --gpus all \
  -v $data_dir:/data/ -v $output_dir:/output/ -v $model_dir:/model/ \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $docker_image \
  $python /MultiDoc-MultiLingual/baselines/mt5/pipeline.py \
    --model_name_or_path /model/ \
    --data_dir /data/ \
    --do_predict True --predict_with_generate \
    --do_eval False --evaluation_strategy no \
    --output_dir /output/
rm -v ${data_dir}/test.source
