#!/bin/bash -x


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-"0"}
export CUDA_VISIBLE_DEVICES
python baselines/mt5/pipeline_serve.py \
  --model_name_or_path ../MultiDoc-MultiLingual_models/models/Multilingual-mt5/ \
  --do_predict True --predict_with_generate --do_eval False --evaluation_strategy no \
  --output_dir dummy_output_dir/ --data_dir dummy_data_dir/

