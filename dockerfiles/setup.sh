#!/bin/bash

# sudo apt install rustc cargo
# baselines/mt5/pipeline.py
# training_args.local_rank = -1
# otherwise we will get error:
# "Default process group has not been initialized, please make sure to call init_process_group"

eval "$(conda shell.bash hook)"
conda create -n MultiDocMultiLingual python=3.10.4
conda activate MultiDocMultiLingual
git clone https://github.com/zaidsheikh/MultiDoc-MultiLingual
cd MultiDoc-MultiLingual/
git checkout docker
pip install -r pip_freeze.txt
cd multilingual_rouge_scoring/
pip3 install --upgrade ./
