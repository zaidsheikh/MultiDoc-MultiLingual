#!/bin/bash

set -x
local_model_path=$(python clearml_scripts/download_huggingface_models.py "google/mt5-small")
clearml_dataset_id=$(clearml-data create --project $project_name --name "google--mt5-small"|grep 'created id='|cut -f2 -d=)
clearml-data add --id $clearml_dataset_id --files $local_model_path
clearml-data upload --id $clearml_dataset_id
clearml-data close --id $clearml_dataset_id
print $clearml_dataset_id
