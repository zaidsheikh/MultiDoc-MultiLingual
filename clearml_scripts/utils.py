#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
import logging
logging.getLogger("clearml").setLevel(logging.DEBUG)
from clearml import Task, Dataset

DATASET_PREFIX = "clearml_dataset"
ARTIFACT_PREFIX = "clearml_artifact_input"
OUTPUT_PREFIX = "clearml_artifact_output"
Path(OUTPUT_PREFIX).mkdir(parents=True, exist_ok=True)

def download_artifacts():
    """
    Iterates through the provided commandline arguments and replaces all filepaths
    starting with DATASET_PREFIX or ARTIFACT_PREFIX with the path of the corresponding
    cached/local copy of the dataset/artifact
    """
    current_task = Task.current_task()
    params = current_task.get_parameters()
    print("Original parameters:")
    print(json.dumps(params, indent=4))
    for k, v in params.items():
        if not v:
            continue
        path = Path(v)
        if path.parts[0] == DATASET_PREFIX:
            # example: clearml_dataset/9795d31965a940f/models/checkpoint1.bin
            dataset_id, rest_of_path = path.parts[1], path.parts[2:]
            print(f"Downloading dataset {dataset_id}")
            dataset_path = Dataset.get(dataset_id=dataset_id).get_local_copy()
            params[k] = str(Path(dataset_path).joinpath(*rest_of_path))
        elif path.parts[0] == ARTIFACT_PREFIX:
            # example: clearml_artifact_input/eee760b9f6234d/data/input.txt
            # Note: this will break if artifact name has '/' and other unusual characters
            task_id, artifact_name, rest_of_path = path.parts[1], path.parts[2], path.parts[3:]
            print(f"Downloading artifact {artifact_name} from task {task_id}")
            task = Task.get_task(task_id=task_id)
            artifact_path = task.artifacts[artifact_name].get_local_copy()
            params[k] = str(Path(artifact_path).joinpath(*rest_of_path))
    current_task.set_parameters(params)
    print("Patched parameters:")
    print(json.dumps(current_task.get_parameters(), indent=4))

def upload_artifacts():
    """
    Scans the commandline arguments for filepaths starting with OUTPUT_PREFIX
    and uploads them as ClearML artifacts. Artifact names is the filepath without
    the OUTPUT_PREFIX and all '/' characters replaced by '.slash.'
    """
    current_task = Task.current_task()
    params = current_task.get_parameters()
    print(json.dumps(params, indent=4))
    for k, v in params.items():
        if not v:
            continue
        path = Path(v)
        if path.exists() and path.parts[0] == OUTPUT_PREFIX:
            # example: clearml_artifact_output/results/output.txt
            artifact_name = str(Path().joinpath(*path.parts[1:])).replace('/', '.slash.')
            task = Task.current_task()
            task.upload_artifact(artifact_name, v)
            print(f"Uploading {str(v)} as artifact {task.id}/{artifact_name}")
