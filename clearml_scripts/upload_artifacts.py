#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import argparse
from clearml import Task
from pathlib import Path




def main(project_name, task_name, files, names):
    if names:
        assert len(names) == len(files)
        file_names = names
    else:
        file_names = [Path(f).parts[-1] for f in files]
        if len(file_names) != len(set(file_names)):
            sys.exit("Some of the provided files have the same names, please use --names to provide unique names for all files")

    task1 = Task.init(project_name=project_name, task_name=task_name)

    for file_path, file_name in zip(files, file_names):
        print(f"Uploading {file_path} as artifact {task1.id}/{file_name}")
        task1.upload_artifact(name=file_name, artifact_object=file_path)
    return task1.id


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name", type=str, help="project name")
    parser.add_argument("task_name", type=str, help="task name")
    parser.add_argument("--files", nargs='+', type=str, help="list of files/directories to upload", required=True)
    parser.add_argument("--names", nargs='+', type=str, help="names corresponding to the provided list of files (uses filename if not provided)", required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(main(args.project_name, args.task_name, args.files, args.names))
