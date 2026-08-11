#!/bin/bash

project_root="$(dirname ${PWD})"

python=python3 #"${project_root}/venv/bin/python"

data=data_v1.1.json
run=${project_root}/src/main.py


$python $run --data=$data

