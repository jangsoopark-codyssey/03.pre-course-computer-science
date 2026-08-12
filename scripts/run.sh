#!/bin/bash

project_root="$(dirname ${PWD})"

python=python3 #"${project_root}/venv/bin/python"

data=data_schema_error.json
run=${project_root}/src/main.py


$python $run --data=$data

