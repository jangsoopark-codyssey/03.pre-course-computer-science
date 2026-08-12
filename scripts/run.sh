#!/bin/bash

project_root="$(dirname ${PWD})"

python=python3 #"${project_root}/venv/bin/python"

data=data_v1.1.json
num_iterations=10000

run=${project_root}/src/main.py

$python $run --data=$data --num-iterations=$num_iterations

