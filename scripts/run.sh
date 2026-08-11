#!/bin/bash

project_root="$(dirname ${PWD})"

python=python3 #"${project_root}/venv/bin/python"

run=${project_root}/src/main.py


$python $run

