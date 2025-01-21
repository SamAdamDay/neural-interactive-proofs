#!/bin/bash

# Define the Python script to be called
PYTHON_SCRIPT="pvg/code_validation/dataset_generation.py"

# Define the arguments to be passed to the Python script
ARGS="--split test"

# Number of times to loop
NUM_LOOPS=100

# Loop and call the Python script
for ((i=1; i<=NUM_LOOPS; i++)); do
    echo "Running iteration $i: python $PYTHON_SCRIPT $ARGS"
    python $PYTHON_SCRIPT $ARGS
    # echo "Running iteration $i: python $PYTHON_SCRIPT"
    # python $PYTHON_SCRIPT
done