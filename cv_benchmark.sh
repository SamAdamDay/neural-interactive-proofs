#!/bin/bash

# Define the Python script to be called
PYTHON_SCRIPT="playground/benchmarking/cv_benchmark.py"

# Define the sets of arguments to be passed to the Python script
ARGS_SET=(
    "--num_data 100 --save_path /path/to/save1"
    "--num_data 200 --save_path /path/to/save2"
    "--num_data 300 --save_path /path/to/save3"
)
MODELS=(
    "openai/gpt-4o-mini"
    "meta-llama/llama-3.1-8b-instruct"
    "deepseek/deepseek-coder"
    "mistralai/codestral-mamba"
    "openai/gpt-4o-mini"
)

# Number of times to loop
NUM_LOOPS=100

# Loop over each set of arguments

    # Loop and call the Python script in the background
for ((i=1; i<=NUM_LOOPS; i++)); do
    for ARGS in "${ARGS_SET[@]}"; do
        echo "Running iteration $i with args '$ARGS': python $PYTHON_SCRIPT $ARGS"
        python $PYTHON_SCRIPT $ARGS &
    done
done

# Wait for all background processes to complete
wait

echo "All iterations have completed."