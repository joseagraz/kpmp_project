#!/bin/bash

# Script Description:
# Spool multiple PMACS jobs 

# Author: Jose L. Agraz, PhD
# Date: 2023-11-07
# Version: 1.1

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <script_name> <number_of_iterations>"
    exit 1
fi

# Assign input arguments to variables
script_name="$1"
num_iterations="$2"

number_of_cpus=4
ram_memory=20480

# Loop through the specified number of iterations
for ((i = 1; i <= num_iterations; i++)); do
    echo "Iteration $i: Running $script_name"
    
    bsub  -M $ram_memory -n $number_of_cpus -R "rusage [mem=$ram_memory] span[hosts=1]" < $script_name
    echo "bsub  -M $ram_memory -n $number_of_cpus -R "rusage [mem=$ram_memory] span[hosts=1]" < $script_name"
    echo "Iteration $i: Completed $script_name"
    # Wait for 2 seconds before processing the next file
    echo "Waiting for 2 seconds..."
    sleep 2    
done

echo "All iterations completed."

