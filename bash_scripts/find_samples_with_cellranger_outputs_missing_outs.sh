#!/bin/bash

# Check if the directory is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Assign the directory to a variable
directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

# List and filter sample_directories
for sample_dir in "$directory"/*/; do
    # Check for 'cellranger_output' directory
    if [ -d "${sample_dir}cellranger_output" ]; then
        # Check if 'outs' subdirectory does not exist
        if [ ! -d "${sample_dir}cellranger_output/outs" ]; then
            echo "Sample $(basename "$sample_dir") has 'cellranger_output' but lacks 'outs' subdirectory."
        fi
    fi
done

