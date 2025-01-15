#!/bin/bash

# Script Description:
# This script reads a list of sample names from a file, navigates to each corresponding
# subdirectory within a specified starting directory, checks for a 'cellranger_output'
# subdirectory, and deletes it if found. It also keeps a count of the number of directories deleted.

# Author: Jose L. Agraz, PhD
# Date: 2023-11-07
# Version: 1.1

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <starting_directory> <input_file>"
    exit 1
fi

# Assign arguments to variables
starting_directory="$1"
input_file="$2"

# Check if the starting directory exists
if [ ! -d "$starting_directory" ]; then
    echo "Error: Starting directory '$starting_directory' does not exist."
    exit 1
fi

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

# Initialize a counter for the number of directories deleted
deleted_count=0

# Change to the starting directory
cd "$starting_directory"

# Read each line from the input file
while IFS= read -r sample
do
    # Check if the sample directory exists
    if [ -d "$sample" ]; then
        # Change to the sample directory
        cd "$sample"

        # Check if the 'cellranger_output' directory exists
        if [ -d "cellranger_output" ]; then
            # Delete the 'cellranger_output' directory
            echo "Deleting 'cellranger_output' in sample directory '$sample'"
            rm -rf "cellranger_output"

            # Increment the deleted counter
            ((deleted_count++))
        else
            echo "'cellranger_output' does not exist in '$sample'"
        fi

        # Change back to the starting directory
        cd "$starting_directory"
    else
        echo "Warning: Sample directory '$sample' does not exist in '$starting_directory'."
    fi

done < "$input_file"

# Display the count of directories deleted
echo "Total 'cellranger_output' directories deleted: $deleted_count"

