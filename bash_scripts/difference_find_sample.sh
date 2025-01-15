#!/bin/bash

# Description: shows the differences between a list of 
# subdirectories in a root directory and a list of 
# subdirectories specified in a CSV file

# Author: Jose L. Agraz, PhD
# Date: 2023-11-28
# Version: 1.1



# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root_directory> <csv_file>"
    exit 1
fi

# Assign arguments to variables
root_directory="$1"
csv_file="$2"

# Check if the root directory exists
if [ ! -d "$root_directory" ]; then
    echo "Error: Directory '$root_directory' does not exist."
    exit 1
fi

# Check if the CSV file exists
if [ ! -f "$csv_file" ]; then
    echo "Error: File '$csv_file' does not exist."
    exit 1
fi

# Read subdirectories from the root directory and remove the path
subdirs_in_root=()
for subdir in "$root_directory"/*/; do
    subdirs_in_root+=("$(basename "$subdir")")
done

# Read subdirectories from the CSV file
subdirs_in_csv=()
while IFS= read -r line; do
    subdirs_in_csv+=("$line")
done < "$csv_file"

# Output the counts
echo "Number of subdirectories in '$root_directory': ${#subdirs_in_root[@]}"
echo "Number of subdirectories listed in '$csv_file': ${#subdirs_in_csv[@]}"

# Compare the two lists and output the differences
echo "Subdirectories in '$root_directory' not in '$csv_file':"
for subdir in "${subdirs_in_root[@]}"; do
    if [[ ! " ${subdirs_in_csv[*]} " =~ " $subdir " ]]; then
        echo "$subdir"
    fi
done

