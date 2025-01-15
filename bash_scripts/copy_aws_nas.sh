#!/bin/bash

# Script Description:
# Copy files from AWS S3

# Author: Jose L. Agraz, PhD
# Date: 2023-11-07
# Version: 1.1

# Check if the input file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Assign the input file to a variable
input_file="$1"
root_dir="/media/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/requested_samples/"
# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

# Read each line from the input file
while IFS= read -r line
do
    # Extract the target directory name using 'cut' to split the URL
    # We are cutting by '/' and taking the 4th field which is after "s3://kpmp-knowledge-environment/"
    target_dir=$(echo "$line" | cut -d'/' -f4)

    # Check if the target directory already exists, if not, create it
    if [ ! -d "$target_dir" ]; then
        echo "Creating directory: $target_dir"
        mkdir $root_dir/$target_dir
    fi

    # Execute the AWS S3 copy command
    echo "Copying $line to $root_dir/$target_dir/"
    aws s3 cp $line "$root_dir/$target_dir/"

    # Check if the AWS command was successful
    if [ "$?" -ne 0 ]; then
        echo "Error: aws s3 cp command failed for '$line'"
        exit 1
    fi

done < "$input_file"

echo "All items have been copied successfully."

