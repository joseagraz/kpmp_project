#!/bin/bash

# Script Description:
# continuously monitor and log the available RAM on a system into a CSV file

# Author: Jose L. Agraz, PhD
# Date: 2023-11-07
# Version: 1.1

# Source and destination directories
source_directory="/media/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023"
destination_directory="/mnt/12TB_Disk/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023"

# Create an array to hold the filtered sample directories
declare -a filtered_sample_directories
cellranger_dir="cellranger_output"

# Step 1: List all subdirectories in the source directory
for dir in "$source_directory"/*/; do
    # Check if the directory contains a 'cellranger_output' subdirectory
    if [ -d "${dir}cellranger_output" ]; then
        # Add the directory to the filtered list
        filtered_sample_directories+=("$dir")
    fi
done

# Step 2: Copy the filtered directories
for sample_dir in "${filtered_sample_directories[@]}"; do
    # Define the destination path for the current sample directory
    dest_path="$destination_directory/$(basename "$sample_dir")"

    # echo "Copying $sample_dir to $dest_path"
    echo $sample_dir
    cp -rn "$sample_dir$cellranger_dir" "$dest_path"

done

echo "Copying complete."