#!/bin/bash

# Script Description:
# copy cellranger output for cellsnp processing

# /media/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Bash_Scripts/copy_cellranger_output_for_cellsnp_processing.sh  /media/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023 /media/jagraz/PMACS/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023 /media/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/list_of_samples_processed_using_cellranger.txt

# Author: Jose L. Agraz, PhD
# Date: 2023-12-07
# Version: 1.1

# Check if the new location and input file are provided as arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_location> <target_location> <input_file>"
    exit 1
fi

# Assign the new location and input file to variables
source_location="$1"
target_location="$2"
input_file="$3"

cellranger_dir="cellranger_output"
cellranger_output_dir="outs"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

# Initialize a counter for the number of subdirectories processed
subdir_count=0

# Initialize a variable to accumulate the total size
total_size=0

# Read each line from the input file
while IFS= read -r sample_name; do
    # Check if the subdirectory exists
    subdir="${source_location}/${sample_name}/${cellranger_dir}"
    if [ -d "$subdir" ]; then
        # Increment the subdirectories counter
        subdir_count=$((subdir_count + 1))

        # Calculate the size of the subdirectory
        size=$(du -s "$subdir" | cut -f1)
        total_size=$((total_size + size))

        # Replace 'media' with the new location in the path
        new_subdir_path="${target_location}/${sample_name}"
        
        # Create the new directory path
        mkdir --parents "${new_subdir_path}"
        
        # Copy the subdirectory to the new location
        cp --force  --recursive "$subdir" "$new_subdir_path"
    else
        echo "Warning: Subdirectory '$subdir' does not exist."
    fi
done < "$input_file"

# Convert the total size to human-readable format (e.g., MB, GB)
human_readable_size=$(numfmt --to=iec-i --suffix=B --padding=7 $total_size)

# Display the total number of subdirectories processed and their total size
echo "Total subdirectories processed: $subdir_count"
echo "Total size of copied subdirectories: $human_readable_size"
