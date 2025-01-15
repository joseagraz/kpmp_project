#!/bin/bash

# Description: This Bash script analyzes subdirectories 
# within a specified directory, identifying those containing
# a "cellranger_output" subdirectory. It calculates and 
# displays statistical data about their sizes, including 
# minimum, maximum, mean, and standard deviation. Additionally,
# it identifies samples closest in size to the mean and one 
# standard deviation above the mean.

# Author: Jose L. Agraz, PhD
# Date: 2023-11-28
# Version: 1.1

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

# Create an associative array to hold the sizes of 'cellranger_output' directories
declare -A cellranger_output_sizes

# List and filter sample_directories
for sample_dir in "$directory"/*/; do
    if [ -d "${sample_dir}cellranger_output" ]; then
        # Get the size of the 'cellranger_output' directory in GB
        size=$(du -sBG "${sample_dir}cellranger_output" | cut -f1)
        # Remove the 'G' from the size
        size=${size%G}
        # Add to the associative array
        cellranger_output_sizes[$(basename "$sample_dir")]=$size
        echo "Calculated size for ${sample_dir}cellranger_output: ${size}GB"
    fi
done

# Check if the array is not empty
if [ ${#cellranger_output_sizes[@]} -eq 0 ]; then
    echo "No 'cellranger_output' directories found in '$directory'."
    exit 0
fi

# Calculate min, mean, max, and standard deviation
min_size=0
max_size=0
total_size=0
sum_sq=0
count=0

for size in "${cellranger_output_sizes[@]}"; do
    if [ $count -eq 0 ] || [ $size -lt $min_size ]; then
        min_size=$size
    fi
    if [ $size -gt $max_size ]; then
        max_size=$size
    fi
    total_size=$((total_size + size))
    sum_sq=$((sum_sq + (size * size)))
    count=$((count + 1))
done

mean_size=$((total_size / count))
std_dev=$(echo "sqrt(($sum_sq - ($total_size * $total_size / $count)) / ($count - 1))" | bc -l)

# Find the sample names corresponding to min, max, and closest to mean and std_dev
closest_mean_sample=""
closest_std_dev_sample=""
min_diff_mean=999999
min_diff_std_dev=999999
mean_plus_std_dev=$(echo "$mean_size + $std_dev" | bc)

for sample in "${!cellranger_output_sizes[@]}"; do
    size=${cellranger_output_sizes[$sample]}
    diff_mean=$(echo "$size - $mean_size" | bc)
    diff_mean=${diff_mean#-} # Absolute value
    diff_std_dev=$(echo "$size - $mean_plus_std_dev" | bc)
    diff_std_dev=${diff_std_dev#-} # Absolute value

    if [ "$(echo "$diff_mean < $min_diff_mean" | bc)" -eq 1 ]; then
        min_diff_mean=$diff_mean
        closest_mean_sample=$sample
    fi

    if [ "$(echo "$diff_std_dev < $min_diff_std_dev" | bc)" -eq 1 ]; then
        min_diff_std_dev=$diff_std_dev
        closest_std_dev_sample=$sample
    fi
done

# Display the statistics
echo "Stat Sample_Name Size(GB)"
echo "min $min_sample ${min_size}GB"
echo "mean - ${mean_size}GB"
echo "max $max_sample ${max_size}GB"
echo "std_dev - $(printf "%.2f" $std_dev)GB"
echo "closest_to_mean $closest_mean_sample"
echo "closest_to_std_dev $closest_std_dev_sample"

