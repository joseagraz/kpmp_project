#!/bin/bash

# Script Description:
# continuously monitor and log the available RAM on a system into a CSV file

# Author: Jose L. Agraz, PhD
# Date: 2023-11-07
# Version: 1.1

# Define CSV file name
output_file="ram_usage_Seurat_v5_full_datasets.csv"

# Write the header to the CSV file
echo "Sample Time, Available RAM (Gi)" > $output_file

# Infinite loop to collect data
while true; do
  # Get the current time in a specific format
  sample_time=$(date +"%Y-%m-%d %H:%M:%S")

  # Get available RAM in GiB
  available_ram=$(free -h | grep Mem | awk '{gsub("G",""); print $7}')

  # Replace G (if it exists) with a comma for Gi
  available_ram_with_comma=$(echo $available_ram | sed 's/G/,/g')

  # Write the data to the CSV file
  echo "$sample_time, $available_ram_with_comma" >> $output_file

  # Wait for 10 seconds before the next iteration
  sleep 10
done
