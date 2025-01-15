#!/bin/bash
# ------------------------------------------------------
# Description: To rename files in the current directory 
# that start with "bamtofastq" and end with "fastq.gz" by 
# replacing "bamtofastq" with the current directory name
#
# retrieves the current directory name using the basename command.
# then iterates through files in the current directory that match 
# the pattern "bamtofastq*fastq.gz."
# For each matching file, it constructs a new name by replacing 
# "bamtofastq" with the current directory name using the ${var/old/new}
# pattern.
# Finally, it renames the file using the mv command and prints a 
# message indicating the renaming action.
#
# Author:
# Jose L. Agraz & Parker Wilson
#
# Date: Nov 2, 2023
# ------------------------------------------------------
# Get the current directory name
current_directory=$(basename "$PWD")

# Rename files
for file in bamtofastq*fastq.gz; do
    new_name="${file/bamtofastq/$current_directory}"
    mv "$file" "$new_name"
    echo "Renamed: $file to $new_name"
done

