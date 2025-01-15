#!/bin/bash
#BSUB -J bamtofastq_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.out     # Name of the job output file 
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.error   # Name of the job error file

SCRIPT_PATH="/home/jagraz/Software/cellranger-7.2.0/lib/bin"
#
echo "Root Path: $ROOT_DIR"
#echo "Sample: $SAMPLE_NAME"
#echo "BAM File: $FILE_NAME"
#
# "$SCRIPT_PATH/bamtofastq" \
#   --nthreads=4 \
#   "$ROOT_DIR/$SAMPLE_NAME/$FILE_NAME"  \
#   "$ROOT_DIR/$SAMPLE_NAME/"

