#!/bin/bash
#BSUB -J bamtofastq_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.out     # Name of the job output file 
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.error   # Name of the job error file

SCRIPT_PATH="/home/jagraz/Software/cellranger-7.2.0/lib/bin"
ROOT_DIR="/home/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023"
SAMPLE_NAME="d029d3d1-9bb9-4d94-839e-f835fe5171ce"
FILE_NAME="b4aaed14-edf5-4200-93d8-e20b7c4644f9_PREMIERE-SamplePRE027-1.bam"
#
"$SCRIPT_PATH/bamtofastq" \
   --nthreads=4 \
   "$ROOT_DIR/$SAMPLE_NAME/$FILE_NAME"  \
   "$ROOT_DIR/$SAMPLE_NAME/fastq_files"

