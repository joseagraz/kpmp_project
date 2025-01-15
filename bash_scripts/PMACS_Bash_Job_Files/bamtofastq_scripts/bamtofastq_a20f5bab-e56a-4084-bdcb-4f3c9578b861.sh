#!/bin/bash
#BSUB -J bamtofastq_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.out     # Name of the job output file 
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.error   # Name of the job error file

SCRIPT_PATH="/home/jagraz/Software/cellranger-7.2.0/lib/bin"
ROOT_DIR="/home/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023"
SAMPLE_NAME="a20f5bab-e56a-4084-bdcb-4f3c9578b861"
FILE_NAME="661496f1-255e-4923-b648-bbb61da55e9e_S-1908-000898_KL-0014795.bam"
#
"$SCRIPT_PATH/bamtofastq" \
   --nthreads=4 \
   "$ROOT_DIR/$SAMPLE_NAME/$FILE_NAME"  \
   "$ROOT_DIR/$SAMPLE_NAME/fastq_files"

