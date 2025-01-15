#!/bin/bash
#BSUB -J bamtofastq_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.out     # Name of the job output file 
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/bamtofastq_job.%J.error   # Name of the job error file

SCRIPT_PATH="/home/jagraz/Software/cellranger-7.2.0/lib/bin"
ROOT_DIR="/home/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023"
SAMPLE_NAME="e48ebcff-a2f6-40ad-8740-6ed1697be357"
FILE_NAME="f5547539-1f06-4afd-a6cb-329fb55dbb4c_KPMP_14317.bam"
#
"$SCRIPT_PATH/bamtofastq" \
   --nthreads=4 \
   "$ROOT_DIR/$SAMPLE_NAME/$FILE_NAME"  \
   "$ROOT_DIR/$SAMPLE_NAME/fastq_files"

