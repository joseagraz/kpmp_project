#!/bin/bash
#BSUB -J cellranger_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.out
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.error

 # to run job at the command prompt. 20GB of RAM using 4 cores
 # bsub  -M 20480 -n 4 -R "rusage [mem=20480] span[hosts=1]" < scanranger_pmacs.sh
 
SAMPLE_NAME="cb1afd1e-d744-4593-8927-4b2bb592ed54"
ROOT_DIR="/home/jagraz/KPMP_Data/Privately_Available_Data"
DATA_DIR="$ROOT_DIR/Original_Download_KPMP_S3_Bucket_Oct_16_2023"
CELLRANGER_OUTPUT_DIR="cellranger_output"
CELLRANGER_PATH="/home/jagraz/Software/cellranger-7.2.0/bin"

"$CELLRANGER_PATH/cellranger" count \
 --id=$SAMPLE_NAME \
 --sample=$SAMPLE_NAME \
 --include-introns=true \
 --transcriptome="$ROOT_DIR/Supporting_Files/refdata-gex-GRCh38-2020-A" \
 --fastqs="$DATA_DIR/$SAMPLE_NAME" \
 --localcores=4 \
 --localmem=20 \
 --output-dir="$DATA_DIR/$SAMPLE_NAME/$CELLRANGER_OUTPUT_DIR"
