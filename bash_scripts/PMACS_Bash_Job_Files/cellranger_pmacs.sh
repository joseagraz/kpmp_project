#!/bin/bash
#BSUB -J cellranger_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.out
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.error
#BSUB -M 20480
#BSUB -n 4
#BSUB -R "rusage [mem=20480] span[hosts=4]" 

/home/jagraz/Software/cellranger-7.2.0/bin/cellranger count \
 --id=0a6a38c6-1669-4f67-ad91-0e1ab2ac982f \
 --sample=0a6a38c6-1669-4f67-ad91-0e1ab2ac982f \
 --include-introns=true \
 --transcriptome=/home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/refdata-gex-GRCh38-2020-A \
 --fastqs=/home/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/0a6a38c6-1669-4f67-ad91-0e1ab2ac982f \
 --localcores=4 \
 --localmem=20 \
 --output-dir=/home/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/0a6a38c6-1669-4f67-ad91-0e1ab2ac982f/cellranger_output
