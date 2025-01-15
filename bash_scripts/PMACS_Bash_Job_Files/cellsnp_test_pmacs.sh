#!/bin/bash
#BSUB -J cellsnp_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellsnp_job.%J.out
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellsnp_job.%J.error

for arg in "$@"; do
    echo "Argument: $arg"
done

