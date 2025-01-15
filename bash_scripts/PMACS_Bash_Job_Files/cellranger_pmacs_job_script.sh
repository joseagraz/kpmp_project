#!/bin/bash
#BSUB -J cellranger_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.out
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.error

#echo "Activate Conda Enviroment"
#conda activate cellranger_env_p310
echo "Execute Cellranger within Python 3.10"
conda run -n cellranger_env_p310 python  /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Python-Scripts/cell_ranger_execution_pmacs.py
 
 # to run job
 # bsub < <script_name> 
 # -M 20480 -n 4 -R "rusage [mem=20480] span[hosts=1]" 
