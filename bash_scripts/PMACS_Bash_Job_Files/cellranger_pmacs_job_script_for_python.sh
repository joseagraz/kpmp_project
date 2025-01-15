#!/bin/bash
#BSUB -J cellranger_job            # LSF job name
#BSUB -o /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.out
#BSUB -e /home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Datalogs/cellranger_job.%J.error

 #Note: activating the Conda Enviroment as done in linux on a PC does not work anymore, see new syntax below
 # to run job at the command prompt. 122GB of RAM using 8 cores
 # bsub  -M 122880 -n 8 -R "rusage [mem=122880] span[hosts=1]" < scanranger_pmacs.sh
 
PYTHON_FILE_DIR="/home/jagraz/KPMP_Data/Privately_Available_Data/Supporting_Files/Python-Scripts"
PYTHON_FILE_NAME="cell_ranger_execution_pmacs.py"
CONDA_ENVIROMENT_NAME="cellranger_env_p310"

echo "Execute Cellranger using Python 3.10"
conda run -n $CONDA_ENVIROMENT_NAME \
   python "$PYTHON_FILE_DIR/$PYTHON_FILE_NAME"
