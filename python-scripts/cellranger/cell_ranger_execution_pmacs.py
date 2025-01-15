"""
CellRanger Single Cell Analysis Automation Script

This script automates the process of reading an Excel file, renaming fastq.gz files based on a regex pattern, 
and running the cellranger count command for single cell analysis.

Authors: Jose L. Agraz, PhD
Credits: Parker Wilson, MD PhD
Version: 1.0
"""

import pandas as pd
import re
import time
import shutil
from pathlib import Path
import subprocess
import tempfile
import datetime
import logging
import random
# ------------------------------------------
# Globals
# When running at HPC
ROOT_PATH              = Path("/home/jagraz/KPMP_Data/Privately_Available_Data")
DATA_ROOT_PATH         = ROOT_PATH / "Original_Download_KPMP_S3_Bucket_Oct_16_2023"
CELL_RANGER_PATH       = Path("/home/jagraz/Software/cellranger-7.2.0/bin/")
# ------------------------------------------
# When running at PC
# ROOT_PATH              = Path("/mnt/12TB_Disk/KPMP_Data/Privately_Available_Data")
# DATA_ROOT_PATH         = ROOT_PATH / "Original_Download_KPMP_S3_Bucket_Oct_16_2023/requested_samples"
# CELL_RANGER_PATH       = Path("/home/jagraz/Documents/yard/cellranger-7.2.0/bin/")
# ------------------------------------------
# When running at PC using HPC mounted data
# ROOT_PATH              = Path("/media/jagraz/PMACS/KPMP_Data/Privately_Available_Data")
# DATA_ROOT_PATH         = ROOT_PATH / "Original_Download_KPMP_S3_Bucket_Oct_16_2023"
# CELL_RANGER_PATH       = Path("/home/jagraz/Documents/yard/cellranger-7.2.0/bin/")
# ------------------------------------------
SUPPORTING_FILES       = ROOT_PATH / "Supporting_Files"
DATALOGS_DIR           = SUPPORTING_FILES / "Datalogs"
CELL_RANGER_OUTPUT_DIR = Path("cellranger_output")
DESTINATION_DRIVE      = DATA_ROOT_PATH
REGEX_PATTERN          = r"^.{36}.*?(?:_(S\d{1,3}))?(?:_all_)?(?:_(L\d{1,3}))?(?:_(I\d{1,3}))?(?:[._]?(R\d{1,3})(?:-V\d{1,3})?)?(_\d{3})?\.fastq\.gz$"
TRANSCRIPTOME_PATH     = SUPPORTING_FILES / "refdata-gex-GRCh38-2020-A"
EXCEL_FILE_PATH        = SUPPORTING_FILES / "Parker_Wilson_PennMed.xlsx"
MISSING_SAMPLES_FILE   = DATALOGS_DIR / "missing_sample.csv"
NUMBER_OF_CORES        = 4
LOCAL_MEMORY           = 20
FASTQ_FILE_EXTENSION   = '.fastq.gz'
# ------------------------------------------
# Set up logging
LOG_FORMAT    = "%(asctime)s — %(levelname)s — %(message)s"
current_time  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXECUTION_LOG = DATALOGS_DIR / f"cellranger_automation_date-time_{current_time}.log"
logging.basicConfig(filename=EXECUTION_LOG, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()
# ------------------------------------------
# Functions
def read_excel(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)
# ------------------------------------------
def get_unique_ids(df: pd.DataFrame) -> list:
    return df["Internal Package ID"].unique()
# ------------------------------------------
def find_source_dir_path(root_path: Path, source_dir: str) -> Path:
    return next(root_path.rglob(source_dir), None)
# ------------------------------------------
def remove_consecutive_non_alphanumeric(s):
    # Replace consecutive non-alphanumeric characters with a single instance of that character
    return re.sub(r'([^a-zA-Z0-9])\1+', r'\1', s)
# ------------------------------------------
def rename_files_in_temp_dir(working_dir: Path, regex_pattern: str, source_dir: str):
    global FASTQ_FILE_EXTENSION

    logger.info(f"Renaming files for Cellranger execution: {working_dir}")
    for file in working_dir.iterdir():
        if file.name.endswith(FASTQ_FILE_EXTENSION):
            logger.info(f"File of Interest: {file.name}")
            match = re.match(regex_pattern, file.name)
            if match:
                logger.info(f"Match.Groups: {match.groups()}")
                # Process each group: strip ".", "_", then prefix and append "_"
                processed_groups = ['_' + group.replace('.', '').replace('_', '') for group in match.groups() if group]
                logger.info(f"Process groups: {processed_groups}")
                #groups_name = ''.join(filter(None, match.groups()))
                groups_name = ''.join(processed_groups)
                # logger.info(f"Joined groups: {groups_name}")
                new_name = f"{source_dir}{groups_name}{FASTQ_FILE_EXTENSION}"
                if not file.name==new_name:
                    logger.info(f"Renaming\nOld name: {file.name}\nNew name: {new_name}")
                    # print(f"Renaming\nOld name: {file.name}\nNew name: {new_name}")                
                    file.rename(working_dir / new_name)            

# ------------------------------------------
def build_cellranger_flags(row: pd.Series) -> str:
    return f"Participant ID:{row['Participant ID']}\nProtocol:{row['Protocol']}\nSample Type:{row['Sample Type']}\nTissue Type:{row['Tissue Type']}\nExperimental Strategy:{row['Experimental Strategy']}"
# ------------------------------------------
def run_cellranger(source_dir: str, temp_dir: Path, cellranger_description: str, cellranger_output_dir: str) -> None:
    global TRANSCRIPTOME_PATH
    global NUMBER_OF_CORES
    global LOCAL_MEMORY
    global CELL_RANGER_PATH

    cell_ranger_program = str(CELL_RANGER_PATH / "cellranger")

    cmd = [
        cell_ranger_program, "count",
        "--id=" + source_dir,
        "--sample=" + source_dir,
        "--include-introns=true",
        "--transcriptome=" + str(TRANSCRIPTOME_PATH),
        "--fastqs=" + str(temp_dir),
        "--localcores=" + str(NUMBER_OF_CORES),
        "--localmem=" + str(LOCAL_MEMORY),
        "--description=" + cellranger_description,
        "--output-dir=" + cellranger_output_dir
    ]
    logger.info(f"-----------------------------------------------------")
    logger.info(f"Executing CellRanger on {source_dir} as:")
    logger.info(f"{cell_ranger_program} count \n --id={source_dir} \n --sample={source_dir} \n --include-introns=true \n --transcriptome={str(TRANSCRIPTOME_PATH)} \n --fastqs={str(temp_dir)} \n --localcores={str(NUMBER_OF_CORES)} \n --localmem={str(LOCAL_MEMORY)} \n --description= {cellranger_description} \n --output-dir={cellranger_output_dir}")
    logger.info(f"-----------------------------------------------------")
    subprocess.run(cmd)
    
    return (cellranger_output_dir/ Path("outs")).exists()
# ------------------------------------------
def verify_contents_identical(dir1: Path, dir2: Path) -> bool:
    return sorted(dir1.rglob('*')) == sorted(dir2.rglob('*'))
# ------------------------------------------
def cell_ranger_execution(cellranger_out_to_dir:str, source_dir_path:str, source_dir:str):
    global CELL_RANGER_OUTPUT_DIR
    global REGEX_PATTERN

    # logger.info(f"Executing CellRanger for: {source_dir}")
    # logger.info(f"\tSource dir: {source_dir_path}")
    # logger.info(f"\tOutput dir: {cellranger_out_to_dir}")

    rename_files_in_temp_dir(source_dir_path, REGEX_PATTERN, source_dir)

    row = df[df["Internal Package ID"] == source_dir].iloc[0]
    cellranger_description = build_cellranger_flags(row)

    cellranger_out_to_dir.mkdir(parents=True, exist_ok=True)

    results = run_cellranger(source_dir, 
                             source_dir_path, 
                             cellranger_description, 
                             str(cellranger_out_to_dir))
    
    if results:
        logger.info(f"Cellranger execution sucess")
    else:
        logger.info(f"Cellranger execution failed")

    return results
# ------------------------------------------
def short_pause_execution()-> None:
    # Generate a random number of seconds between 1 to 30 minutes
    sleep_time = random.randint(1, 3)  # 60 seconds * 3 = 100 seconds
    print(f"Pausing for {sleep_time} secs...")
    time.sleep(sleep_time)
    print("Resuming...")
    return
# ------------------------------------------
def long_pause_execution()-> None:
    # Generate a random number of seconds between 1 to 30 minutes
    sleep_time = random.randint(30, 180)  # 60 seconds * 3 = 180 seconds
    print(f"Pausing for {sleep_time/60:2f} mins...")
    time.sleep(sleep_time)
    print("Resuming...")
    return
# ------------------------------------------
def count_fastq_files(directory: Path) -> int:
    """
    Count the number of files in the specified directory with names ending in ".fastq*".

    Parameters:
    - directory (Path): The directory in which to search for files.

    Returns:
    - int: The number of files with names ending in ".fastq*".
    """
    
    # Use a generator expression to count the files
    number_of_files = sum(1 for file in directory.glob("*.fastq*") if file.is_file())
    logger.info(f"Number of fastq files found {number_of_files}:\n{directory}")

    return number_of_files
# ------------------------------------------
def check_no_fastq_gz(dir_path: Path) -> bool:
    global FASTQ_FILE_EXTENSION
    for file in dir_path.iterdir():
        if file.is_file() and file.name.endswith(FASTQ_FILE_EXTENSION):
            # logger.info(f"File found {file.suffix}, continuing")
            return False
    return True
# ------------------------------------------
# Main
if __name__ == "__main__":
        
    start_time = datetime.datetime.now().replace(microsecond=0)
    logger.info(f'Script start: {start_time}')

    df         = read_excel(EXCEL_FILE_PATH)
    unique_ids = get_unique_ids(df)
    random.shuffle(unique_ids)

    for idx, sample_name in enumerate(unique_ids, 1):
        logger.info(f"Sample {sample_name}: {idx} of {len(unique_ids)}")

        # if not sample_name=="0a6a38c6-1669-4f67-ad91-0e1ab2ac982f":
        #     continue

        # logger.info(f'Looking for sample {sample_name} in\n{DATA_ROOT_PATH}')
        source_dir_path = find_source_dir_path(DATA_ROOT_PATH, sample_name)        

        # quit if sample not found
        if not source_dir_path:
            # logger.info(f"Missing Sample {sample_name}. \nCheck {MISSING_SAMPLES_FILE} for more info")
            # with open(MISSING_SAMPLES_FILE, "a") as f:
            #     f.write(sample_name + "\n")
            continue

        if check_no_fastq_gz(source_dir_path):
            logger.info(f"Sample {sample_name} lacks fastq files\n{source_dir_path}")
            continue

        cellranger_data_final_location = source_dir_path / CELL_RANGER_OUTPUT_DIR 
        # Skip if already cellranger results
        if not cellranger_data_final_location.exists():
            number_of_fastq_files = count_fastq_files(source_dir_path)
            if number_of_fastq_files:            
                logger.info(f"Sample {sample_name}: {idx} of {len(unique_ids)}")
                cell_ranger_execution(cellranger_data_final_location, source_dir_path, sample_name)
                break
            else:
                logger.info(f"No fastq files found in:\n{source_dir_path}")

    logger.info(f"Complete {sample_name} execution")
    end_time = datetime.datetime.now().replace(microsecond=0)
    logger.info(f'Script end: {end_time}')
    logger.info(f'Elapsed time: {end_time - start_time}')