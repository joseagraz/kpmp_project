"""
CellRanger Single Cell Analysis Automation Script

This script automates the process of reading an Excel file, renaming fastq.gz files based on a regex pattern, 
and running the cellranger count command for single cell analysis.

Authors: [Your Name]
Version: 1.0
"""

import pandas as pd
import re
import shutil
from pathlib import Path
import subprocess
import tempfile
import datetime
import concurrent.futures
import logging
import random
# ------------------------------------------
# Globals
ROOT_PATH             = Path("/home/jagraz/Documents/10x_Genomics")
DATA_ROOT_PATH        = Path("/mnt/12TB_Disk/KPMP_Data/Single-Cell-Files")
CELLRANGER_OUTPUT_DIR = "cellranger_output"
DESTINATION_DRIVE     = Path("/media/jagraz/3T_Disk")
REGEX_PATTERN         = r"^[a-zA-Z0-9\-]{36}_.*?(_S\d{1,3})?(_L\d{1,3})?(_I\d{1,3})?([._]?R\d{1,3}(-[vV]\d{1,3})?[._]?)?(_\d{3}\.)?fastq\.gz$"
TRANSCRIPTOME_PATH    = ROOT_PATH / "refdata-gex-GRCh38-2020-A"
EXCEL_FILE_PATH       = ROOT_PATH / "Parker_Wilson_PennMed.xlsx"
MISSING_SAMPLES_FILE  = ROOT_PATH / "missing_sample.csv"
NUMBER_OF_CORES       = 4
LOCAL_MEMORY          = 128
MAX_WORKERS           = 3  # Number of parallel cellranger instances
# ------------------------------------------
# Set up logging
LOG_FORMAT = "%(asctime)s — %(levelname)s — %(message)s"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"cellranger_automation_parallel_date-time_{current_time}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()
# ------------------------------------------
# Functions
def read_excel(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def get_unique_ids(df: pd.DataFrame) -> list:
    return df["Internal Package ID"].unique()

def find_source_dir_path(root_path: Path, source_dir: str) -> Path:
    return next(root_path.rglob(source_dir), None)

def rename_files_in_temp_dir(temp_dir: Path, regex_pattern: str, source_dir: str):
    for file in temp_dir.iterdir():
        match = re.match(regex_pattern, file.name)
        if match:
            groups_name = ''.join(filter(None, match.groups()))
            new_name = f"{source_dir}{groups_name}fastq.gz"
            logger.info(f"Rename {file} to {new_name}")
            file.rename(temp_dir / new_name)

def build_cellranger_flags(row: pd.Series) -> str:
    return f"Participant ID:{row['Participant ID']}\nProtocol:{row['Protocol']}\nSample Type:{row['Sample Type']}\nTissue Type:{row['Tissue Type']}\nExperimental Strategy:{row['Experimental Strategy']}"

def run_cellranger(source_dir: str, temp_dir: Path, cellranger_description: str, cellranger_output_dir: str) -> None:
    global TRANSCRIPTOME_PATH
    global NUMBER_OF_CORES
    global LOCAL_MEMORY
    cmd = [
        "cellranger", "count",
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
    logger.info(f"Executing CellRanger on {source_dir} as:")
    logger.info(f"cellranger count \n --id={source_dir} \n --sample={source_dir} \n --include-introns=true \n --transcriptome={str(TRANSCRIPTOME_PATH)} \n --fastqs={str(temp_dir)} \n --localcores={str(NUMBER_OF_CORES)} \n --localmem={str(LOCAL_MEMORY)} \n --description= {cellranger_description} \n --output-dir={cellranger_output_dir}")
    subprocess.run(cmd)
    return 

def verify_contents_identical(dir1: Path, dir2: Path) -> bool:
    return sorted(dir1.rglob('*')) == sorted(dir2.rglob('*'))

def cell_ranger_execution(cellranger_out_to_dir:str, source_dir_path:str, source_dir:str):

    with tempfile.TemporaryDirectory() as temp_dir_path:
        temp_dir = Path(temp_dir_path)
        logger.info(f"Created temp file: {temp_dir}")

        logger.info(f"Copying fastq files from\n{source_dir_path} to {temp_dir}")
        for fastq_file in source_dir_path.rglob("*.fastq.gz"):
            shutil.copy(fastq_file, temp_dir)

        rename_files_in_temp_dir(temp_dir, REGEX_PATTERN, source_dir)

        row = df[df["Internal Package ID"] == source_dir].iloc[0]
        cellranger_description = build_cellranger_flags(row)

        cellranger_output_dir = temp_dir / CELLRANGER_OUTPUT_DIR
        cellranger_output_dir.mkdir(parents=True, exist_ok=True)

        run_cellranger(source_dir, 
                    temp_dir, 
                    cellranger_description, 
                    str(cellranger_output_dir))
        
        cellranger_out_from_dir = str(cellranger_output_dir / "outs")
        logger.info(f"Copying cellranger output from {cellranger_out_from_dir} to {cellranger_out_to_dir}")
        shutil.copytree(cellranger_out_from_dir, cellranger_out_to_dir, dirs_exist_ok=True)

        if not verify_contents_identical(cellranger_out_from_dir, cellranger_out_to_dir):
            logger.info(f"Contents of {cellranger_out_from_dir} and {cellranger_out_to_dir} are not identical!")
        else:
            logger.info(f"Cellranger Contents tranfer validated")

    return None
# ------------------------------------------
def process_sample(source_dir: str) -> None:
    source_dir_path = find_source_dir_path(DATA_ROOT_PATH, source_dir)
    if not source_dir_path:
        logger.info(f"Path not found: {source_dir_path}")
        logger.info(f"Missing Sample {source_dir}. \nCheck {MISSING_SAMPLES_FILE} for more info")
        with open(MISSING_SAMPLES_FILE, "a") as f:
            f.write(source_dir + "\n")
        return

    cellranger_data_final_location = DESTINATION_DRIVE / source_dir /CELLRANGER_OUTPUT_DIR

    # Check if the directory exists and has contents
    if cellranger_data_final_location.exists() and any(cellranger_data_final_location.iterdir()):                
        logger.info(f"CellRanger output already exists: {cellranger_data_final_location}\nSkipping execution!")
    else:
        logger.info(f"Executing CellRanger for {source_dir}")
        cell_ranger_execution(cellranger_data_final_location, source_dir_path, source_dir)

    logger.info(f"Complete {source_dir} execution")

# Main
if __name__ == "__main__":
        
    start_time = datetime.datetime.now().replace(microsecond=0)
    logger.info(f'Script start: {start_time}')

    df         = read_excel(EXCEL_FILE_PATH)
    unique_ids = get_unique_ids(df)
    random.shuffle(unique_ids)

    # Use a ThreadPool to run cellranger in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_sample, source_dir) for source_dir in unique_ids]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            logger.info(f"Sample: {idx} of {len(unique_ids)}")

    end_time = datetime.datetime.now().replace(microsecond=0)
    logger.info(f'Script end: {end_time}')
    logger.info(f'Elapsed time: {end_time - start_time}')