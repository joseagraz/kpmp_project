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
            file.rename(temp_dir / new_name)

def build_cellranger_flags(row: pd.Series) -> str:
    return f"Participant ID:{row['Participant ID']}\nProtocol:{row['Protocol']}\nSample Type:{row['Sample Type']}\nTissue Type:{row['Tissue Type']}\nExperimental Strategy:{row['Experimental Strategy']}"

def run_cellranger(source_dir: str, temp_dir: Path, cellranger_description: str, cellranger_output_dir: str):
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
    print (f"Executing CellRanger on {source_dir} as:")
    print (f"cellranger count \n --id={source_dir} \n --sample={source_dir} \n --include-introns=true \n --transcriptome={str(TRANSCRIPTOME_PATH)} \n --fastqs={str(temp_dir)} \n --localcores={str(NUMBER_OF_CORES)} \n --localmem={str(LOCAL_MEMORY)} \n --description= {cellranger_description} \n --output-dir={cellranger_output_dir}")
    subprocess.run(cmd)

def verify_contents_identical(dir1: Path, dir2: Path) -> bool:
    return sorted(dir1.rglob('*')) == sorted(dir2.rglob('*'))

# Main
if __name__ == "__main__":

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
    LOCAL_MEMORY          = 256
        
    start_time = datetime.datetime.now().replace(microsecond=0)
    print(f'Script start: {start_time}')

    df         = read_excel(EXCEL_FILE_PATH)
    unique_ids = get_unique_ids(df)

    for idx, source_dir in enumerate(unique_ids, 1):
        print(f"Sample {source_dir}: {idx} of {len(unique_ids)}")

        with tempfile.TemporaryDirectory() as temp_dir_path:
            temp_dir = Path(temp_dir_path)
            print(f"Created temp file: {temp_dir}")

            source_dir_path = find_source_dir_path(DATA_ROOT_PATH, source_dir)
            if not source_dir_path:
                print(f"Path not found: {source_dir_path}")
                print(f"Missing Sample {source_dir}. \nCheck {MISSING_SAMPLES_FILE} for more info")
                with open(MISSING_SAMPLES_FILE, "a") as f:
                    f.write(source_dir + "\n")
                continue

            cellranger_output_at_source = source_dir_path / CELLRANGER_OUTPUT_DIR
            cellranger_data_final_location = DESTINATION_DRIVE / source_dir /CELLRANGER_OUTPUT_DIR

            # Check if the directory exists and has contents
            if cellranger_data_final_location.exists() and any(cellranger_data_final_location.iterdir()):                
                print(f"CellRanger output already exists: {cellranger_data_final_location}\nSkipping execution!")
            else:
                print(f"does not exist\n{cellranger_data_final_location}")
                print(f"Copying fastq files from\n{source_dir_path}")
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
                                
                #cellranger_data_final_location.mkdir(parents=True, exist_ok=True)

                shutil.copytree(cellranger_output_dir, cellranger_data_final_location, dirs_exist_ok=True)

                if not verify_contents_identical(cellranger_output_dir, cellranger_data_final_location):
                    print(f"Contents of {cellranger_output_dir} and {cellranger_data_final_location} are not identical!")

    end_time = datetime.datetime.now().replace(microsecond=0)
    print(f'Script end: {end_time}')
    print(f'Elapsed time: {end_time - start_time}')




    