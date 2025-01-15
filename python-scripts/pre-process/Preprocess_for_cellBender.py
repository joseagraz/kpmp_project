"""
Preprocess Samples Script for Single-Cell Genomic Data

This Python script automates data cleaning and label transfering using an Atlas in preparation for data integration. 
Utilizing parallel processing, Scanpy, Pandas, NumPy, and scVI, the scripts streamlines the analysis 
pipeline, ensuring that your data is ready for downstream analysis.

Author: Jose L. Agraz and Parker Wilson
Date: March 27, 2024
Version: 1.0
"""
from typing import Tuple
from typing import List
from pandas import DataFrame
from pathlib import Path
from scipy.sparse import csr_matrix, issparse
import scipy
import anndata as ad
from anndata import AnnData
from datetime import datetime
import scvi
import logging
import concurrent.futures
import torch
import re
import numpy as np
import csv
import os
import scrublet as scr, scanpy as sc, pandas as pd,matplotlib.pyplot as plt
import doubletdetection
# ------------------------------------------
# reference: 
# Complete single-cell RNAseq analysis walkthrough | Advanced introduction
# https://www.youtube.com/watch?v=uvyG9yLuNSE&t=635s
# ------------------------------------------
global gene_expression_df
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
# ------------------------------------------
# 8TB disk path
ROOT_PATH               = Path('/media/jagraz/8TB/KPMP_Data/Privately_Available_Data')
# ROOT_PATH              = Path('/media/jagraz/3T_USB')
# ------------------------------------------
# NAS path
# ROOT_PATH             = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR       = 'Supporting_Files'
DATA_DIR                = 'Original_Download_KPMP_S3_Bucket_Oct_16_2023'
SOURCE_DATA_PATH        = Path('cellranger_output/outs/filtered_feature_bc_matrix')
CVS_FILE_PATH           = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'metadata.csv'
RIBO_LOOKUP_FILE_PATH   = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
LABEL_REFERENCE_FILE    = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'References/Atlas_References'
KPMP_EXCEL_LEGEND_PATH  = ROOT_PATH / Path(SUPPORT_FILES_DIR) / Path('Excel_Files') / 'Parker_Wilson_PennMed_Update_V2.xlsx'
COMBINED_OUTPUT_FILE    = ROOT_PATH / Path(DATA_DIR) / 'combined.h5ad'
SAMPLES_OF_INTEREST_DIR = ROOT_PATH / Path(DATA_DIR)
STORE_FILE_PATH         = ROOT_PATH / SUPPORT_FILES_DIR / 'valid_files.csv'
UPPER_QUANTILE          = 0.98
LOWER_QUANTILE          = 0.02
MT_QUANTILE_TITLE       = f'{UPPER_QUANTILE}%_quantiles_{LOWER_QUANTILE}%'
MT_COUNT_LIMIT          = 30
MITO_TITLE              = f'mitochondrial_genes_<{MT_COUNT_LIMIT}_pct_count'
MINIMUM_GENES           = 500
MINIMUM_CELLS           = 50
NORMALIZATION_SCALE     = 1e4
COLUMN_OF_INTEREST      = 'celltype'
DOUBLETS_PREDICTION     = 'prediction'    
LABEL_PREDICTION        = 'predicted'
RIBOSOME_GENES_TITLE    = 'ribo_presence'
SCVI_LATENT_KEY         = "X_scVI"
DOUBLETS                = 'doublet'
OUTPUT_FILE_EXTENSION   = 'h5ad'
OUTPUT_FILE_NAME        = f'_sample-filtered-and-labeled.{OUTPUT_FILE_EXTENSION}'
DATA_FILE_FORMAT        = 'h5ad'
ATLAS_FILE_NAME         = LABEL_REFERENCE_FILE / f'Kidney_Reference_Atlas.{DATA_FILE_FORMAT}'
KPMP_EXCEL_COLUMN_NAMES = ['Participant ID',
                           'Protocol_x',
                           'Sample Type_x',
                           'Tissue Type_x',
                           'Experimental Strategy',
                           'Data Format',
                           'Platform',
                           'Data Type',
                           'Size',
                           'File Name',
                           'Internal Package ID',
                           'uuid',
                           'filename',
                           '2',
                           'checksum',
                           'Status',
                           'Tissue Source',
                           'Protocol_y',
                           'Sample Type_y',
                           'Tissue Type_y',
                           'Sex',
                           'Age (Years) (Binned)',
                           'Race',
                           'KDIGO Stage',
                           'Baseline eGFR (ml/min/1.73m2) (Binned)',
                           'Proteinuria (mg) (Binned)',
                           'A1c (%) (Binned)']
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------
def read_and_process_data(sample_name: str, root_path: Path, data_dir: str, source_data_path: Path) -> AnnData:
    """
    Reads and processes single-cell data for a given sample.

    This function constructs the path to the sample data using the provided root path, data directory, 
    and sample name. It then processes the sample by calling `process_sample` and returns the processed 
    AnnData object. If an error occurs during processing, logs the error and returns None.

    Parameters: 
    - sample_name (str): The name of the sample to process.
    - root_path (Path): The root path to the base directory where data is stored.
    - data_dir (str): The directory under the root path where sample directories are located.
    - source_data_path (Path): The relative path to the source data file from the sample directory. 
      The actual file used will have a `.h5` suffix.

    Returns:
    - anndata.AnnData or None: The processed AnnData object if successful, None otherwise.

    Raises:
    - Logs an error message if any exception occurs during processing.
    """
    data_location = root_path / Path(data_dir) / Path(sample_name.strip()) / source_data_path.with_suffix('.h5')    
    adata         = sc.read_10x_h5(data_location)  
    new_file_name = data_location.parent / "raw_feature_bc_matrix.h5"
    # adata.write_h5ad(new_file_name)
    file_exists = STORE_FILE_PATH.exists()               
    try:
        # Open the file with the appropriate mode
        with STORE_FILE_PATH.open('a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)

            # If creating the file for the first time, write the header
            if not file_exists:
                writer.writerow(['File_Paths','Observations'])
            else:
                writer.writerow([new_file_name,10*adata.n_obs])

        print(f"File paths have been {'appended to' if file_exists else 'written to'} {STORE_FILE_PATH}.")
    except IOError as e:
        print(f"An error occurred while accessing {STORE_FILE_PATH}: {e}")

    return []
# ------------------------------------------   
def find_subdirectories(root_path: Path) -> List[str]:
    """
    Finds and lists the names of subdirectories that contain a specific target directory 
    within a given root directory. This function is particularly tailored to search for 
    sample names based on a directory structure convention where sample names are 
    positioned immediately before a 'cellranger_output' directory.

    Parameters:
    - root_path (Path): The root directory path from which to start the search. This should 
      be a Path object representing the top-level directory.

    Returns:
    - List[str]: A list of sample names extracted from the directory paths that contain 
      the target 'cellranger_output' directory.
    """
    target_path = 'cellranger_output'
    sample_names = []

    for path in root_path.rglob(target_path):
        # Assuming sample name is the directory immediately before 'cellranger_output'
        sample_name = path.parts[-2]
        sample_names.append(sample_name)

    return sample_names
# ------------------------------------------  
if __name__ == "__main__":
    # Set the normalization scale and seed for reproducibility    
    scvi.settings.seed = 0
    print(f"Last run with scvi-tools version: {scvi.__version__}")

    # Set default figure parameters and higher precision for float32 matrix multiplication
    sc.settings.n_jobs=8
    # sc.set_figure_params()
    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")

    # Initialize 
    adata_array:  List[sc.AnnData] = []
    sample_names: List[str]        = find_subdirectories(SAMPLES_OF_INTEREST_DIR)
    completed_samples              = 0
    total_samples                  = len(sample_names)

    if STORE_FILE_PATH.exists():
        print(f"Removing existing file: {STORE_FILE_PATH}")
        STORE_FILE_PATH.unlink()

    # Remove existing sample tracker log, if any
    # if CVS_FILE_PATH.exists():
    #     CVS_FILE_PATH.unlink()

    # -------------------------------
    # Comment line before after done testing!!!
    # sample_names=sample_names[1:20]   
    # adata=read_and_process_data('b4db7d55-83fc-491f-a8cb-e9473ed36f97', ROOT_PATH, DATA_DIR, SOURCE_DATA_PATH)
    # adata=read_and_process_data('0a8d4f18-84ca-4593-af16-3aaf605ca328', ROOT_PATH, DATA_DIR, SOURCE_DATA_PATH)
    # sample_names = ['3f79f07f-c357-44a3-866e-1999820ab445', \
    #                 'e048e99c-c18c-44b8-9c1f-db9730f1f240' ]
    # -------------------------------         

    # sample_names = remove_matching_elements(sample_names)      

    # Process samples concurrently to improve efficiency
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sample_name in sample_names:
            logging.info("----------------------------------------------")
            logging.info(f"Starting processing for sample: {sample_name} ({len(futures) + 1}/{total_samples})")
            future = executor.submit(read_and_process_data, sample_name, ROOT_PATH, DATA_DIR, SOURCE_DATA_PATH)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed_samples += 1
            if result is not None:
                adata_array.append(result)
                logging.info("----------------------------------------------")
                logging.info(f"Completed processing sample {completed_samples} of {total_samples}")
                logging.info("----------------------------------------------")
            else:
                logging.error(f"A sample failed to process or returned no data ({completed_samples}/{total_samples})")

    # Concatenate results from all samples into a single AnnData object
    # logging.info(f"Concatenating {total_samples} samples")
    # adata = sc.concat(adata_array, index_unique='_')
    
    # Ensure the data matrix is stored in a compressed sparse row format for efficiency
    # adata.X = csr_matrix(adata.X)

    # Save the combined dataset to a file, overwriting if it already exists
    # if COMBINED_OUTPUT_FILE.exists():
    #     COMBINED_OUTPUT_FILE.unlink()
    # adata.write_h5ad(COMBINED_OUTPUT_FILE)

    # plt.figure(figsize=(10, 8))
    # ax = sc.pl.umap(adata, color='celltype', show=False) 
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('/home/jagraz/Pictures/fig.png',bbox_inches='tight')

    print('Done')


