from typing import Tuple
import scanpy as sc
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import AnnData
import scvi
import logging
import torch
import concurrent.futures
import time
from typing import List
import numpy as np
from scipy.sparse import csr_matrix, issparse
# ------------------------------------------
# Script Information
__author__ = "Jose L. Agraz, PhD"
__status__ = "Prototype"
__email__ = "jose@agraz.email"
__credits__ = ["Jose L. Agraz", "Parker Wilson"]
__license__ = "MIT"
__version__ = "1.0"
# ------------------------------------------
# reference: 
# Complete single-cell RNAseq analysis walkthrough | Advanced introduction
# https://www.youtube.com/watch?v=uvyG9yLuNSE&t=635s
# ------------------------------------------
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
MINIMUM_GENES      = 500
MINIMUM_CELLS      = 100
ROOT_PATH          = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR  = Path('Supporting_Files')
DATA_DIR           = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
METADATA_FILE_PATH = ROOT_PATH / SUPPORT_FILES_DIR / 'source_metadata.csv'
LOOKUP_TABLE       = { 'Healthy Reference': 0,'CKD': 1,'AKI': 2,'DM-R': 3,'None': 4 }
PROCESSED_DIR      = ROOT_PATH / Path('Pre-processed_and_Labeled_Samples')
OUTPUT_DIR         = ROOT_PATH / DATA_DIR / Path('0a0a_Results')
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------
def find_files(root_path:Path, file_name:str) -> List[Path]:
    """
    Find all instances of a file with a given name under the root path.

    Parameters:
    root_path (str): The root directory to start the search from.
    file_name (str): The name of the file to search for.

    Returns:
    list: A list of Paths where the file is found.
    """
    return [file for file in root_path.rglob(file_name)]
# ------------------------------------------   
def read_files_in_parallel(file_paths:List[Path]) -> List[AnnData]:
    """
    Reads multiple .h5ad files in parallel and returns a list of AnnData objects.

    Parameters:
    - file_paths (list of str): A list of paths to .h5ad files to be read.

    Returns:
    - A list of AnnData objects loaded from the provided file paths.
    """
    adata_array = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the file paths to the executor to read in parallel
        results = executor.map(sc.read_h5ad, file_paths)

        columns_to_keep = ['n_genes',           \
                           'n_genes_by_counts', \
                           'total_counts',      \
                           'total_counts_mt',   \
                           'pct_counts_mt',     \
                           'Sample_Name',       \
                           'Sample_Status',     \
                           'Sample_Type',       \
                           'Sample_Tissue_Type',\
                           'Sample_Sex',        \
                           'Sample_Age',        \
                           'Sample_Race',       \
                           'celltype']      
          
        for adata in results:
            # remove obs.columns that are not in columns_to_keep
            columns_to_drop = [col for col in adata.obs.columns if col not in columns_to_keep]
            adata.obs.drop(columns_to_drop, axis=1, inplace=True)
            # remove var.columns that are not in 'columns_to_keep'
            columns_to_drop = [col for col in adata.var.columns if col not in ['n_cells']]
            adata.var.drop(columns_to_drop, axis=1, inplace=True)

            # Convert the 'Sample_Age' column to integers by extracting the first two digits, handling NaN values
            adata.obs['Sample_Age'] = (
                adata.obs['Sample_Age']
                .str.extract(r'(\d{2})')  # Extract the first two digits
                .astype(float)            # Convert to float to handle NaNs
                .fillna(0)                # Replace NaN with 0
                .astype(int)              # Convert back to integer
            )            

            if issparse(adata.X):
                adata.X = adata.X.astype(np.int64)
            else:
                # For dense arrays
                adata.X = np.asarray(adata.X, dtype=np.int64)   
            adata_array.append(adata)
    
    return adata_array
# ------------------------------------------
def poor_cell_and_gene_removal(adata: AnnData) -> Tuple[AnnData, int, int]:
    """
    Removes cells with fewer than a specified number of genes and genes present in fewer than a specified number of cells.

    This function filters out low-quality cells and genes from an AnnData object based on specified thresholds. It is designed to improve the quality of the dataset for further analysis.

    Parameters:
    - adata (AnnData): The AnnData object containing single-cell gene expression data.

    Returns:
    - Tuple[AnnData, int, int]: A tuple containing the filtered AnnData object, the number of cells removed, and the number of genes removed.
    
    """
    # Count the number of cells and genes before filtering
    num_cells_before = adata.n_obs
    num_genes_before = adata.n_vars

    # Filter out cells with fewer than 200 genes and genes found in fewer than 100 cells
    print(f"Filtering genes < {MINIMUM_GENES} and cells < {MINIMUM_CELLS}..")
    sc.pp.filter_cells(adata, min_genes=MINIMUM_GENES)
    sc.pp.filter_genes(adata, min_cells=MINIMUM_CELLS)

    # Calculate the number of cells and genes removed
    num_cells_removed = num_cells_before - adata.n_obs
    num_genes_removed = num_genes_before - adata.n_vars

    print(f"Number of input genes:     {num_genes_before}")
    print(f"Number of genes removed:   {num_genes_removed}")
    print(f"Number of genes remaining: {adata.n_vars}")
    print(f"Number of input cells:     {num_cells_before}")
    print(f"Number of cells removed:   {num_cells_removed}")
    print(f"Number of cells remaining: {adata.n_obs}")
    
    return adata, num_cells_removed, num_genes_removed
# ------------------------------------------
if __name__ == "__main__":    
    scvi.settings.seed  = 0
    start_time = time.time()
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    
    adata_array       = []
    found_files       = []
    file_of_interest  = 'filtered_feature_bc_matrix.h5_sample-filtered-and-labeled.h5ad'
    # -------------------------------
    # csv file with valid dataset metadata
    metadata_df  = pd.read_csv(METADATA_FILE_PATH)
    dataset_name = list(set(metadata_df.sample_name.tolist()))
    
    found_files = list(PROCESSED_DIR.rglob("*.h5ad"))

    logging.info(f"Found {len(found_files)} files to concatenate")

    adata_array = read_files_in_parallel(found_files) 
    # -------------------------------
    logging.info(f"Concatenating {len(found_files)} samples")
    # adata                = ad.concat(adata_array, join="outer", merge="first")
    import anndata
    adata = anndata.concat(adata_array, label="batch")
    adata.var['n_cells'] = np.sum(adata.X > 0, axis=0).A1
    adata,_,_            = poor_cell_and_gene_removal(adata)
    # -------------------------------
    adata.obs['Sample_Name'] = adata.obs['Sample_Name'].astype(str)
    adata.obs['Sample_Status'] = adata.obs['Sample_Status'].astype(str)
    adata.obs['Sample_Type'] = adata.obs['Sample_Type'].astype(str)
    adata.obs['Sample_Tissue_Type'] = adata.obs['Sample_Tissue_Type'].astype(str)
    adata.obs['Sample_Sex'] = adata.obs['Sample_Sex'].astype(str)
    adata.obs['Sample_Age'] = adata.obs['Sample_Age'].astype(str)
    adata.obs['Sample_Race']= adata.obs['Sample_Race'].astype(str)
    adata.obs['celltype']    = adata.obs['celltype'].astype(str)
    # -------------------------------
    print('Remove non-Male/Female')
    adata       = adata[adata.obs.Sample_Sex.isin(['Male', 'Female'])]
    print('Remove missing Age data')
    adata       = adata[adata.obs.Sample_Age != 'None']
    print('Map disease')
    adata.obs['Sample_Disease'] = adata.obs['Sample_Tissue_Type'].map(LOOKUP_TABLE)  
    # -------------------------------
    # If Male then 0, if Female then 1
    adata.obs['Sample_Gender'] = adata.obs.Sample_Sex.apply(lambda x: 0 if x == 'Male' else 1)
    # Only use the first two digits of any age
    adata.obs['Sample_Ages'] = adata.obs['Sample_Age'].str[:2].astype(int)
    # adata.X     = csr_matrix(adata.X)
    result_file = OUTPUT_DIR / f'{len(found_files)}_samples_concatenated_v2.h5ad'

    # Check if the file exists and then delete it
    if result_file.exists():
        result_file.unlink()    
    
    adata.write_h5ad(str(result_file))

    sequential_duration = time.time() - start_time
    print(f"Processed: {adata.n_obs} observations and {adata.n_vars} features")
    print(f"Processed: {len(found_files)} samples")
    print(f"File name: {result_file.name}")
    print(f"File path: {result_file.parent}")
    print(f"Processing time: {sequential_duration} seconds")    
  
    print('Done')
