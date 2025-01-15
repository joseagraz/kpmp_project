import re
from typing import Tuple
import scanpy as sc
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import AnnData
from pandas import DataFrame
import scvi
import logging
import torch
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
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
DOUBLETS           = 'doublet'
MINIMUM_GENES      = 500
MINIMUM_CELLS      = 100
# 12TB disk path
ROOT_PATH          = Path('/media/jagraz/8TB/KPMP_Data/Privately_Available_Data')
# NAS path
# ROOT_PATH         = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR  = Path('Supporting_Files')
DATA_DIR           = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
METADATA_FILE_PATH = ROOT_PATH / SUPPORT_FILES_DIR / 'metadata.csv'
LOOKUP_TABLE       = { 'Healthy Reference': 0,'CKD': 1,'AKI': 2,'DM-R': 3,'None': 4 }
OUTPUT_FILE_NAME   = ROOT_PATH / DATA_DIR / 'combined.h5ad'
KPMP_EXCEL_LEGEND_PATH  = ROOT_PATH / Path(SUPPORT_FILES_DIR) / Path('Excel_Files') / 'Parker_Wilson_PennMed_Update_V2.xlsx'
UPPER_QUANTILE          = 0.98
LOWER_QUANTILE          = 0.02
MT_QUANTILE_TITLE       = f'{UPPER_QUANTILE}%_quantiles_{LOWER_QUANTILE}%'
MT_COUNT_LIMIT          = 30
MITO_TITLE              = f'mitochondrial_genes_<{MT_COUNT_LIMIT}_pct_count'
MINIMUM_GENES           = 500
MINIMUM_CELLS           = 50
NORMALIZATION_SCALE     = 1e4
COLUMN_OF_INTEREST      = 'celltype'
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
def Find_Metadata(df: DataFrame, sample_name: str, metadata_title: str) -> str:
    """
    Retrieves specific metadata for a given sample from a pandas DataFrame.

    This function searches the DataFrame for rows matching the specified sample name,
    then extracts and returns the desired piece of metadata from the first matching row.
    Assumes all matching rows contain duplicate information and only processes the first match.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing sample metadata.
    - sample_name (str): The name of the sample to search for within the DataFrame.
    - metadata_title (str): The title of the metadata column from which to extract information.

    Returns:
    - str: The metadata of interest for the specified sample. Returns 'None' if the sample or metadata is not found.

    Note:
    - The function is designed to work with DataFrames where each row represents a sample and columns represent metadata fields.
    - It's assumed that the 'Internal Package ID' column is used to identify samples.
    """
    SAMPLE_OF_INTEREST_COLUMN_TITLE = 'Internal Package ID'
    
    # Find all rows matching the specified sample name and fill missing values with 'None'
    sample_rows = df[df[SAMPLE_OF_INTEREST_COLUMN_TITLE].astype(str).str.contains(sample_name, na=False)].fillna('None')
    
    # Fetch the first matching row as all others are considered duplicates
    sample_row = sample_rows.iloc[0]
    
    # Extract and return the requested metadata
    metadata_of_interest = sample_row[metadata_title]
    print(metadata_of_interest)
    return metadata_of_interest
# ------------------------------------------
def merge_duplicate_genes(adata: AnnData) -> AnnData:
    """
    Merges duplicate genes in an AnnData object by summing their expression counts.

    This function identifies all duplicate gene names within the `var_names` of the
    input AnnData object. For each set of duplicates, their expression counts across
    all cells are summed. The resulting dataset contains unique genes only, with
    expression counts adjusted to reflect the merging of duplicates. A new AnnData
    object is returned, preserving the original metadata and annotations.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing single-cell expression data, potentially
        with duplicate gene names.

    Returns
    -------
    AnnData
        A new AnnData object with duplicate genes merged by summing their counts. The
        returned object retains the original's observation (obs) and variable (var)
        annotations, adjusted for the merged gene set.

    Notes
    -----
    - The function handles both dense and sparse matrix formats for the expression data
      (`adata.X`). Sparse data is converted to dense for processing and converted back
      if the original data was sparse.
    - It is assumed that the gene names are stored in `adata.var_names`.
    - The `var` DataFrame of the returned AnnData object is adjusted to reflect only
      the unique genes, with the first occurrence's annotations preserved.

    Examples
    --------
    >>> adata = AnnData(...)  # Load or create an AnnData object
    >>> merged_adata = merge_duplicate_genes(adata)
    >>> print(merged_adata.var_names)  # Unique gene names
    """    
    print("Identifying duplicate genes...")
    # Detect all genes that are duplicates (True for all occurrences of a duplicate)
    dup_genes_mask = adata.var_names.duplicated(keep=False)
    
    if not any(dup_genes_mask):
        print("No duplicate genes found. Returning original adata object.")
        return adata
    
    # Creating a DataFrame for manipulation
    gene_counts = pd.DataFrame(adata.X.toarray(), columns=adata.var_names) if isinstance(adata.X, (np.ndarray, np.generic)) else pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
    
    # Summing counts for each gene, this automatically handles duplicates by summing their counts
    summed_gene_counts = gene_counts.groupby(gene_counts.columns, axis=1).sum()
        
    # Update gene names to reflect the changes
    new_gene_names = summed_gene_counts.columns
    
    # Creating a new AnnData object
    print("Creating a new anndata object with updated counts...")
    if isinstance(adata.X, np.ndarray):
        new_X = summed_gene_counts.values
    else:        
        new_X = csr_matrix(summed_gene_counts.values)
    
    new_var = adata.var.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    new_var = new_var.loc[new_gene_names]
    
    new_adata = AnnData(new_X, obs=adata.obs.copy(), var=new_var)
    
    print("Operation completed. Returning updated adata object.")
    return new_adata

def process_sample(adata: AnnData, sample_name: str) -> AnnData:
    """
    Processes a single-cell RNA sequencing sample, performing various preprocessing steps 
    including doublet detection, poor cell and gene removal, mitochondrial and ribosomal gene removal. 
    It also annotates the AnnData object with sample metadata and writes processing information to a CSV file.

    Parameters:
    - data_location (str): The file path to the input data in .h5 format.
    - sample_name (str): The name of the sample being processed.

    Returns:
    - AnnData: The processed AnnData object with cells and genes filtered, and metadata annotations added.

    Raises:
    - Exception: If there are issues in doublet detection or other preprocessing steps, an error is logged with details.

    Note:
    - This function assumes the existence of several globally defined variables such as `KPMP_EXCEL_LEGEND_PATH` 
      and `CVS_FILE_PATH`, which should be set before calling this function.
    - It uses external functions like `sum_duplicate_genes`, `doublet_removal`, `detect_doublets`, 
      `poor_cell_and_gene_removal`, `mitochondrial_genes_removal`, `ribo_genes_removal`, and `Find_Metadata` 
      which should be defined elsewhere in the codebase.
    - It also assumes the existence of a specific Excel file for metadata and writes processing results to a CSV file.
    """
    doublets_removed_solo          = 0
    doublets_removed_DoubletDetection  = 0

    # Define the headers for the CSV file
    headers = [ "sample_name", 
                "number_of_input_cells", 
                "number_of_input_genes",   
                f"num_cells_poor_removed_{MINIMUM_CELLS}",
                f"num_genes_poor_removed_{MINIMUM_GENES}", 
                f"num_cells_mito_removed_{MT_QUANTILE_TITLE}",
                f"num_genes_mito_removed_{MT_QUANTILE_TITLE}", 
                "ribo_genes_removed", 
                "doublets_removed_SOLO",
                "doublets_removed_DoubletDetection",
                "num_of_output_cells",
                "num_of_output_genes",
                "Sample_Status", 
                "Sample_Type",
                "Sample_Tissue_Type", 
                "Sample_Sex", 
                "Sample_Age", 
                "Sample_Race"]

    Metadata_df     = pd.read_excel(KPMP_EXCEL_LEGEND_PATH, engine='openpyxl')
    #
    # Initialize counters for the number of cells/genes removed
    num_of_original_cells = adata.n_obs
    num_of_original_genes = adata.n_vars
    # ---------------------------------------  
    # Sum duplicate gene counts
    adata.var_names = adata.var_names.str.upper()
    adata           = merge_duplicate_genes(adata)
    # ---------------------------------------
    num_of_output_cells = adata.n_obs
    num_of_output_genes = adata.n_vars
    # Annotate the AnnData object with sample metadata
    Sample_Status      = Find_Metadata(Metadata_df, sample_name, 'Status')
    Sample_Type        = Find_Metadata(Metadata_df, sample_name, 'Sample Type_y')
    Sample_Tissue_Type = Find_Metadata(Metadata_df, sample_name, 'Tissue Type_y')
    Sample_Gender      = Find_Metadata(Metadata_df, sample_name, 'Sex')
    Sample_Age         = Find_Metadata(Metadata_df, sample_name, 'Age (Years) (Binned)')
    Sample_Race        = Find_Metadata(Metadata_df, sample_name, 'Race')

    adata.obs['Sample_Name']        = sample_name.strip()
    adata.obs['Sample_Status']      = Sample_Status
    adata.obs['Sample_Type']        = Sample_Type
    adata.obs['Sample_Tissue_Type'] = Sample_Tissue_Type
    adata.obs['Sample_Sex']         = Sample_Gender
    adata.obs['Sample_Age']         = Sample_Age
    adata.obs['Sample_Race']        = Sample_Race
    return adata
# ------------------------------------------ 
def process_file(file_path):
    
    OUTPUT_FILE_EXTENSION   = 'h5ad'
    OUTPUT_FILE_NAME        = f'_sample-filtered-and-labeled_celbender.{OUTPUT_FILE_EXTENSION}'
    input_file = Path(str(file_path)+OUTPUT_FILE_NAME)
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
    
    # Read the file into an AnnData object
    adata = sc.read_h5ad(input_file)
    # Typecast the matrix to np.int64
    adata.X = adata.X.astype(np.int64)

    columns_to_drop = [col for col in adata.obs.columns if col not in columns_to_keep]
    adata.obs.drop(columns_to_drop, axis=1, inplace=True)
    columns_to_drop = [col for col in adata.var.columns if col not in ['n_cells','gene_ids','feature_types','genome']]
    adata.var.drop(columns_to_drop, axis=1, inplace=True)

    return adata

def process_files_in_parallel(file_paths):
    # Create a thread pool executor to process files in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_file function over the list of file paths and collect results
        results = list(executor.map(process_file, file_paths))
    return results

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
def extract_sample_names_from_log(filepath):
    """
    Extracts sample names from each line of a given log file.
    Assumes that the sample names are in UUID format and are part of a longer path in each log entry.
    
    Args:
    filepath (str): Path to the log file.

    Returns:
    list of str: List of extracted sample names (UUIDs).
    """
    sample_names = []
    uuid_pattern = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
    
    with open(filepath, 'r') as file:
        for line in file:
            # Search for the UUID pattern in each line
            match = uuid_pattern.search(line)
            if match:
                sample_names.append(match.group())
    
    return sample_names

if __name__ == "__main__":    
    scvi.settings.seed  = 0
    start_time = time.time()
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    
    adata_array       = []
    found_files       = []
    # file_of_interest  = 'filtered_feature_bc_matrix.h5_sample-filtered-and-labeled.h5ad'
    file_of_interest  = 'raw_feature_bc_matrix_cellbender_denoised_filtered.h5'
    # -------------------------------
    # csv file with valid dataset metadata
    metadata_df  = pd.read_csv(METADATA_FILE_PATH)
    dataset_name = list(set(metadata_df.sample_name.tolist()))

    dataset_name  = extract_sample_names_from_log('/media/jagraz/8TB/KPMP_Data/Privately_Available_Data/Supporting_Files/success_short.log')

    for sample_name in dataset_name:
        data_location = ROOT_PATH / Path(DATA_DIR) / Path(sample_name.strip()) / Path('cellranger_output/outs') / Path(file_of_interest)
        found_files.append(data_location)
    # -------------------------------
    logging.info(f"Found {len(found_files)} files to concatenate")
    Metadata_df     = pd.read_excel(KPMP_EXCEL_LEGEND_PATH, engine='openpyxl')

    adata_array = process_files_in_parallel(found_files) 
    # -------------------------------
    logging.info(f"Concatenating {len(found_files)} samples")
    adata                = ad.concat(adata_array, join="outer", merge="first")
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
    result_file = ROOT_PATH / DATA_DIR / 'combined_10_samples_with_cellbender.h5ad'
    # Check if the file exists and then delete it
    if result_file.exists():
        result_file.unlink()    
    adata.write_h5ad(str(result_file))

    sequential_duration = time.time() - start_time
    print(f"Processing time: {sequential_duration} seconds")    

    print('Done')