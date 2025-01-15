"""
Preprocess Samples Script for Single-Cell Genomic Data

This script is designed to preprocess single-cell genomic data. It includes functionalities for doublet detection, gene filtering, data cleaning, and more, utilizing libraries such as Scanpy, Pandas, NumPy, and Scrublet.

Usage:
    Ensure all dependencies are installed and the dataset is properly formatted as per the requirements of the used libraries.
    Run the script in an environment where the necessary Python packages are installed.

Author: Jose L. Agraz and Parker Wilson
Date: Feb 23, 2024
Version: 1.0
"""
from pathlib import Path
from anndata import AnnData
import anndata as ad
import scanpy  as sc
import pandas  as pd
import numpy   as np
import concurrent.futures
import re
import scvi
import torch
# ------------------------------------------
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
scvi.settings.seed = 0
REF_ROOT_PATH      = Path("/media/KPMP_Data/Privately_Available_Data/Supporting_Files/References/Atlas_References")
DATA_FILE_FORMAT   = 'h5ad'
OUTPUT_FILE_NAME   = f'Kidney_Reference_Atlas.{DATA_FILE_FORMAT}'
# ------------------------------------------   
def list_h5ad_files(directory):
    """
    Returns a list of .h5ad files in the given directory using pathlib.

    Parameters:
    - directory (str): The path to the directory to search in.

    Returns:
    - list: A list of paths to .h5ad files found in the directory.
    """
    # Convert the directory to a Path object
    path = Path(directory)
    
    # Use a glob pattern to find all .h5ad files
    h5ad_files = path.glob(f'*.{DATA_FILE_FORMAT}')
    
    # Convert the Path objects to strings (if needed) and return them as a list
    return [str(file) for file in h5ad_files]
# ------------------------------------------
def clean_reference_data(reference_data: AnnData) -> AnnData:
    """
    Cleans the given AnnData object by simplifying its structure. This function performs the following operations:
    - Strips cell barcode prefixes to standardize barcode formats across datasets.
    - Clears 'obsm', 'varm', and 'uns' slots to remove unneeded metadata and annotations.
    - Resets the 'obs' dataframe to only contain the index, removing all other columns.

    Parameters:
    - reference_data (AnnData): The AnnData object to be cleaned.

    Returns:
    - AnnData: The cleaned AnnData object with simplified structure.
    """    
    # Clear other stuff
    reference_data.obsm.clear()
    reference_data.varm.clear()
    reference_data.uns.clear()
    reference_data.obsp.clear()
    # delete all var but the index
    reference_data.var = pd.DataFrame(index=reference_data.var.index)

    return reference_data
# ------------------------------------------
def remove_barcode_with_dissimilar_celltypes(adata: AnnData) -> AnnData:
    """
    Filters the input AnnData object to remove barcodes associated with dissimilar cell types. This function ensures that
    only barcodes corresponding to a single cell type or identical cell types are retained. It is useful for cleaning data
    where a single barcode (representing a single cell) has been mistakenly assigned multiple cell types due to technical errors
    or data processing artifacts.

    Parameters:
    - adata (AnnData): The AnnData object containing the single-cell genomic data.

    Returns:
    - AnnData: A filtered AnnData object with barcodes having consistent cell types.
    """    
    DELETE_BARCODE     = 'delete'
    KEEP_BARCODE       = 'keep'
    COLUMN_OF_INTEREST = 'celltype'
    COLUMN_NAME        = 'action'

    print("Scanning barcodes for duplicates")
    # Create a DataFrame to keep track of decisions (keep or delete) for each barcode
    decisions              = pd.DataFrame(index=adata.obs.index.unique())
    decisions[COLUMN_NAME] = DELETE_BARCODE  # Default action is to delete

    # Iterate through each unique barcode
    for barcode in adata.obs.index.unique():
        # Extract celltype values for the current barcode
        celltypes = adata.obs.loc[adata.obs.index == barcode, COLUMN_OF_INTEREST].tolist()
        
        # Condition 1: If the celltype value contains a single item
        if len(celltypes) == 1:
            decisions.at[barcode, COLUMN_NAME] = KEEP_BARCODE
        # Condition 2: More than one item, check if all items are identical
        elif len(celltypes) != 1:
            # print(celltypes)
            if len(set(celltypes)) <= 1:
                # All items are identical, mark to keep and no need to adjust celltype as it's already done
                decisions.at[barcode, COLUMN_NAME] = KEEP_BARCODE

    # Filter the adata object to keep barcodes labeled 'keep'
    to_keep        = decisions[decisions[COLUMN_NAME] == KEEP_BARCODE].index
    filtered_adata = adata[adata.obs.index.isin(to_keep)]
    # Remove duplicate barcodes and celltypes
    filtered_adata.obs['barcode'] = filtered_adata.obs.index   

    non_duplicates = ~filtered_adata.obs.duplicated(subset=['barcode', 'celltype'], keep='first')
    filtered_adata = filtered_adata[non_duplicates].copy()
    filtered_adata.obs.drop('barcode', axis=1, inplace=True)

    # Summarize the decisions
    num_kept    = decisions[decisions[COLUMN_NAME] == KEEP_BARCODE].shape[0]
    num_deleted = decisions[decisions[COLUMN_NAME] == DELETE_BARCODE].shape[0] 

    # Print the summary to the user
    print(f"Summary of barcode processing:")
    print(f"Barcodes kept   : {num_kept}")
    print(f"Barcodes deleted: {num_deleted}")
    print(f"Percent deleted : {100*num_deleted/(num_kept+num_deleted):.1f}%")
    return filtered_adata
# ------------------------------------------
def clean_barcode(barcode):
    REGEX = r'[ACTG]{16}'
    match = re.search(REGEX, barcode)
    if match:
        return match.group(0)  # Return the matched group
    else:
        return barcode  # Return the original barcode if no match was found (or adjust as needed)
# ------------------------------------------
def prepare_anndata_object(adata: AnnData) -> AnnData:
    """
    Prepares an AnnData object by cleaning its barcode format and filtering out barcodes associated with dissimilar cell types.
    This function performs two main tasks:
    1. Cleans the barcode format by removing any leading or trailing identifiers, making the barcodes uniform across different datasets.
    2. Filters out barcodes that are associated with more than one cell type, ensuring that each barcode represents a single, consistent cell type.

    Parameters:
    - adata (AnnData): The AnnData object to be processed, containing single-cell genomic data.

    Returns:
    - AnnData: The processed AnnData object with uniform barcode formats and consistent cell type associations.
    """    
    adata.obs.index = adata.obs.index.map(clean_barcode)
        
    adata.obs       = adata.obs[['celltype']]
    adata           = remove_barcode_with_dissimilar_celltypes(adata)   
    # adata           = clean_reference_data(adata)
    return adata
# ------------------------------------------
def load_and_process_data(file_name):
    """
    Loads a single-cell dataset from a specified .h5ad file and applies preprocessing steps to clean and prepare the data
    for downstream analysis. This function wraps several preprocessing steps including barcode cleaning, and filtering
    barcodes with dissimilar cell types to ensure data quality and consistency.

    The preprocessing steps are designed to standardize and simplify the dataset, making it ready for further analysis such
    as clustering, visualization, or integration with other datasets.

    Parameters:
    - file_name (str): The name of the file to be loaded. This file should be in the .h5ad format, a common format for
                       storing single-cell gene expression data.

    Returns:
    - AnnData: The preprocessed AnnData object ready for further analysis.
    """
    ref_data = sc.read_h5ad(REF_ROOT_PATH / file_name)
    ref_data = prepare_anndata_object(ref_data)
    return ref_data
# ------------------------------------------
if __name__ == "__main__":
    """
    The main entry point of the script. This section orchestrates the preprocessing of single-cell genomic data files.
    It utilizes multiprocessing to parallelize the loading and processing of data files to enhance efficiency.

    Steps:
    1. Initializes the reference list and sets the path to the directory containing reference data files.
    2. Lists all .h5ad files in the specified directory.
    3. Parallelizes the processing of these files using ProcessPoolExecutor to load and preprocess each file concurrently.
    4. Concatenates the processed AnnData objects into a single AnnData object.
    5. Removes the 'batch' column from the concatenated object and creates a new variable dataframe.
    6. Prints a completion message.
    """
    print(f"Last run with scvi-tools version: {scvi.__version__}")
    reference_list = []    
    reference_file_name = list_h5ad_files(REF_ROOT_PATH)       
    
    # For troubleshooting
    # adata = load_and_process_data(reference_file_name[0])

    # Using a ProcessPoolExecutor to parallelize file loading and processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to executor
        future_to_file = {executor.submit(load_and_process_data, file_name): file_name for file_name in reference_file_name}
        
        reference_list = []
        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                data = future.result()
                reference_list.append(data)
            except Exception as exc:
                print(f"{file_name} generated an exception: {exc}")
    
    # concatenate the loaded and processed data
    concatenated_reference = reference_list[0]
    for ref_data in reference_list[:-1]:
        concatenated_reference = concatenated_reference.concatenate(ref_data)  
    
    # concatenated_reference.obs.drop('batch', axis=1, inplace=True)   
    # concatenated_reference = prepare_anndata_object(concatenated_reference)
    # concatenated_reference = ad.concat(reference_list)
    concatenated_reference.write(REF_ROOT_PATH / OUTPUT_FILE_NAME)

    print('Done')