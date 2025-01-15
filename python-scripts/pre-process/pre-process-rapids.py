"""
Preprocess Samples Script for Single-Cell Genomic Data

This Python script automates data cleaning and label transfering using an Atlas in preparation for data integration. 
Utilizing parallel processing, Scanpy, Pandas, NumPy, and scVI, the scripts streamlines the analysis 
pipeline, ensuring that your data is ready for downstream analysis.

Author: Jose L. Agraz and Parker Wilson
Date: March 27, 2024
Version: 1.0
"""
import rapids_singlecell as rsc
import scanpy as sc
import cupy as cp
import rmm
import time
import rapids_singlecell as rsc
from rmm.allocators.cupy import rmm_cupy_allocator
import warnings
import gc
warnings.filterwarnings("ignore")

from typing import Tuple
from typing import List
from pandas import DataFrame
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, issparse, vstack
import scipy
import anndata as ad
from anndata import AnnData
from datetime import datetime
import numpy as np
# import scvi
import logging
import concurrent.futures
# import torch
import re
import numpy as np
import csv
import os
import scanpy as sc
import matplotlib.pyplot as plt
# import omicverse as ov
import rapids_singlecell as rsc
# ------------------------------------------
# reference: 
# Complete single-cell RNAseq analysis walkthrough | Advanced introduction
# https://www.youtube.com/watch?v=uvyG9yLuNSE&t=635s
# ------------------------------------------
ROOT_PATH               = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR       = 'Supporting_Files'
DATA_DIR                = 'Original_Download_KPMP_S3_Bucket_Oct_16_2023'
SOURCE_DATA_PATH        = Path('cellranger_output/outs/filtered_feature_bc_matrix')
CVS_FILE_PATH           = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'metadata.csv'
RIBO_LOOKUP_FILE_PATH   = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
LABEL_REFERENCE_FILE    = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'References/Atlas_References'
KPMP_EXCEL_LEGEND_PATH  = ROOT_PATH / Path(SUPPORT_FILES_DIR) / Path('Excel_Files') / 'Parker_Wilson_PennMed_Update_V2.xlsx'
COMBINED_OUTPUT_FILE    = ROOT_PATH / Path(DATA_DIR) / 'combined.h5ad'
OUTPUT_DIR              = ROOT_PATH / DATA_DIR / Path('0a0a_Results')
SAMPLES_OF_INTEREST_DIR = ROOT_PATH / Path(DATA_DIR)
PROCESSED_SAMPLES_DIR   = ROOT_PATH / Path('Pre-processed_and_Labeled_Samples')
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
    doublets_removed_scrublet      = 0
    doublets_removed_DoubletDetection  = 0
    num_cells_poor_removed = 0
    num_genes_poor_removed = 0
    num_cells_mito_removed = 0
    num_genes_mito_removed = 0
    ribo_genes_removed = 0

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
                "Sample_Race", 
                "Time_Elapsed_Hours",
                "Time_Elapsed_Minutes"]

    start_time      = datetime.now()
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
    # adata, \
    #     num_cells_poor_removed, \
    #     num_genes_poor_removed  = poor_cell_and_gene_removal(adata)
    # adata, \
    #     num_cells_mito_removed, \
    #     num_genes_mito_removed  = mitochondrial_genes_removal(adata)
    # adata, \
    #     ribo_genes_removed      = ribo_genes_removal(adata)    
    # ---------------------------------------
    # Doublet detection
    # print('Doublet detection using scVI SOLO')
    # try:
    #     adata, doublets_removed_solo     = doublet_removal_solo(adata)    
    # except Exception as e:
    #     logging.error(f"Error processing doublets SOLO {sample_name}: {e}")
    #     try:
    #         adata, doublets_removed_DoubletDetection = doublet_detection(adata)
    #     except:
    #         logging.error(f"Error processing doublets detect {sample_name}: {e}")
    #         doublets_removed_solo=0
    #         doublets_removed_DoubletDetection = 0
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

    end_time     = datetime.now()
    time_taken   = (end_time - start_time).total_seconds()
    # Calculate hours and minutes for the processing time
    hours        = int(time_taken // 3600)
    minutes      = int((time_taken % 3600) // 60)

    # Prepare data for the CSV
    data_for_csv = [sample_name, 
                    num_of_original_cells,
                    num_of_original_genes,
                    num_cells_poor_removed, 
                    num_genes_poor_removed, 
                    num_cells_mito_removed,
                    num_genes_mito_removed, 
                    ribo_genes_removed, 
                    doublets_removed_solo,
                    doublets_removed_DoubletDetection,
                    num_of_output_cells,
                    num_of_output_genes,
                    Sample_Status, 
                    Sample_Type,
                    Sample_Tissue_Type, 
                    Sample_Gender, 
                    Sample_Age, 
                    Sample_Race, 
                    hours,
                    minutes]

    # Check if the CSV file exists and write headers if it doesn't
    if not os.path.isfile(CVS_FILE_PATH):
        print(f"Creating a new CSV file for tracking sample processing...\n{CVS_FILE_PATH}")
        with open(CVS_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    # Append the data to the CSV file
    with open(CVS_FILE_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_for_csv)

    return adata

# ------------------------------------------
def read_and_process_data(data_location: Path) -> AnnData:
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
    # Define the percentile thresholds for filtering
    min_genes_percentile = 2  # 2% percentile for genes
    max_genes_percentile = 98  # 98% percentile for genes
    min_cells_percentile = 2  # 2% percentile for cells
    max_cells_percentile = 98  # 98% percentile for cells    

    adata         = sc.read_10x_h5(data_location)

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    rsc.get.anndata_to_GPU(adata)
    print('Calculate QC metrics')
    rsc.pp.flag_gene_family(adata, gene_family_name="MT", gene_family_prefix="MT")
    rsc.pp.flag_gene_family(adata, gene_family_name="RIBO", gene_family_prefix="RPS")
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["MT", "RIBO"])
    print('Calculate thresholds for genes')
    min_genes_threshold = adata.var['n_cells_by_counts'].quantile(min_genes_percentile / 100)
    max_genes_threshold = adata.var['n_cells_by_counts'].quantile(max_genes_percentile / 100)

    print('Calculate thresholds for cells')
    min_cells_threshold = adata.obs['n_genes_by_counts'].quantile(min_cells_percentile / 100)
    max_cells_threshold = adata.obs['n_genes_by_counts'].quantile(max_cells_percentile / 100)

    print('Apply filtering')
    rsc.pp.filter_cells(adata, qc_var='n_genes_by_counts', min_count=min_cells_threshold, max_count=max_cells_threshold,verbose=True)
    rsc.pp.filter_genes(adata, qc_var='n_cells_by_counts', min_count=min_genes_threshold, max_count=max_genes_threshold,verbose=True)

    adata = adata[adata.obs["pct_counts_MT"] < 20].copy()

    try:
        rsc.pp.scrublet(adata)
        adata = adata[~adata.obs['predicted_doublets']]
        del adata.obs['predicted_doublets']
        del adata.obs['doublet_score']
        adata.uns.clear()

    except Exception as ex:
        print(f"Shape of adata.X: {adata.X.shape}")

    gc.collect()
    rsc.get.anndata_to_CPU(adata)

    return adata
# ------------------------------------------       
def produce_cell_map(adata:AnnData,adata_ref:AnnData) -> dict:
    """
    Concatenates two AnnData objects, preprocesses the combined dataset, and applies the scVI and SCANVI models 
    to produce cell type predictions for each cell. The function ultimately returns a mapping of cell barcodes to 
    their predicted cell types.

    Parameters:
    - adata (AnnData): The main AnnData object containing single-cell data to be analyzed.
    - adata_ref (AnnData): An AnnData object to be concatenated with `adata` for analysis.

    Returns:
    - dict: A dictionary mapping cell barcodes to their predicted cell types.

    Note:
    - The function modifies `adata` in place by concatenating `adata_ref`, normalizing data, identifying highly 
      variable genes, and setting up and training the scVI and SCANVI models.
    - NORMALIZATION_SCALE, COLUMN_OF_INTEREST, and LABEL_PREDICTION are constants that need to be defined externally.
    """   
    # Label Concatenation and preprocessing 
    adata.var_names_make_unique()
    adata = adata.concatenate(adata_ref)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)    
    adata.raw = adata.copy()

    # Singlularity error with Seurat_v3, change span to 0.6 and increase 
    # https://github.com/scverse/scanpy/issues/1504
    sc.pp.highly_variable_genes(adata, flavor = 'seurat_v3', n_top_genes=2000,
                                layer = "counts", batch_key="batch", subset = True,span=0.6)
    
    # scVI model setup and training
    scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key='batch')
    vae=scvi.model.SCVI(adata)
    vae.train()

    # SCANVI model setup and training for label prediction
    adata.obs[COLUMN_OF_INTEREST] = adata.obs[COLUMN_OF_INTEREST].cat.add_categories('Unknown')
    adata.obs                     = adata.obs.fillna(value = {COLUMN_OF_INTEREST: 'Unknown'})
    lvae = scvi.model.SCANVI.from_scvi_model(vae, 
                                             adata = adata, 
                                             unlabeled_category = 'Unknown',
                                             labels_key = COLUMN_OF_INTEREST)

    # Cell type prediction and barcode cleaning
    lvae.train(max_epochs=20, n_samples_per_label=100)
    adata.obs[LABEL_PREDICTION] = lvae.predict(adata)
    adata.obs['bc2']            = adata.obs.index.map(lambda x: x[:-2])
    cell_mapper                 = dict(zip(adata.obs.bc2, adata.obs[LABEL_PREDICTION]))

    return cell_mapper
# ------------------------------------------
def clean_barcode(barcode: str) -> str:
    """
    Cleans a given barcode string by matching it against a predefined regular expression pattern. If a match is 
    found, the matched sequence is returned; otherwise, the original barcode is returned.

    Parameters:
    - barcode (str): The barcode string to be cleaned.

    Returns:
    - str: The cleaned barcode if a match is found; otherwise, the original barcode.

    Note:
    - The function uses a regular expression pattern to identify valid barcodes consisting of 16 characters from 
      the set [ACTG]. Adjust the REGEX pattern as needed based on the expected barcode format.
    """    
    REGEX = r'[ACTG]{16}'
    match = re.search(REGEX, barcode)
    if match:
        return match.group(0)  # Return the matched group
    else:
        return barcode  # Return the original barcode if no match was found (or adjust as needed)
# ------------------------------------------   
def find_subdirectories(root_path: Path):
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

    sample_names = []

    # Initialize list to store paths of the target files
    file_paths = []

    # Traverse through the directories
    for sample_dir in root_path.iterdir():
        if sample_dir.is_dir():  # Check if it's a directory
            sample_name = sample_dir.name  # Extract sample name (xxxx)
            target_file = sample_dir / "cellranger_output" / "outs" / "filtered_feature_bc_matrix.h5"
            if target_file.exists():
                file_paths.append(target_file)
                sample_names.append(sample_name)

    return file_paths
# ------------------------------------------  
def remove_matching_elements(large_list:List[str]) -> List[str]:
    
    metadata_df  = pd.read_csv(CVS_FILE_PATH)
    small_set    = list(set(metadata_df.sample_name.tolist()))
    print(f"Number of samples executed: {len(small_set)}")
    
    # Use list comprehension to filter out matching elements
    filtered_list = [element for element in large_list if element not in small_set]
    print(f"Number of samples to execute: {len(filtered_list)}")
    return filtered_list
# ------------------------------------------  
def clean_anndata(adata):
    """
    Drop duplicates in adata.obs_names, keep the first instance, and create a new cleaned AnnData object.

    Parameters:
    - adata: AnnData object to process.

    Returns:
    - cleaned_adata: A new AnnData object with duplicates removed and re-indexed.
    """
    print("Checking for duplicates in adata.obs_names...")
    if not adata.obs.index.has_duplicates:
        print("No duplicates found. Returning original AnnData object.")
        return adata.copy()  # Return a copy to maintain immutability

    print("Step 1: Identify unique indices")
    _, unique_indices = np.unique(adata.obs_names, return_index=True)
    unique_indices = np.sort(unique_indices)  # Ensure proper order

    print("Step 2: Create separate variables")
    cleaned_obs_names = adata.obs_names[unique_indices]
    cleaned_obs = adata.obs.iloc[unique_indices]
    if isinstance(adata.X, csr_matrix):
        cleaned_X = adata.X[unique_indices, :]
    else:
        cleaned_X = adata.X[unique_indices, :]

    print("Step 3: Create a new AnnData object")
    cleaned_adata = AnnData(
        X=cleaned_X,
        obs=cleaned_obs,
        var=adata.var,
        uns=adata.uns,
        obsm=adata.obsm,
        varm=adata.varm,
        layers=adata.layers,
    )

    # Step 4: Set cleaned obs_names
    cleaned_adata.obs_names = cleaned_obs_names

    print("Cleaned AnnData object created successfully.")
    return cleaned_adata


# ------------------------------------------  
if __name__ == "__main__":
    # ------------------------------------
    # List of renamed samples

    rmm.reinitialize(
        managed_memory=False,  # Allows oversubscription
        pool_allocator=False,  # default is False
        devices=0,  # GPU device IDs to register. By default registers only GPU 0.
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)

    # Set the normalization scale and seed for reproducibility    
    # ov.utils.ov_plot_set()
    # ov.settings.gpu_init()
    # scvi.settings.seed = 0
    sc.set_figure_params(dpi=100)
    # torch.set_float32_matmul_precision("high")    
    # print(f"Last run with scvi-tools version: {scvi.__version__}")

    # Set default figure parameters and higher precision for float32 matrix multiplication
    sc.settings.n_jobs=8
    # sc.set_figure_params()
    sc.set_figure_params(figsize=(4, 4))

    # Initialize 
    adata_array:  List[sc.AnnData] = []

    # sample_names: List[str] = list(SAMPLES_OF_INTEREST_DIR.rglob("filtered_feature_bc_matrix.h5"))
    sample_names       = find_subdirectories(SAMPLES_OF_INTEREST_DIR)
    completed_samples              = 0
    total_samples                  = len(sample_names)
    
    # Comment line before after done testing!!!
    # sample_names = sample_names[0:3]
    # adata=read_and_process_data(sample_names[20])
    # -------------------------------         

    # Process samples concurrently to improve efficiency
    print('Spool parallel pre-processing')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sample_name in sample_names:
            # logging.info("----------------------------------------------")
            # logging.info(f"Starting processing for sample: {sample_name} ({len(futures) + 1}/{total_samples})")
            future = executor.submit(read_and_process_data, sample_name)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed_samples += 1
            if result is not None:
                result.obs['batch']=str(completed_samples)
                adata_array.append(result)
                logging.info("----------------------------------------------")
                logging.info(f"Completed processing sample {completed_samples} of {total_samples}")
                logging.info("----------------------------------------------")
            else:
                logging.error(f"A sample failed to process or returned no data ({completed_samples}/{total_samples})")

    # Concatenate results from all samples into a single AnnData object
    adata=sc.concat(adata_array,merge='same')
    # adata.X=adata.X.astype(np.int64)
    adata=clean_anndata(adata)
    # --------------------------------------------------------
    # Save the combined dataset to a file, overwriting if it already exists
    concatenated_results_file = OUTPUT_DIR / 'rapids_single_cell_concatenated_results_file.h5ad'
    if concatenated_results_file.exists():
        concatenated_results_file.unlink()
    # adata.write_h5ad('/media/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/0a0a_Results/concatenated_results_file.h5ad')
    adata.write_h5ad(concatenated_results_file)
    adata=sc.read_h5ad(concatenated_results_file)
    # --------------------------------------------------------
    print('Normalizing')
    rsc.get.anndata_to_GPU(adata)
    adata.layers["counts"] = adata.X.copy()    
    rsc.pp.normalize_total(adata, target_sum=1e4)
    rsc.pp.log1p(adata)
    gc.collect()
    rsc.get.anndata_to_CPU(adata)
    # --------------------------------------------------------
    print('highly_variable_genes')
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", batch_key= "batch",layer="counts")
    adata.raw = adata
    # rsc.pp.filter_highly_variable(adata)
    adata = adata[:, adata.var["highly_variable"]]
    rsc.pp.regress_out(adata, keys=["total_counts", "pct_counts_MT"])
    rsc.pp.scale(adata, max_value=10)
    # --------------------------------------------------------
    print('pca')
    rsc.tl.pca(adata, n_comps=100)
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=100)

    rsc.pp.harmony_integrate(adata, key="batch")

    # Save the combined dataset to a file, overwriting if it already exists
    concatenated_results_file = OUTPUT_DIR / 'rapids_single_cell_integrated_results_file.h5ad'
    if concatenated_results_file.exists():
        concatenated_results_file.unlink()
    # adata.write_h5ad('/media/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/0a0a_Results/concatenated_results_file.h5ad')
    adata.write_h5ad(concatenated_results_file)
    adata=sc.read_h5ad(concatenated_results_file)       

    rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    rsc.tl.umap(adata)

    rsc.tl.louvain(adata, resolution=0.6)
    rsc.tl.leiden(adata, resolution=0.6)
    sc.pl.umap(adata, color=["louvain", "leiden"], legend_loc="on data")
    sc.pl.umap(adata, color=["batch"])

    gc.collect()
    rsc.get.anndata_to_CPU(adata)

    # Save the combined dataset to a file, overwriting if it already exists
    concatenated_results_file = OUTPUT_DIR / 'rapids_single_cell_clustered_results_file.h5ad'
    if concatenated_results_file.exists():
        concatenated_results_file.unlink()
    # adata.write_h5ad('/media/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/0a0a_Results/concatenated_results_file.h5ad')
    adata.write_h5ad(concatenated_results_file)
    adata=sc.read_h5ad(concatenated_results_file)

    print('Done')


