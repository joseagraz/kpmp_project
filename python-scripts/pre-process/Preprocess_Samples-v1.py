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
# OUTPUT_FILE_NAME        = f'_sample-filtered-and-labeled.{OUTPUT_FILE_EXTENSION}'
OUTPUT_FILE_NAME        = f'_sample-filtered-and-labeled_celbender.{OUTPUT_FILE_EXTENSION}'
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
def detect_doublets(adata: AnnData, expected_doublet_rate: float = 0.06) -> AnnData:
    """
    Detects doublets in single-cell RNA-seq data using the Scrublet tool.

    This function applies the Scrublet algorithm to identify doublet (dual-cell) signatures within the provided AnnData object. It calculates doublet scores for each cell, predicts doublets, and annotates the input AnnData object with these predictions.

    Parameters:
    - adata (AnnData): The AnnData object containing single-cell RNA-seq data to be analyzed.
    - expected_doublet_rate (float, optional): The expected doublet rate in the dataset. It is used by Scrublet to adjust its sensitivity. Default is 0.06.

    Returns:
    - AnnData: The input AnnData object is returned after appending the 'doublet_scores' and 'predicted_doublets' to its `.obs` attribute. The 'doublet_scores' column contains the computed doublet scores for each cell, while the 'predicted_doublets' column contains boolean values indicating predicted doublet status.

    Note:
    - It's important to have Scrublet installed in your environment to use this function (`pip install scrublet`).
    - The function modifies the input AnnData object in-place by adding new columns to its `.obs` dataframe.
    """
    # Initialize a Scrublet object with the expected doublet rate
    scrub = scr.Scrublet(adata.X, expected_doublet_rate=expected_doublet_rate)

    # Compute doublet scores and predict doublets
    _, predicted_doublets = scrub.scrub_doublets()

    # Annotate the AnnData object with doublet scores and predictions
    # adata.obs['doublet_scores']     = doublet_scores
    adata.obs[DOUBLETS_PREDICTION] = predicted_doublets
    number_of_doublets             = adata.obs[DOUBLETS_PREDICTION].value_counts()[True]
    
    adata = adata[adata.obs[DOUBLETS_PREDICTION] == False]
    adata.obs.drop(DOUBLETS_PREDICTION, axis=1, inplace=True)    

    return adata, number_of_doublets
# ------------------------------------------
def mitochondrial_genes_removal(adata: AnnData) -> Tuple[AnnData, int, int]:
    """
    Removes cells based on mitochondrial gene content and gene count quantiles from an AnnData object.

    This function annotates mitochondrial genes, calculates quality control metrics, and filters
    cells that have an unusually high or low number of genes based on specified quantiles, as well
    as cells with a high percentage of mitochondrial genes. It modifies the input AnnData object in place,
    adding annotations for mitochondrial gene filtering and quantile-based gene count filtering.

    Parameters:
    - adata (AnnData): The single-cell data to be filtered. It is modified in place.

    Returns:
    - Tuple[AnnData, int]: A tuple containing the filtered AnnData object and the number of cells filtered out.

    Raises:
    - ValueError: If `adata` is not an instance of AnnData.

    """
    MITO_REGEX       = '(?i)^MT-'
    num_genes_before = adata.n_vars
    num_cells_before = adata.n_obs

    # Annotate mitochondrial genes
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX, case=False, regex=True)

    # Calculate QC metrics for mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Determine upper and lower limits for gene counts
    upper_lim = int(np.quantile(adata.obs.n_genes_by_counts.values, UPPER_QUANTILE))
    lower_lim = int(np.quantile(adata.obs.n_genes_by_counts.values, LOWER_QUANTILE))
    print(f'Gene count quantile limits: {lower_lim} to {upper_lim}')

    # Filter cells based on gene count quantiles    
    adata.obs[MT_QUANTILE_TITLE]     = (adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)
    # Keep True, remove False
    number_of_mito_quantiles_removal = adata.obs[MT_QUANTILE_TITLE].value_counts()[False]
    adata = adata[adata.obs[MT_QUANTILE_TITLE] == True]
    adata.obs.drop(MT_QUANTILE_TITLE, axis=1, inplace=True)

    # Filter cells based on mitochondrial gene count percentage
    # Keep True, remove False
    adata.obs[MITO_TITLE]             = adata.obs.pct_counts_mt < MT_COUNT_LIMIT
    if len(adata.obs[MITO_TITLE].value_counts())>1:
        number_of_mito_removal_percentage = adata.obs[MITO_TITLE].value_counts()[False]
    adata = adata[adata.obs[MITO_TITLE] == True]
    adata.obs.drop(MITO_TITLE, axis=1, inplace=True)

    number_of_cells_removed = num_cells_before - adata.n_obs
    number_of_genes_removed = num_genes_before - adata.n_vars
    # Calculate the number of filtered cells
    print(f'Number of input genes:     {num_genes_before}')
    print(f'Number of genes removed:   {number_of_genes_removed}')
    print(f'Number of genes remaining: {adata.n_vars}')
    print(f'Number of input cells:     {num_cells_before}')
    print(f'Number of cells removed:   {number_of_cells_removed}')
    print(f'Number of cells remaining: {adata.n_obs}')

    return adata, number_of_cells_removed,number_of_genes_removed
# ------------------------------------------
def ribo_genes_removal(adata_original: AnnData) -> tuple[AnnData, int]:
    """
    Removes ribosomal genes from the dataset and filters cells based on gene count quantiles.

    This function copies the original AnnData object, identifies ribosomal genes,
    and filters out cells with extreme gene counts based on the specified upper and
    lower quantiles. It also filters cells based on the percentage of ribosomal genes.

    Parameters:
    - adata_original: AnnData
        The original single-cell dataset contained in an AnnData object.

    Returns:
    - tuple[AnnData, int]
        A tuple containing the filtered AnnData object and the number of cells filtered.
    """    
    # Copy the input dataset to avoid modifying the original
    adata = adata_original.copy()

    # Initial counts of genes    
    num_genes_before = adata.n_vars

    # Load ribosomal genes from a predefined list
    ribo_genes = pd.read_table(RIBO_LOOKUP_FILE_PATH, skiprows=2, header=None)    
    adata      = adata[:, ~adata.var_names.isin(ribo_genes[0].values)]    
    
    num_gene_filtered = num_genes_before - adata.n_vars

    return adata, num_gene_filtered

# ------------------------------------------
def doublet_removal_solo(adata_original: AnnData) -> tuple[AnnData, int]:
    """
    Removes doublet cells from an AnnData object using the scVI and SOLO models for doublet prediction.

    Parameters:
    ----------
    adata_original : AnnData
        The original AnnData object containing the single-cell RNA sequencing data.

    Returns:
    -------
    tuple[AnnData, int]
        A tuple containing the AnnData object after doublet removal (where doublets are flagged)
        and the number of detected doublets.
    
    Details:
    --------
    This function first identifies highly variable genes, then uses scVI to model the data,
    and finally applies the SOLO model to predict doublets. Cells predicted as doublets are flagged
    in the returned AnnData object, allowing users to decide on further action (e.g., removal).
    """
    # Copy the original data to preserve it
    adata          = adata_original.copy()
    adata_doublets = adata_original.copy()

    # Ensure the counts are stored in a layer for scVI processing
    adata.layers['counts'] = adata.X.copy()

    # Identify highly variable genes without subsetting the original data
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        subset=False,  # Keep all genes in the object, mark highly variable ones
        flavor="seurat_v3",
        layer="counts"
    )
        
    # Model to predict doublets using scVI and SOLO  
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    doublets_model = scvi.model.SCVI(adata) 
    doublets_model.train()
    
    # Pass the scVI model to the SOLO model for doublet detection    
    solo = scvi.external.SOLO.from_scvi_model(doublets_model)
    solo.train()
    
    # Convert doublet predictions to a DataFrame
    df = solo.predict()
    df[DOUBLETS_PREDICTION] = solo.predict(soft=False)

    # Fine-tune doublet labeling by deleting records with a significant difference in prediction scores
    df['dif'] = df.doublet - df.singlet
    doublets  = df[(df.prediction == DOUBLETS) & (df.dif > 1)]
    adata_doublets.obs[DOUBLETS_PREDICTION] = adata_doublets.obs.index.isin(doublets.index)

    number_of_doublets = adata_doublets.obs[DOUBLETS_PREDICTION].value_counts()[True]    
    adata_doublets     = adata_doublets[adata_doublets.obs[DOUBLETS_PREDICTION] == False]
    adata_doublets.obs.drop(DOUBLETS_PREDICTION, axis=1, inplace=True) 

    print(f"Number of doublets detected: {number_of_doublets}")   

    return adata_doublets, number_of_doublets

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
def doublet_detection(adata: AnnData) -> Tuple[AnnData, int]:
    """
    Detects doublets in single-cell RNA sequencing data using the DoubletDetection library.
    
    This function processes an AnnData object containing single-cell RNA sequencing data,
    identifies potential doublets using a BoostClassifier from the DoubletDetection package,
    and filters out detected doublets from the data. It returns a cleaned AnnData object
    without the detected doublets and the number of doublets identified.
    
    Parameters:
    - adata (AnnData): An AnnData object containing the single-cell RNA sequencing data.
    
    Returns:
    - Tuple[AnnData, int]: A tuple where the first element is the cleaned AnnData object
      (without detected doublets) and the second element is the number of doublets identified.
    
    Example:
    >>> cleaned_adata, num_doublets = doublet_detection(adata)
    
    Note:
    This function requires the DoubletDetection package and an AnnData object as input.
    The AnnData object should have unique variable names. For more information on how to
    use the DoubletDetection package, visit:
    https://doubletdetection.readthedocs.io/en/latest/tutorial.html
    """
    THRESHOLD = 0.5
    number_of_doublets=0
    adata.var_names_make_unique()
    clf = doubletdetection.BoostClassifier(
        n_iters=10,
        clustering_algorithm="louvain",
        standard_scaling=False,
        pseudocount=0.1,
        n_jobs=-1,
    )
    doublets      = clf.fit(adata.X).predict(p_thresh=1e-16, voter_thresh=THRESHOLD)
    # doublet_score = clf.doublet_score()    

    adata.obs[DOUBLETS_PREDICTION] = doublets>THRESHOLD
    doublet_counts = adata.obs[DOUBLETS_PREDICTION].value_counts()
    if True in doublet_counts.index:
        number_of_doublets = doublet_counts[True]
    adata = adata[adata.obs[DOUBLETS_PREDICTION] == False]
    adata.obs.drop(DOUBLETS_PREDICTION, axis=1, inplace=True)    

    print(f"Number of doublets detected: {number_of_doublets}")

    return adata, number_of_doublets
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
    adata, \
        num_cells_poor_removed, \
        num_genes_poor_removed  = poor_cell_and_gene_removal(adata)
    adata, \
        num_cells_mito_removed, \
        num_genes_mito_removed  = mitochondrial_genes_removal(adata)
    adata, \
        ribo_genes_removed      = ribo_genes_removal(adata)    
    # ---------------------------------------
    # Doublet detection
    print('Doublet detection using scVI SOLO')
    try:
        adata, doublets_removed_solo     = doublet_removal_solo(adata)    
    except Exception as e:
        logging.error(f"Error processing sample {sample_name}: {e}")
        adata, doublets_removed_DoubletDetection = doublet_detection(adata)
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
    # if not os.path.isfile(CVS_FILE_PATH):
    #     print(f"Creating a new CSV file for tracking sample processing...\n{CVS_FILE_PATH}")
    #     with open(CVS_FILE_PATH, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(headers)

    # # Append the data to the CSV file
    # with open(CVS_FILE_PATH, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(data_for_csv)

    return adata

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

    # data_location = data_location.parent / 'raw_feature_bc_matrix_cellbender_denoised_filtered.h5'

    adata         = sc.read_10x_h5(data_location)  
    adata.X       = adata.X.astype(np.int64)           
    adata         = process_sample(adata, sample_name)

    ref_data      = sc.read_h5ad(ATLAS_FILE_NAME)
    ref_data.X    = ref_data.X.astype(np.int64)
    ref_data,_,_  = poor_cell_and_gene_removal(ref_data)  
    ref_data.obs  = ref_data.obs[[COLUMN_OF_INTEREST]]  
    ref_data.var.rename(columns={'features-0': 'features'}, inplace=True)
    ref_data.var  = ref_data.var[['features']]    
    
    cell_mapper   = produce_cell_map(adata,ref_data)
    adata.obs[COLUMN_OF_INTEREST] = adata.obs.index.map(cell_mapper)

    output_file = Path(str(data_location)+OUTPUT_FILE_NAME)

    # Check if the file exists and then delete it
    if output_file.exists():
        output_file.unlink()
    adata.write_h5ad(output_file, compression='gzip')

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
def remove_matching_elements(large_list:List[str]) -> List[str]:
    
    metadata_df  = pd.read_csv(CVS_FILE_PATH)
    small_set    = list(set(metadata_df.sample_name.tolist()))
    print(f"Number of samples executed: {len(small_set)}")
    
    # Use list comprehension to filter out matching elements
    filtered_list = [element for element in large_list if element not in small_set]
    print(f"Number of samples to execute: {len(filtered_list)}")
    return filtered_list

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

    # Remove existing sample tracker log, if any
    # if CVS_FILE_PATH.exists():
    #     CVS_FILE_PATH.unlink()

    # -------------------------------
    # Comment line before after done testing!!!
    # sample_names=sample_names[1:20]   
    # adata=read_and_process_data('25269e2d-3370-4893-bc28-4c8cd871e730', ROOT_PATH, DATA_DIR, SOURCE_DATA_PATH)
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


