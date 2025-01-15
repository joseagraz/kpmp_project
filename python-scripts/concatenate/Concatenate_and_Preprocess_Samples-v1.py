# This 
# https://hub.docker.com/r/scverse/scvi-tools
# docker pull scverse/scvi-tools:py3.11-cu12-base
from typing import Tuple
from tqdm import tqdm
import scanpy as sc
import pandas as pd
from matplotlib.pyplot import rc_context
import numpy as np
from pathlib import Path
import scvi
import logging
import concurrent.futures
import csv
import re
import torch
import tempfile
from collections import Counter
import seaborn as sns
import time
from scipy.sparse import csr_matrix
from anndata import AnnData
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
# ------------------------------------------
ROOT_PATH               = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR       = Path('Supporting_Files')
DATA_DIR                = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
RESULTS_DIR             = 'Results'
CELLRANGER_PATH         = Path('cellranger_output/outs/filtered_feature_bc_matrix')
CVS_FILE_PATH           = ROOT_PATH / SUPPORT_FILES_DIR / 'list_of_removals.csv'
METADATA_FILE_PATH      = ROOT_PATH / SUPPORT_FILES_DIR / 'metadata.csv'
SOURCE_METADATA_PATH    = ROOT_PATH / SUPPORT_FILES_DIR / 'source_metadata.csv'
RIBO_LOOKUP_FILE_PATH   = ROOT_PATH / SUPPORT_FILES_DIR / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
SAMPLES_OF_INTEREST_DIR = ROOT_PATH / DATA_DIR
SCVI_LATENT_KEY         = "X_scVI"
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------
def mitochondrial_genes_removal(adata):
    UPPER_QUANTILE   = 0.98
    LOWER_QUANTILE   = 0.02
    MT_COUNT_LIMIT   = 40

    # Count the number of cells before filtering
    num_cells_before = adata.n_obs
    num_genes_before = adata.n_vars    

    adata.var['mt'] = adata.var_names.str.contains('^MT-', 
                                                   case  = False, 
                                                   regex = True)  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, 
                               qc_vars     = ['mt'], 
                               percent_top = None, 
                               log1p       = False, 
                               inplace     = True)    

    adata.var.sort_values('n_cells_by_counts')
    #instead of picking subjectively, you can use quanitle
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, UPPER_QUANTILE)
    lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, LOWER_QUANTILE)
    print(f'{lower_lim} to {upper_lim}')

    # adata = adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]    
    quantile_obs_title = f'{UPPER_QUANTILE}%_quantiles_{LOWER_QUANTILE}%'
    adata.obs[quantile_obs_title]  = (adata.obs.n_genes_by_counts < upper_lim) & \
                                     (adata.obs.n_genes_by_counts > lower_lim)

    # adata = adata[adata.obs.pct_counts_mt   < MT_COUNT_LIMIT]
    mito_obs_title = f'mitochondrial_genes_<{MT_COUNT_LIMIT}_pct_count'
    adata.obs[mito_obs_title] = adata.obs.pct_counts_mt < MT_COUNT_LIMIT

    # Count the number of cells after filtering
    num_cells_after = adata.n_obs

    # Calculate the number of filtered cells
    num_filtered = num_cells_before - num_cells_after

    return adata,num_filtered     
# ------------------------------------------
def ribo_genes_removal(adata_original):
    UPPER_QUANTILE       = 0.98
    LOWER_QUANTILE       = 0.02
    RIBO_COUNT_LIMIT     = 2
    RIBOS0ME_GENES_TITLE = 'ribo'
    
    adata = adata_original.copy()

    # Count the number of cells before filtering
    num_cells_before = adata.n_obs
    num_genes_before = adata.n_vars

    ribo_genes        = pd.read_table(RIBO_LOOKUP_FILE_PATH, skiprows=2, header = None)
    adata.var[RIBOS0ME_GENES_TITLE] = adata.var_names.isin(ribo_genes[0].values)
    sc.pp.calculate_qc_metrics(adata, 
                               qc_vars     = [RIBOS0ME_GENES_TITLE], 
                               percent_top = None, 
                               log1p       = False, 
                               inplace     = True)    

    # #instead of picking subjectively, you can use quanitle
    adata.var.sort_values('n_cells_by_counts')
    adata.obs.sort_values('n_genes_by_counts')
    # #instead of picking subjectively, you can use quanitle
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, UPPER_QUANTILE)
    lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, LOWER_QUANTILE)
    print(f'{lower_lim} to {upper_lim}')

    # adata = adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]      
    ribosome_genes_title            = f'pct_counts_{RIBOS0ME_GENES_TITLE}'
    adata.obs[f'{RIBOS0ME_GENES_TITLE}_<{RIBO_COUNT_LIMIT}_pct_count'] = adata.obs[ribosome_genes_title] < RIBO_COUNT_LIMIT
    # adata = adata[adata.obs['pct_counts_ribosome_genes'] < RIBO_COUNT_LIMIT]
    
    # Count the number of cells after filtering
    num_cells_after = adata.n_obs

    # Calculate the number of filtered cells
    num_filtered = num_cells_before - num_cells_after

    return adata,num_filtered    

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
    MINIMUM_GENES           = 200
    MINIMUM_CELLS           = 10
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
def doublet_removal(adata_original):
    DOUBLETS_PREDICTION = 'prediction'
    DOUBLETS            = 'doublet'
    SINGLET             = 'singlet'

    adata          = adata_original.copy()
    adata_doublets = adata_original.copy()
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes = 3000,
        subset      = False,
        flavor      = "seurat_v3"
    )
    # Model to predict doublets using scVI
    scvi.model.SCVI.setup_anndata(adata)
    doublets_model = scvi.model.SCVI(adata, use_batch_norm=False) 
    doublets_model.train()
    # Pass the scVI model to SOLO model
    solo           = scvi.external.SOLO.from_scvi_model(doublets_model)
    solo.train()
    #
    # Convert doublet preditions to dataframe
    df                      = solo.predict()
    df[DOUBLETS_PREDICTION] = solo.predict(soft = False)

    df.groupby('prediction').count()

    # Fine tune doublet labeling, by deleting dif>1 in distribution
    df['dif'] = df.doublet - df.singlet
    # Plot doublet label distribution
    # sns.displot(df[df.prediction == 'doublet'], x = 'dif')
    doublets                      = df[(df.prediction == DOUBLETS) & (df.dif >  1)]
    adata_doublets.obs[DOUBLETS]  = adata_doublets.obs.index.isin(doublets.index)
    # adata_doublets                = adata_doublets[~adata_doublets.obs.doublet]
    number_of_doublets            = len(doublets)

    return adata_doublets,number_of_doublets
# ------------------------------------------
def doublet_removal_scrublet(adata):

    DATA_LOCATION = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'doublet_removal_scrublet_histograms'
    DATA_LOCATION.mkdir(exist_ok=True)

    counts_matrix = adata.X.toarray()

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=0.06)
    n_samples = adata.n_obs
    n_features = adata.n_vars
    n_prin_comps = min(10, n_samples, n_features)

    doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=10, 
                                                            min_cells=10, 
                                                            min_gene_variability_pctl=85, 
                                                            n_prin_comps=30)
    # Use adaptive thresholding
    predicted_doublets = scrub.call_doublets(threshold=None)

    # Add results to AnnData
    adata.obs['doublet_scores'] = doublet_scores
    adata.obs['predicted_doublets'] = predicted_doublets.astype(bool)
    adata.uns['doublet_threshold'] = scrub.threshold_

    # Filter out predicted doublets and preserve raw data
    adata_filtered = adata[~adata.obs['predicted_doublets']].copy()  
    # Compare original and filtered data
    print(f"Original dataset: {adata.n_obs} cells")
    print(f"Filtered dataset: {adata_filtered.n_obs} cells")
    print(f"Number of doublets removed: {adata.n_obs - adata_filtered.n_obs}")

    # Plotting the differences
    # 1. Histogram of doublet scores
    # plt.figure(figsize=(8, 5))
    # plt.hist(adata.obs['doublet_scores'], bins=50, alpha=0.7, label='Original')
    # plt.hist(adata_filtered.obs['doublet_scores'], bins=50, alpha=0.7, label='Filtered')
    # plt.xlabel('Doublet Scores')
    # plt.ylabel('Cell Count')
    # plt.legend()
    # plt.title('Doublet Score Distribution')
    # plt.show()   
    # plt.savefig(DATA_LOCATION / f'{adata.obs.Sample_Name.iloc[0]}.png')

    return adata_filtered
# ------------------------------------------
def convert_sample_age(age_str):
    # If the age_str is None or indicates missing data, return 0
    if pd.isna(age_str) or age_str == 'None':
        return 0

    # Use regex to find the starting digits
    match = re.match(r"(\d+)-\d+ Years", age_str)
    if match:
        return int(match.group(1))
    
    return 0
# ------------------------------------------
def get_sample_details(sample_name):


    df = pd.read_csv(SOURCE_METADATA_PATH)
    # Filter the dataframe to find the row matching the given sample_name
    sample_row = df[df['sample_name'] == sample_name]

    # If the sample is found, return the relevant details
    if not sample_row.empty:
        sample_details = sample_row[['Sample_Type', 'Sample_Tissue_Type', 'Sample_Sex', 'Sample_Age', 'Sample_Race']].iloc[0]
        sample_details = sample_details.fillna('None')
        # Convert age string to integer
        sample_details['Sample_Age'] = convert_sample_age(sample_details['Sample_Age'])

        return sample_details.tolist()
    else:
        return "Sample not found."
# ------------------------------------------
def process_sample(sample_name):
    num_cells_removed        = 0
    num_genes_removed        = 0
    mito_genes_removed_count = 0
    ribo_genes_removed_count = 0
    doublets_removed_count   = 0
    # Define the headers for the CSV file
    headers = [ "sample_name", 
                "doublets_labeled", 
                "cells<200_labeled", 
                "genes<3_labeled",      
                "mitochondrial_labeled", 
                "ribo_labeled"]    
    # ------------------------------    
    # logging.info(f"processing sample: {sample_name.strip()}")
    data_location = ROOT_PATH / Path(DATA_DIR) / Path(sample_name.strip()) / CELLRANGER_PATH

    Sample_Type,Sample_Tissue_Type,Sample_Sex,Sample_Age,Sample_Race = get_sample_details(sample_name)

    adata = sc.read_10x_h5(data_location.with_suffix('.h5'))

    adata.obs['Sample_Name']        = sample_name.strip()
    adata.obs['Sample_Type']        = Sample_Type
    adata.obs['Sample_Tissue_Type'] = Sample_Tissue_Type
    adata.obs['Sample_Sex']         = Sample_Sex
    adata.obs['Sample_Age']         = Sample_Age
    adata.obs['Sample_Race']        = Sample_Race
    
    # Force all names are upper case
    adata.var_names = [gene_name.upper() for gene_name in adata.var_names]

    adata.var_names_make_unique()

    # adata, doublets_removed_count = doublet_removal(adata)   
    # adata = doublet_removal_scrublet(adata)    
    doublets_removed_count=0      

    adata,num_cells_removed,num_genes_removed = poor_cell_and_gene_removal(adata)

    adata,mito_genes_removed_count            = mitochondrial_genes_removal(adata)

    adata, ribo_genes_removed_count           = ribo_genes_removal(adata)
    
    # Prepare data for CSV
    data_for_csv = [sample_name,
                    doublets_removed_count,
                    num_cells_removed,
                    num_genes_removed,
                    mito_genes_removed_count,
                    ribo_genes_removed_count
                    ]
    
    # Check if the CSV file exists and write headers if it doesn't
    if not METADATA_FILE_PATH.exists():
        with open(METADATA_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    # Append the data to the CSV file
    with open(METADATA_FILE_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_for_csv)                   
    
    output_file = Path(str(data_location)+'_preprocessing-filtered').with_suffix('.h5ad')
    adata.X     = csr_matrix(adata.X)
    # Check if the file exists and then delete it
    if output_file.exists():
        output_file.unlink()
    adata.write_h5ad(output_file,compression='gzip')

    return adata
# ------------------------------------------   
def find_subdirectories(root_path):

    target_path = 'cellranger_output'
    sample_names = []

    for path in root_path.rglob(target_path):
        # Assuming sample name is the directory immediately before 'cellranger_output'
        sample_name = path.parts[-2]
        sample_names.append(sample_name)

    return sample_names 
# ------------------------------------------   
if __name__ == "__main__":    
    
    scvi.settings.seed  = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    save_dir = tempfile.TemporaryDirectory()

    # Remove metadata file if it exists. Fresh start
    if METADATA_FILE_PATH.exists():
        METADATA_FILE_PATH.unlink()

    adata_array       = []
    sample_names      = find_subdirectories(SAMPLES_OF_INTEREST_DIR)    
    completed_samples = 0
    # -------------------------------
    # Check if the sample exists, if not delete from list
    removed_list      = []
    keep_list         = []
    for sample_name in sample_names:
        data_location = ROOT_PATH / Path(DATA_DIR) / Path(sample_name.strip()) / CELLRANGER_PATH
        file_name = data_location.with_suffix('.h5')
        if file_name.exists():
            keep_list.append(sample_name)
        else:
            removed_list.append(sample_name)
    sample_names  = keep_list
    # -------------------------------
    # Comment line before after done testing!!!
    # sample_names=[sample_names[0]]
    sample_names=sample_names[0:3]   
    # sample_names = ['3f79f07f-c357-44a3-866e-1999820ab445']
    # adata=read_and_process_data('3e436c7a-542e-4980-b52e-92d5d4bca496', ROOT_PATH, DATA_DIR, CELLRANGER_PATH)
    # adata=process_sample(sample_names[10])
    # -------------------------------
    total_samples = len(sample_names)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    with concurrent.futures.ThreadPoolExecutor() as executor:        
        futures = []
        progress_bar = tqdm(total=total_samples, desc="Processing Samples", unit="sample")
        for sample_name in sample_names:
            logging.info(f"----------------------------------------------")
            logging.info(f"Starting processing for sample: {sample_name} ({len(futures) + 1}/{total_samples})")
            future = executor.submit(process_sample, sample_name)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed_samples += 1
            progress_bar.update(1)
            if result is not None:
                adata_array.append(result)
                logging.info(f"----------------------------------------------")
                logging.info(f"Completed processing a sample ({completed_samples}/{total_samples})")
            else:
                logging.error(f"A sample failed to process or returned no data ({completed_samples}/{total_samples})")
        progress_bar.close()
    # -------------------------------
    logging.info(f"Concatenating {total_samples} samples")
    adata = sc.concat(adata_array, index_unique='_')
    
    adata.X     = csr_matrix(adata.X)
    # use 000 to have directory pop up first in list
    result_file = ROOT_PATH / Path(DATA_DIR) / Path('0a0a_'+RESULTS_DIR) / f'{total_samples}_concatenated_samples.h5ad'
    result_file.parent.mkdir(parents=True, exist_ok=True)
    # Check if the file exists and then delete it
    if result_file.exists():
        result_file.unlink()    
    adata.write_h5ad(result_file)

    print('Done')