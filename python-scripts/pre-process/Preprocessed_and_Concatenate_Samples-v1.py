import scanpy as sc
import pandas as pd
from matplotlib.pyplot import rc_context
import numpy as np
from pathlib import Path
import scvi
import logging
import concurrent.futures
import csv
import os
import torch
import tempfile
from collections import Counter
import seaborn as sns
import time
from scipy.sparse import csr_matrix
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,
)
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
# 12TB disk path
ROOT_PATH             = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR     = 'Supporting_Files'
DATA_DIR              = 'Original_Download_KPMP_S3_Bucket_Oct_16_2023'
RESULTS_DIR           = 'Results'
#
sample_name           = '0a8d4f18-84ca-4593-af16-3aaf605ca328'
source_data_path      = Path('cellranger_output/outs/filtered_feature_bc_matrix')
data_location         = ROOT_PATH / Path(DATA_DIR) / Path(sample_name) / source_data_path
#
MODEL_DIR             = ROOT_PATH / Path(RESULTS_DIR) / "scVI-model"
ADATA_FILE_PATH       = ROOT_PATH / Path(RESULTS_DIR) / "175_samples.h5ad"
CVS_FILE_PATH         = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'list_of_removals.csv'
RIBO_LOOKUP_FILE_PATH = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
# Testing
# samples_of_interest = root_path / Path(support_files_dir) / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
SAMPLES_OF_INTEREST_DIR = ROOT_PATH / Path(DATA_DIR)
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
def doublet_removal(adata_original):
    DOUBLETS_PREDICTION = 'prediction'
    DOUBLETS            = 'doublet'
    SINGLET             = 'singlet'

    adata          = adata_original.copy()
    adata_doublets = adata_original.copy()
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes = 3000,
        subset      = True,
        flavor      = "seurat_v3"
    )
    # Model to predict doublets using scVI
    scvi.model.SCVI.setup_anndata(adata)
    doublets_model = scvi.model.SCVI(adata) 
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
def poor_cell_and_gene_removal(adata):
    # Count the number of cells and genes before filtering
    num_cells_before = adata.n_obs
    num_genes_before = adata.n_vars

    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=100) #get rid of genes that are found in fewer than xx cells    

    # Count the number of cells and genes after filtering
    num_cells_after = adata.n_obs
    num_genes_after = adata.n_vars

    # Calculate the number of cells and genes removed
    num_cells_removed = num_cells_before - num_cells_after
    num_genes_removed = num_genes_before - num_genes_after

    return adata, num_cells_removed, num_genes_removed
# ------------------------------------------
def process_sample(data_location,sample_name):
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
    adata = sc.read_10x_h5(data_location.with_suffix('.h5'))

    adata.var_names_make_unique()

    try:
        adata, doublets_removed_count = doublet_removal(adata)
        # print        
    except Exception as e:
        logging.error(f"Error doublet processing sample {sample_name}: {e}")  

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
    if not os.path.isfile(CVS_FILE_PATH):
        with open(CVS_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    # Append the data to the CSV file
    with open(CVS_FILE_PATH, 'a', newline='') as file:
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
# def read_and_process_data(sample_name, root_path, data_dir, source_data_path):
#     try:
#         # logging.info(f"processing sample: {sample_name.strip()}")
#         data_location = root_path / Path(data_dir) / Path(sample_name.strip()) / source_data_path
#         # logging.info(f"Path: {data_location}")
#         adata = process_sample(data_location,sample_name)
#         adata.obs['Sample_Name'] = sample_name.strip()
#         return adata
#     except Exception as e:
#         logging.error(f"Error processing sample {sample_name}: {e}")
#         return None
# ------------------------------------------
def read_and_process_data(sample_name, root_path, data_dir, source_data_path):
        # logging.info(f"processing sample: {sample_name.strip()}")
        data_location = root_path / Path(data_dir) / Path(sample_name.strip()) / source_data_path
        # logging.info(f"Path: {data_location}")
        adata = process_sample(data_location,sample_name)
        adata.obs['Sample_Name'] = sample_name.strip()
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
def find_files(root_path, file_name):
    """
    Find all instances of a file with a given name under the root path.

    Parameters:
    root_path (str): The root directory to start the search from.
    file_name (str): The name of the file to search for.

    Returns:
    list: A list of Paths where the file is found.
    """
    root = Path(root_path)
    return [file for file in root.rglob(file_name)]
# ------------------------------------------   
if __name__ == "__main__":    

    NORMALIZATION_SCALE = 1e4
    scvi.settings.seed  = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    
    adata_array       = []
    file_of_interest  = 'filtered_feature_bc_matrix_preprocessing-filtered.h5ad'
    # -------------------------------
    found_files = find_files(ROOT_PATH / Path(DATA_DIR) , file_of_interest)
    logging.info(f"Found {len(found_files)} files to concatenate")

    for file_path in found_files:
        adata = sc.read_h5ad(file_path)
        adata_array.append(adata)
    # -------------------------------
    logging.info(f"Concatenating {len(found_files)} samples")
    adata = sc.concat(adata_array, index_unique='_')
        
    adata.X     = csr_matrix(adata.X)
    result_file = ROOT_PATH / Path(DATA_DIR) / 'combined.h5ad'
    # Check if the file exists and then delete it
    if result_file.exists():
        result_file.unlink()    
    adata.write_h5ad(result_file)

    print('Done')