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
root_path             = Path('/media/jagraz/12TB_Disk/KPMP_Data/Privately_Available_Data')
# NAS path
# root_path           = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
support_files_dir     = 'Supporting_Files'
data_dir              = 'Original_Download_KPMP_S3_Bucket_Oct_16_2023'
results_dir           = 'Results'
sample_name           = '0a8d4f18-84ca-4593-af16-3aaf605ca328'
source_data_path      = Path('cellranger_output/outs/filtered_feature_bc_matrix')
data_location         = root_path / Path(data_dir) / Path(sample_name) / source_data_path
MODEL_DIR             = root_path / Path(results_dir) / "scVI-model"
ADATA_FILE_PATH       = root_path / Path(results_dir) / "175_samples.h5ad"
CVS_FILE_PATH         = root_path / Path(support_files_dir) / 'list_of_removals.csv'
RIBO_LOOKUP_FILE_PATH = root_path / Path(support_files_dir) / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
# Testing
# samples_of_interest = root_path / Path(support_files_dir) / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
SAMPLES_OF_INTEREST_DIR = root_path / Path(data_dir)
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

    # try:
    #     adata, doublets_removed_count = doublet_removal(adata)        
    # except Exception as e:
    #     logging.error(f"Error doublet processing sample {sample_name}: {e}")  

    # It's assumed these samples don't have doublets
    adata.obs['doublet']                      = False

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
if __name__ == "__main__":    

    NORMALIZATION_SCALE = 1e4
    scvi.settings.seed  = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    save_dir = tempfile.TemporaryDirectory()

    adata_array       = []
    sample_names      = find_subdirectories(SAMPLES_OF_INTEREST_DIR)

    total_samples     = len(sample_names)
    completed_samples = 0
    removed_list      = []
    # -------------------------------
    # Comment line before after done testing!!!
    # sample_names=[sample_names[0]]
    # sample_names=sample_names[10:11]   
    # sample_names = ['3f79f07f-c357-44a3-866e-1999820ab445']
    # adata=read_and_process_data('3e436c7a-542e-4980-b52e-92d5d4bca496', root_path, data_dir, source_data_path)
    # Failing samples
    sample_names = ['3f79f07f-c357-44a3-866e-1999820ab445', \
                    'e048e99c-c18c-44b8-9c1f-db9730f1f240', \
                    '3e436c7a-542e-4980-b52e-92d5d4bca496', \
                    '99faf665-3081-4c2f-9f25-866ff52cfc98', \
                    'a797d182-005c-41e5-a6dd-3c0f0badce95', \
                    'a84202f7-b01c-4831-9e14-7a9261d96afa', \
                    'faf11845-5073-4451-850c-5ed537e601c4', \
                    'fce6b986-e65e-45af-a148-a18deac621dd', \
                    '1b487afa-ac6f-4cda-996b-c943e082b5f1'  \
                    ]
    # -------------------------------
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sample_name in sample_names:
            logging.info(f"----------------------------------------------")
            logging.info(f"Starting processing for sample: {sample_name} ({len(futures) + 1}/{total_samples})")
            future = executor.submit(read_and_process_data, sample_name, root_path, data_dir, source_data_path)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed_samples += 1
            if result is not None:
                adata_array.append(result)
                logging.info(f"----------------------------------------------")
                logging.info(f"Completed processing a sample ({completed_samples}/{total_samples})")
            else:
                logging.error(f"A sample failed to process or returned no data ({completed_samples}/{total_samples})")
    # -------------------------------
    # logging.info(f"Concatenating {total_samples} samples")
    # adata = sc.concat(adata_array, index_unique='_')
    
    
    # adata.X     = csr_matrix(adata.X)
    # result_file = root_path / Path(data_dir) / 'combined.h5ad'
    # # Check if the file exists and then delete it
    # if result_file.exists():
    #     result_file.unlink()    
    # adata.write_h5ad(result_file)

    print('Done')