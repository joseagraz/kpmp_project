import scanpy as sc
import pandas as pd
from matplotlib.pyplot import rc_context
import numpy as np
from pathlib import Path
import scvi
import logging
import concurrent.futures
import os
import torch
import tempfile
from scipy.sparse import csr_matrix
# ------------------------------------------
# Script Information
__author__ = "Jose L. Agraz, PhD"
__status__ = "Prototype"
__email__ = "jose@agraz.email"
__credits__ = ["Jose L. Agraz", "Parker Wilson"]
__license__ = "MIT"
__version__ = "1.0"
# ------------------------------------------
sc.set_figure_params(dpi=100)
# 12TB disk path
root_path           = Path('/mnt/12TB_Disk/KPMP_Data/Privately_Available_Data')
# NAS path
# root_path           = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
support_files_dir   = 'Supporting_Files'
data_dir            = 'Original_Download_KPMP_S3_Bucket_Oct_16_2023'
results_dir         = 'Results'
sample_name         = '0a8d4f18-84ca-4593-af16-3aaf605ca328'
source_data_path    = Path('cellranger_output/outs/filtered_feature_bc_matrix')
data_location       = root_path / Path(data_dir) / Path(sample_name) / source_data_path
MODEL_DIR           = root_path / Path(results_dir) / "scVI-model"
ADATA_FILE_PATH     = root_path / Path(results_dir) / "175_samples.h5ad"
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
def remove_mitochondrial_genes(adata):
    UPPER_QUANTILE = 0.98
    LOWER_QUANTILE = 0.02
    MT_COUNT_LIMIT = 20

    adata.var['mt'] = adata.var_names.str.contains('^MT-', 
                                                   case=False, 
                                                   regex=True)  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, 
                               qc_vars=['mt'], 
                               percent_top=None, 
                               log1p=False, 
                               inplace=True)

    # Plot results
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

    # sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    #instead of picking subjectively, you can use quanitle
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, UPPER_QUANTILE)
    lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, LOWER_QUANTILE)
    print(f'{lower_lim} to {upper_lim}')

    adata = adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]
    adata = adata[adata.obs.pct_counts_mt < MT_COUNT_LIMIT]

    #normalize every cell to 10,000 UMI
    # sc.pp.normalize_total(adata, target_sum=NORMALIZATION_SCALE) 

    #change to log counts
    # sc.pp.log1p(adata) 
    return(adata)     
# ------------------------------------------
def remove_doublets(adata):
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        subset=True,
        flavor="seurat_v3"
    )

    # casting
    # converting your sparse matrix to Compressed Sparse Row (CSR) format
    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train()    

    return(adata)
# ------------------------------------------
def process_sample(data_location):

    adata = sc.read_10x_mtx(
        data_location,  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)  
    
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=20) #get rid of genes that are found in fewer than 3 cells
    # 6806x25968

    adata = remove_mitochondrial_genes(adata)

    adata = remove_doublets(adata)

    return(adata)
# ------------------------------------------
def read_and_process_data(sample_name, root_path, data_dir, source_data_path):
    try:
        # logging.info(f"processing sample: {sample_name.strip()}")
        data_location = root_path / Path(data_dir) / Path(sample_name.strip()) / source_data_path
        # logging.info(f"Path: {data_location}")
        adata = process_sample(data_location)   
        adata.obs['Sample'] = sample_name.strip()
        return adata
    except Exception as e:
        logging.error(f"Error processing sample {sample_name}: {e}")
        return None
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

    # Comment line before after done testing!!!
    sample_names=sample_names[1:5]

    total_samples     = len(sample_names)
    completed_samples = 0

    for sample_name in sample_names:            
        read_and_process_data(sample_name, root_path, data_dir, source_data_path)

    adata = sc.concat(adata_array, index_unique='_')