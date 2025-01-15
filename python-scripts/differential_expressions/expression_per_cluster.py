from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from typing import Tuple
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib.pyplot import rc_context
from pathlib import Path
import scvi
import os
import logging
import torch
import seaborn as sns
import shutil
import matplotlib.pyplot as plt
import anndata as ad
from anndata import AnnData
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
# ------------------------------------------
# Script Information
__author__ = "Jose L. Agraz, PhD"
__status__ = "Prototype"
__email__ = "jose@agraz.email"
__credits__ = ["Jose L. Agraz", "Parker Wilson MD, PhD"]
__license__ = "MIT"
__version__ = "1.0"
# ------------------------------------------
# reference: 
# Complete single-cell RNAseq analysis walkthrough | Advanced introduction
# https://www.youtube.com/watch?v=uvyG9yLuNSE&t=635s
# ------------------------------------------
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# 12TB disk path
ROOT_PATH               = Path('/media/jagraz/8TB/KPMP_Data/Privately_Available_Data')
# NAS path
# ROOT_PATH               = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR       = Path('Supporting_Files')
DATA_DIR                = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
PLOT_FILE_PATH          = ROOT_PATH / SUPPORT_FILES_DIR / 'Plots'
CLUSTERING_MODEL_PATH   = ROOT_PATH / SUPPORT_FILES_DIR / 'scVI_Models'
IMAGING_DIR             = ROOT_PATH / SUPPORT_FILES_DIR
METADATA_FILE_PATH      = ROOT_PATH / SUPPORT_FILES_DIR / 'metadata.csv'
RIBO_LOOKUP_FILE_PATH   = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
MEAN_EXPRESSION_PATH    = ROOT_PATH / SUPPORT_FILES_DIR / 'mean_expression_per_cluster.csv'
# Testing
# samples_of_interest = ROOT_PATH / SUPPORT_FILES_DIR / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
SCVI_LATENT_KEY         = "X_scVI"
LEIDEN_RESOLUTION       = 0.8
NORMALIZATION_SCALE     = 1e4
NUMBER_OF_GENES         = 3000
MINIMUM_GENES           = 500
MINIMUM_CELLS           = 200
LEIDEN_CLUSTER_NAME     = 'leiden'
INTEGRATED_OUTPUT_FILE  = ROOT_PATH / DATA_DIR / 'integrated.h5ad'
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------  


if __name__ == "__main__":    

    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(8, 8))
    torch.set_float32_matmul_precision("high")
    inference = DefaultInference(n_cpus=20)
    # -------------------------------
    # csv file with valid dataset metadata
    metadata_df  = pd.read_csv(METADATA_FILE_PATH)
    dataset_name = list(set(metadata_df.sample_name.tolist()))
    # ---------------------------
    number_of_samples_to_keep = 175
    directory_path = IMAGING_DIR / Path(f'{number_of_samples_to_keep}_samples_integration_images_Leiden_resolution_{LEIDEN_RESOLUTION}')
    output_file    = directory_path / INTEGRATED_OUTPUT_FILE
    adata          = sc.read_h5ad(output_file)
    # sc.pl.umap(adata, color=["total_counts", "clusters"], legend_loc='on data',wspace=0.4)
    # sc.pl.umap(adata,color=['celltype'],legend_loc='on data',title='Cell Types',frameon=False, layer='scvi_normalized')
    print(f'Integrated dataset shape: {adata.shape}')

    # Normalize data
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Log-transform data
    sc.pp.log1p(adata)

    # Scale data
    sc.pp.scale(adata, max_value=10)

    # Calculate Mean Expression Per Cluster
    # Calculate the mean expression for each gene in each cluster
    cell_type_column_name = 'celltype'
    cluster_column_name = 'clusters'
    grouped = adata.to_df().groupby(adata.obs[cluster_column_name])
    mean_expression = grouped.mean()

    # Remove Cluster 18 - no useful data there
    mean_expression.drop([18], inplace=True)

    # Map cluster to cell type    
    cluster_to_celltype = adata.obs.set_index(cluster_column_name)[cell_type_column_name]
    cluster_to_celltype = cluster_to_celltype[~cluster_to_celltype.index.duplicated(keep='first')]
    mean_expression.index = mean_expression.index.map(str)
    cluster_to_celltype.index = cluster_to_celltype.index.map(str)
    mean_expression[cell_type_column_name] = mean_expression.index.map(lambda x: cluster_to_celltype.get(x, 'Unknown'))

    columns = [col for col in mean_expression.columns if col != cell_type_column_name]
    new_column_order = [cell_type_column_name] + columns
    mean_expression = mean_expression[new_column_order]

    mean_expression.to_csv(MEAN_EXPRESSION_PATH)






    print