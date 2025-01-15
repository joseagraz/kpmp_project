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
if __name__ == "__main__":
    # ------------------------------------
    rmm.reinitialize(
        managed_memory=False,  # Allows oversubscription
        pool_allocator=False,  # default is False
        devices=0,  # GPU device IDs to register. By default registers only GPU 0.
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    # --------------------------------------------------------
    # Save the combined dataset to a file, overwriting if it already exists
    concatenated_results_file = OUTPUT_DIR / 'rapids_single_cell_concatenated_results_file.h5ad'
    adata=sc.read_h5ad(concatenated_results_file)
    adata.var_names_make_unique()
    adata.obs_names_make_unique() 

    adata = adata[:, ~(adata.var['MT'] | adata.var['RIBO'])].copy() 

    del adata.obs['log1p_n_genes_by_counts']
    del adata.obs['log1p_total_counts_MT']
    del adata.obs['total_counts_RIBO']
    del adata.obs['pct_counts_RIBO']
    del adata.obs['log1p_total_counts_RIBO']
    del adata.obs['log1p_total_counts']
    del adata.obs['total_counts_MT']
    del adata.obs['pct_counts_MT']
    del adata.var['MT']
    del adata.var['RIBO']
    del adata.var['feature_types']

    gc.collect()
    # --------------------------------------------------------
    print('Normalizing')
    rsc.get.anndata_to_GPU(adata)
    adata.layers["counts"] = adata.X.copy()    
    gc.collect()
    rsc.get.anndata_to_CPU(adata)
    # --------------------------------------------------------
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    gc.collect()    
    rsc.get.anndata_to_CPU(adata)
    # --------------------------------------------------------
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.log1p(adata)
    gc.collect()
    rsc.get.anndata_to_CPU(adata)
    # --------------------------------------------------------
    print('highly_variable_genes')
    rsc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", batch_key= "batch",layer="counts")
    gc.collect()
    rsc.get.anndata_to_CPU(adata, layer="counts")
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


