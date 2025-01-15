"""
Preprocess Samples Script for Single-Cell Genomic Data

This Python script automates data cleaning and label transfering using an Atlas in preparation for data integration. 
Utilizing parallel processing, Scanpy, Pandas, NumPy, and scVI, the scripts streamlines the analysis 
pipeline, ensuring that your data is ready for downstream analysis.

Author: Jose L. Agraz and Parker Wilson
Date: March 27, 2024
Version: 1.0
"""
# import scvelo as scv
import scanpy as sc
import time
import warnings
import gc
import omicverse as ov
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
import torch
import re
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

ov.utils.ov_plot_set()
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
    # --------------------------------------------------------
    # Save the combined dataset to a file, overwriting if it already exists
    ov.settings.gpu_init()
    concatenated_results_file = OUTPUT_DIR / 'rapids_single_cell_concatenated_results_file.h5ad'
    adata=sc.read_h5ad(concatenated_results_file)
    adata.var_names_make_unique()
    adata.obs_names_make_unique() 

    adata = adata[:, ~(adata.var['MT'] | adata.var['RIBO'])].copy() 
    adata.X=adata.X.astype(np.int64)
    # adata.layers["counts"] = adata.X.copy()  
    ov.utils.store_layers(adata,layers='counts')
    # --------------------------------------------------------            
    adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=3000,target_sum=50*1e4)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable_features]

    ov.pp.scale(adata)
    ov.pp.pca(adata,layer='scaled',n_pcs=50)

    adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca']
    ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
               use_rep='scaled|original|X_pca',method='cagra')

    # adata.obsm["X_mde_pca"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
    adata.obsm["X_mde"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
    # --------------------------------------------------------            
    # ov.utils.embedding(adata,basis='X_mde_pca',frameon='small',color=['batch'],show=False)
    ov.utils.embedding(adata,basis='X_mde',frameon='small',color=['batch'],show=False)
    # --------------------------------------------------------            
    ov.single.batch_correction(adata,batch_key='batch',
                               methods='harmony',n_pcs=50)
    
    adata.obsm["X_mde_harmony"] = ov.utils.mde(adata.obsm["X_harmony"])
    
    ov.utils.embedding(adata,
                basis='X_mde_harmony',frameon='small',
                color=['batch'],show=False)


    # ov.utils.plot_pca_variance_ratio(adata)
    sc.pp.neighbors(adata,n_neighbors=15,n_pcs=50,use_rep='scaled|original|X_pca')  
    ov.pp.umap(adata) 
    # --------------------------------------------------------            
    # Latent Dirichlet Allocation (LDA) model implementation
    LDA_obj=ov.utils.LDA_topic(adata,
                               feature_type='expression',
                               highly_variable_key='highly_variable_features',
                               layers='counts',
                               batch_key='batch',
                               learning_rate=1e-3)
    LDA_obj.plot_topic_contributions(6)
    # --------------------------------------------------------


    # Save the combined dataset to a file, overwriting if it already exists
    concatenated_results_file = OUTPUT_DIR / 'rapids_single_cell_integrated_results_file.h5ad'
    if concatenated_results_file.exists():
        concatenated_results_file.unlink()
    
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


