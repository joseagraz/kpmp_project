"""
Preprocess Samples Script for Single-Cell Genomic Data

This Python script automates data cleaning and label transfering using an Atlas in preparation for data integration. 
Utilizing parallel processing, Scanpy, Pandas, NumPy, and scVI, the scripts streamlines the analysis 
pipeline, ensuring that your data is ready for downstream analysis.

Author: Jose L. Agraz and Parker Wilson
Date: March 27, 2024
Version: 1.0
"""
import omicverse as ov
import gc
from typing import Tuple,List
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
import scanpy as sc
import matplotlib.pyplot as plt
import scanpy as sc
# ------------------------------------------
# reference: 
# Complete single-cell RNAseq analysis walkthrough | Advanced introduction
# https://www.youtube.com/watch?v=uvyG9yLuNSE&t=635s
# ------------------------------------------
ROOT_PATH               = Path('/media/KPMP_Data/Publicly_Available_Data/Visium_Spatial_Data')
SUPPORT_FILES_DIR       = 'Supporting_Files'
DATA_DIR                = 'Visium_Spatial_Data_by_Participant_ID'
SOURCE_DATA_PATH        = Path('cellranger_output/outs/filtered_feature_bc_matrix')
OUTPUT_DIR              = ROOT_PATH / DATA_DIR / Path('0a0a_Results')
# ------------------------------------------  
if __name__ == "__main__":
    # Set the normalization scale and seed for reproducibility    
    ov.utils.ov_plot_set()
    # ov.settings.gpu_init()
    # scvi.settings.seed = 0
    sc.set_figure_params(dpi=100)
    torch.set_float32_matmul_precision("high")    
    # print(f"Last run with scvi-tools version: {scvi.__version__}")

    # Set default figure parameters and higher precision for float32 matrix multiplication
    sc.settings.n_jobs=8
    # sc.set_figure_params()

    # Save the combined dataset to a file, overwriting if it already exists
    # input_file = OUTPUT_DIR / 'single_harmony_cell_labeled_results_file.h5ad'        
    # adata=sc.read_h5ad(input_file)

    sample_name = '21-015'
    file_path= ROOT_PATH / DATA_DIR / sample_name / Path('space_ranger_outs')
    adata = sc.read_visium(path=file_path, count_file='filtered_feature_bc_matrix.h5')
                           
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[:,adata.var['total_counts']>100]
    adata=ov.space.svg(adata,mode='prost',n_svgs=3000,target_sum=1e4,platform="visium",)

    print('Done')



ROOT_PATH               = Path('/media/KPMP_Data/Publicly_Available_Data/Visium_Spatial_Data')
SUPPORT_FILES_DIR       = 'Supporting_Files'
DATA_DIR                = 'Visium_Spatial_Data_by_Participant_ID'

