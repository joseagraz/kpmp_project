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
OUTPUT_DIR              = ROOT_PATH / Path('0a0a_Results')
IMAGING_DIR             = Path('Imaging')
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
            target_file = sample_dir / "space_ranger_outs" / "filtered_feature_bc_matrix.h5"
            if target_file.exists():
                file_paths.append(target_file)
                sample_names.append(sample_name)

    return file_paths,sample_names
# ------------------------------------------ 
if __name__ == "__main__":
  # Set the normalization scale and seed for reproducibility    
  ov.utils.ov_plot_set()

  sc.set_figure_params(dpi=100)
  torch.set_float32_matmul_precision("high")    
  # print(f"Last run with scvi-tools version: {scvi.__version__}")

  visium_dir_of_interest = ROOT_PATH / DATA_DIR 
  sample_path_list,sample_names_list  = find_subdirectories(visium_dir_of_interest)

  sample_of_interest = 0

  sample_name_path = sample_path_list[sample_of_interest].parent
  adata = sc.read_visium(path=sample_name_path, count_file='filtered_feature_bc_matrix.h5')
                          
  adata.var_names_make_unique()
  sc.pp.calculate_qc_metrics(adata, inplace=True)
  adata = adata[:,adata.var['total_counts']>100]
  adata=ov.space.svg(adata,mode='prost',n_svgs=3000,target_sum=1e4,platform="visium",)

  output_path = OUTPUT_DIR / Path(sample_names_list[sample_of_interest])
  sample_name_file = output_path / 'omicverse_spatial_clustering_and_denoising_expressions.h5ad'
  output_path.mkdir(parents=True, exist_ok=True)

  adata.write(str(sample_name_file))
  adata=ov.read(str(sample_name_file))
  # ------------------------------------------ 
  output_path = OUTPUT_DIR / Path(sample_names_list[sample_of_interest]) / IMAGING_DIR
  output_path.mkdir(parents=True, exist_ok=True)
  # ------------------------------------------ 
  # # Method1: GraphST
  # methods_kwargs={}
  # methods_kwargs['GraphST']={
  #     'device':'cuda:0',
  #     'n_pcs':30
  # }
  # adata=ov.space.clusters(adata,
  #                   methods=['GraphST'],
  #                   methods_kwargs=methods_kwargs,
  #                   lognorm=1e4)  

  # ov.utils.cluster(adata,use_rep='graphst|original|X_pca',method='mclust',n_components=10,
  #                 modelNames='EEV', random_state=112,
  #                 )
  # adata.obs['mclust_GraphST'] = ov.utils.refine_label(adata, radius=50, key='mclust') 
  # adata.obs['mclust_GraphST']=adata.obs['mclust_GraphST'].astype('category')     

  # res=ov.space.merge_cluster(adata,groupby='mclust_GraphST',use_rep='graphst|original|X_pca',
  #                 threshold=0.2,plot=True)   

  # sc.pl.spatial(adata, color=['mclust_GraphST','mclust'],show=False)
  # plt.savefig(output_path/"spatial_plot.png", dpi=300, bbox_inches="tight")
  # plt.close()

  # ------------------------------------------       
  # Method2: BINARY
  #               
  methods_kwargs={}
  methods_kwargs['BINARY']={
      'use_method':'KNN',
      'cutoff':6,
      'obs_key':'BINARY_sample',
      'use_list':None,
      'pos_weight':10,
      'device':'cuda:0',
      'hidden_dims':[512, 30],
      'n_epochs': 1000,
      'lr':  0.001,
      'key_added': 'BINARY',
      'gradient_clipping': 5,
      'weight_decay': 0.0001,
      'verbose': True,
      'random_seed':0,
      'lognorm':1e4,
      'n_top_genes':2000,
  }
  adata=ov.space.clusters(adata,
                    methods=['BINARY'],
                  methods_kwargs=methods_kwargs)  

  res=ov.space.merge_cluster(adata,groupby='mclust_BINARY',use_rep='BINARY',
                  threshold=0.01,plot=True)


  plt.savefig(output_path/"binayr_spatial_plot.png", dpi=300, bbox_inches="tight")
  plt.close()

  print('Done')

  