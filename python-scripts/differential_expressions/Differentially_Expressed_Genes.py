from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from typing import Tuple
import scanpy as sc
import pandas as pd
import numpy as np
import bbknn    
import scvelo as scv
from matplotlib.pyplot import rc_context
from pathlib import Path
import scvi
import os
import logging
import torch
import seaborn as sns
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import anndata as ad
from anndata import AnnData
import random
import warnings
# from gseapy.plot import gseaplot
import networkx as nx
from gseapy import gseaplot
import gseapy as gp
from gseapy import dotplot
from gseapy import enrichment_map
from gseapy import gseaplot2
from gseapy import heatmap
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
# Testing
# samples_of_interest = ROOT_PATH / SUPPORT_FILES_DIR / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
MEN                     = 0
WOMEN                   = 1
SCVI_LATENT_KEY         = "X_scVI"
LEIDEN_RESOLUTION       = 0.8
NORMALIZATION_SCALE     = 1e4
NUMBER_OF_GENES         = 3000
MINIMUM_GENES           = 200
MINIMUM_CELLS           = 3
LEIDEN_CLUSTER_NAME     = 'leiden'
INTEGRATED_OUTPUT_FILE  = ROOT_PATH / DATA_DIR / 'integrated.h5ad'
LOOKUP_TABLE            = { 'Healthy Reference': 0,'CKD': 1,'AKI': 2,'DM-R': 3,'None': 4 }
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
    
    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=80, color_map='viridis')
    # ---------------------------
    number_of_samples_to_keep = 175
    directory_path = IMAGING_DIR / Path(f'{number_of_samples_to_keep}_samples_integration_images_Leiden_resolution_{LEIDEN_RESOLUTION}')
    output_file    = directory_path / INTEGRATED_OUTPUT_FILE
    adata          = sc.read_h5ad(output_file)
    adata.X        = adata.layers['counts']
    adata.var_names_make_unique()
    adata.obs_names_make_unique()    

    # Perform differential expression analysis
    # Here, 'cell_type' is a column in the .obs attribute that defines groups
    sc.tl.rank_genes_groups(adata, 'celltype', method='wilcoxon')

    # View the results
    sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

    # Access the ranked genes results
    # You can inspect the results using:
    print(adata.uns['rank_genes_groups'].keys())  # Shows available keys
    # The main result (gene names) is under the 'names' key
    ranked_genes = adata.uns['rank_genes_groups']

    # Initialize a DataFrame using one of the lists (e.g., 'names')
    group_names = ranked_genes['names'].dtype.names


    # Initialize a dictionary to gather all the data
    data = []

    # Iterate through each group and extract the gene data
    for group in group_names:
        group_data = []
        for i in range(ranked_genes['names'][group].shape[0]):
            gene_data = {
                'group': group,
                'gene': ranked_genes['names'][group][i],
                'score': ranked_genes['scores'][group][i],
                'pval': ranked_genes['pvals'][group][i],
                'logfoldchange': ranked_genes['logfoldchanges'][group][i],
                'pval_adj': ranked_genes['pvals_adj'][group][i]
            }
            data.append(gene_data)
        #     group_data.append(gene_data)
        # df = pd.DataFrame(group_data)
        # df = df[df['score'] >= 0]
        # df.sort_values('score', ascending = False).reset_index(drop = True)
        # df.set_index('gene')

        # df.to_csv(PLOT_FILE_PATH/Path(gene_data['group']+'_ranked_genes.csv'), index=False)


    # Create a DataFrame from the gathered data
    df_ranked_genes = pd.DataFrame(data)
    df_ranked_genes.sort_values('score', ascending = False).reset_index(drop = True)
    df_ranked_genes.set_index('gene')

    df_ranked_genes.to_csv(PLOT_FILE_PATH/'ranked_genes.csv', index=False)
    print