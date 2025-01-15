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
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter(action="ignore", category=Warning)
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
PLOT_FILE_PATH          = ROOT_PATH / SUPPORT_FILES_DIR / Path('Plots') /'parker'
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


if __name__ == "__main__":    

    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(8, 8))
    torch.set_float32_matmul_precision("high")
    
    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=80, color_map='viridis')
    # ---------------------------
    source_dir = Path('/media/KPMP_Data/Privately_Available_Data/Supporting_Files/Data_Integration/May_15_2024/175_samples_integration_images_Leiden_resolution_0.6/')
    df_loy = pd.read_csv(source_dir / 'loy.csv')

    df_pt = df_loy[df_loy['celltype'] == 'PT']              # XY=8514 LY=843
    df_PT_MT = df_loy[df_loy['celltype'] == 'PT_MT']        # XY=1907 LY=425
    df_PT_VCAM1 = df_loy[df_loy['celltype'] == 'PT_VCAM1']  # XY=8642 LY=1163
    df_PT_PROM1 = df_loy[df_loy['celltype'] == 'PT_PROM1']  # XY=1427 LY=247

    genotype_counts = df_loy.groupby(['celltype', 'genotype']).size().reset_index(name='counts')

    # Save the genotype counts per celltype to a new CSV file
    genotype_counts.to_csv(source_dir / 'loy_ratios.csv', index=False)

    data= df_loy

    cell_types = ["PT_MT", "PT_PROM1", "PT_VCAM1", "LEUK", "DCT1", "DCT2", "ICA", "ICB", "TAL1", "TAL2"]
    cell_types = ["PT_MT"]
    filtered_data = data[data['celltype']==cell_types]

    # Calculate the proportion of LOY to XY for each cell type
    filtered_data['LOY_XY_ratio'] = filtered_data.apply(lambda row: row['corrected_rna_counts'] / row['total_rna_counts'] if row['genotype'] == 'LOY' else 0, axis=1)

    # Plotting the data using boxplots for each cell type
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='celltype', y='LOY_XY_ratio', data=filtered_data)
    plt.title('Proportion of LOY to XY for Selected Cell Types')
    plt.xlabel('Cell Type')
    plt.ylabel('LOY to XY Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()   

    print


    # sample_gender_df = adata.obs[['Sample_Name', 'Sample_Age']]
    # x=sample_gender_df.drop_duplicates(subset=['Sample_Name']).value_counts()
    # x.groupby('Sample_Age').sum()

    # sample_gender_df = adata.obs[['Sample_Name', 'Sample_Tissue_Type']]
    # x=sample_gender_df.drop_duplicates(subset=['Sample_Name']).value_counts()
    # x.groupby('Sample_Tissue_Type').sum()
