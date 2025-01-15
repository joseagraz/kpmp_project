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

def find_gene_matches(dict_list,direction='high_to_low'):
    gene_count = {}
    gene_sources = {}

    for idx, dct in enumerate(dict_list):
        for gene in dct['high_to_low']:
            if gene:
                if gene not in gene_count:
                    gene_count[gene] = 0
                    gene_sources[gene] = []
                gene_count[gene] += 1
                gene_sources[gene].append(f'dct{idx + 1}')
    
    # Filter genes that appear in more than one dictionary
    common_genes = {gene: sources for gene, sources in gene_sources.items() if gene_count[gene] > 1}
    
    return common_genes
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
    
    dct1_gene_expression_direction = {'high_to_low': ['GPC5','GPX3','AC092078.2','TRIM50','XIST','DNER','GOT1'],
                                      'low_to_high': ['ERBB4','POLR2J3','DCDC2','FAM155A','NEAT1'] }
    dct2_gene_expression_direction = {'high_to_low': ['XIST','BICC1','GLIS3','LINC01099','GPC5','PDE4D'],
                                      'low_to_high': ['NEAT1','NSD11B2','POLR2J3'] }
    fib_vsmc_mc_gene_expression_direction = {
                                      'high_to_low': [],
                                      'low_to_high': ['CALD1','ADAMTS9-AS2','MECOM','NEAT1'] }
    ica_gene_expression_direction = { 'high_to_low': ['SHOC1'],
                                      'low_to_high': [] }    
    icb_gene_expression_direction = { 'high_to_low': [],
                                      'low_to_high': ['ARL17B'] }        
    leuk_gene_expression_direction = { 'high_to_low': [''],
                                      'low_to_high': ['PDE4D','AC019197.1','SLC8A1'] }            
    pec_gene_expression_direction = { 'high_to_low': [],
                                      'low_to_high': ['RBFOX1'] }   
    pt_gene_expression_direction = {  'high_to_low': ['TACC1'],
                                      'low_to_high': [] }    
    pt_mt_gene_expression_direction = {'high_to_low': [],
                                      'low_to_high': ['NEAT1','DDX17','DDX5'] }  
    pt_prom1_gene_expression_direction = {
                                      'high_to_low': [],
                                      'low_to_high': ['STAT3'] }   
    pt_vcam1_gene_expression_direction = {
                                      'high_to_low': [],
                                      'low_to_high': ['NEAT1','SAMD4A','PAPPA'] }   
    tal1_gene_expression_direction = {'high_to_low': [],
                                      'low_to_high': ['LMO1'] }     
    tal2_gene_expression_direction = {'high_to_low': ['UMOD'],
                                      'low_to_high': ['LINC01320','RHOBTB3','ZBTB16','THSD7A','LMO7'] }  
    list_of_dict_gene_expression_direction = [dct1_gene_expression_direction,
                                              dct2_gene_expression_direction,
                                              fib_vsmc_mc_gene_expression_direction,
                                              ica_gene_expression_direction,
                                              icb_gene_expression_direction,
                                              leuk_gene_expression_direction,
                                              pec_gene_expression_direction,
                                              pt_gene_expression_direction,
                                              pt_mt_gene_expression_direction,
                                              pt_prom1_gene_expression_direction,
                                              pt_vcam1_gene_expression_direction,
                                              tal1_gene_expression_direction,
                                              tal2_gene_expression_direction]

    h_to_low, low_to_high = find_common_genes(list_of_dict_gene_expression_direction)

    high_gene_occurrences = find_gene_matches(list_of_dict_gene_expression_direction, direction='high_to_low')
    """
    GPC5:DCT1, DCT2
    XIST:DCT1, DCT2

    """

    low_gene_occurrences = find_gene_matches(list_of_dict_gene_expression_direction, direction='low_to_high')
    """
    POLR2J3: DCT1, DCT2
    NEAT1: DCT1, DCT2, FIB_VSMC_MC, PT_MT, PT_VCAM1
    """
    print