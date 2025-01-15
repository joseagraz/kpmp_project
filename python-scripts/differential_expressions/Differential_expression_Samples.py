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
from pathlib import Path
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
# Testing
# samples_of_interest = ROOT_PATH / SUPPORT_FILES_DIR / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
MEN                     = 0
WOMEN                   = 1
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
def filter_low_count_cells(adata: AnnData) -> AnnData: 
    cell_counts  = adata.obs['Sample_Name'].value_counts()
    good_samples = cell_counts[cell_counts >= 30].index
    adata        = adata[adata.obs['Sample_Name'].isin(good_samples)]
    return adata

def bulk_data_analysis(adata: AnnData,cell_types: str) -> AnnData:

    pbs = []
    # Loop through each sample
    for sample in cell_subset.obs['Sample_Name'].unique():
        samp_cell_subset   = cell_subset[cell_subset.obs['Sample_Name'] == sample].copy()        
        samp_cell_subset.X = samp_cell_subset.layers['counts'] #make sure to use raw data
        # Sum the counts for each gene, transfer gene names
        rep_adata = sc.AnnData(X   = samp_cell_subset.X.sum(axis = 0),
                               var = samp_cell_subset.var[[]])
        
        rep_adata.obs_names = [sample]
        # rep_adata.obs['Sample_Disease'] = samp_cell_subset.obs['Sample_Disease'].iloc[0]
        # rep_adata.obs['Sample_Gender']  = samp_cell_subset.obs['Sample_Gender'].iloc[0]
        rep_adata.obs['Sample_Ages']    = samp_cell_subset.obs['Sample_Ages'].iloc[0]
        pbs.append(rep_adata)  

    if len(pbs) != 0:
        pb = sc.concat(pbs)
        if pb.n_obs > 2:
            counts_df = pd.DataFrame(pb.X.astype(np.int32) , index=pb.obs.index, columns=pb.var.index)    
            metadata  = pb.obs.copy()

            condition_list = ['Sample_Ages']
            # Filter out samples with missing data
            for condition in condition_list:        
                samples_to_keep = ~metadata[condition].isna()
                counts_df = counts_df.loc[samples_to_keep]
                metadata = metadata.loc[samples_to_keep]

            dds = DeseqDataSet(
                counts         = counts_df,
                metadata       = metadata,
                design_factors = condition_list,
                refit_cooks=True,
                inference=inference
                )
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast = ('Sample-Ages','70','20'),inference=inference)
            stat_res.summary()
            res = stat_res.results_df
            res = res[res.baseMean >= 10]
            # significant values
            sigs = res[(res.padj < 0.05) & (abs(res.log2FoldChange) > 0.5)]

            sc.pp.filter_genes(dds,min_cells=1)
            if dds.n_obs > 12:
                dds.deseq2()

                file_name  = f'Males_20_to_70yrs_old' 
                plot_title = 'PCA:' + file_name.replace('_',' ') + f', Cell type: {cell_types}'
                file_path  = PLOT_FILE_PATH / Path('Pseudobulk_single-cell_analysis')/ Path('Sample_Disease')/(file_name+ f'_celltype_{cell_types}.png')
                file_path.parent.mkdir(parents=True, exist_ok=True)

                plt.switch_backend('Agg')
                plt.rcParams["figure.figsize"] = (16, 8)
                sc.tl.pca(dds)
                sc.pl.pca(dds, color = 'Sample-Ages', size = 200, title = plot_title)    
                fig, ax = plt.subplots()
                sc.pl.pca(dds, color='Sample-Ages', size=200, title=plot_title, ax=ax, show=False)  # Use the created axes
                plt.savefig(file_path)
                plt.close()  
            else:
                print(f'Not enough samples for cell type: {cell_types}')    
        else:
            print(f'Not enough samples for cell type: {cell_types}')
    else:
        print(f'Not enough samples for cell type: {cell_types}')

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

    # Subset data 
    # 0=male, 1=female, "Plasma Cells" = PC
    gender      = MEN
    # cell_types  = ['PC']    
    cell_types  = adata.obs['celltype'].unique().tolist()  # All cell types

    for cell_type in cell_types:
        print(f'Processing cell type: {cell_type}')
        cell_subset = adata[(adata.obs['Sample_Gender'] == 0) & (adata.obs['celltype'].isin([cell_type]))].copy()
        # cell_subset = adata[(adata.obs['Sample_Gender'] == 0) & (adata.obs['celltype'] == 'PC')].copy()

        # cell_subset.obs.groupby(['Sample_Disease','Sample_Name']).size()
        cell_subset.obs.groupby(['Sample_Name']).size()

        cell_subset = filter_low_count_cells(cell_subset)
        # # Remove cells with counts <30 
        # cell_counts               = cell_subset.obs['Sample_Name'].value_counts()
        # samples_with_enough_cells = cell_counts[cell_counts >= 30].index
        # cell_subset               = cell_subset[cell_subset.obs['Sample_Name'].isin(samples_with_enough_cells)]
    
        pb = bulk_data_analysis(cell_subset,cell_type)

        # PC, LOH
        # PT_VCAM1

    # pbs = []
    # # Loop through each sample
    # for sample in cell_subset.obs['Sample_Name'].unique():
    #     samp_cell_subset   = cell_subset[cell_subset.obs['Sample_Name'] == sample].copy()        
    #     samp_cell_subset.X = samp_cell_subset.layers['counts'] #make sure to use raw data
    #     # Sum the counts for each gene, transfer gene names
    #     rep_adata = sc.AnnData(X   = samp_cell_subset.X.sum(axis = 0),
    #                            var = samp_cell_subset.var[[]])
        
    #     rep_adata.obs_names = [sample]
    #     # rep_adata.obs['Sample_Disease'] = samp_cell_subset.obs['Sample_Disease'].iloc[0]
    #     # rep_adata.obs['Sample_Gender']  = samp_cell_subset.obs['Sample_Gender'].iloc[0]
    #     rep_adata.obs['Sample_Ages']    = samp_cell_subset.obs['Sample_Ages'].iloc[0]
    #     pbs.append(rep_adata)  

    # pb = sc.concat(pbs)

    # counts_df = pd.DataFrame(pb.X.astype(np.int32) , columns = pb.var_names) #need to do this to pass var names
    # Convert the expression matrix to a DataFrame
    counts_df = pd.DataFrame(pb.X.astype(np.int32) , index=pb.obs.index, columns=pb.var.index)    
    metadata  = pb.obs.copy()

    condition_list = ['Sample_Ages']
    # Filter out samples with missing data
    for condition in condition_list:        
        samples_to_keep = ~metadata[condition].isna()
        counts_df = counts_df.loc[samples_to_keep]
        metadata = metadata.loc[samples_to_keep]

    dds = DeseqDataSet(
        counts         = counts_df,
        metadata       = metadata,
        design_factors = condition_list,
        refit_cooks=True,
        inference=inference
        )

    sc.pp.filter_genes(dds,min_cells=1)
    dds.deseq2()

    stat_res_70yrs_vs_20yrs = DeseqStats(dds, contrast=["Sample-Ages", "70","20"], inference=inference)
    sc.tl.pca(dds)
    sc.pl.pca(dds, color = 'Sample-Ages', size = 200, title = 'PCA: 70 vs 20 years old, Cell type: PC')


    de = stat_res_70yrs_vs_20yrs.results_df
    de.sort_values('stat', ascending = False)


    stat_res_B_vs_A = DeseqStats(dds, contrast=["Sample-Ages", "60","20"], inference=inference)
    de= stat_res_B_vs_A.results_df
    de.sort_values('stat', ascending = False)

  
    sc.tl.pca(dds)
    sc.pl.pca(dds, color = 'Sample-Ages', size = 200)
    stat_res = DeseqStats(dds, n_cpus=8, contrast=('Sample-Ages', '30', '20', '50', '40', '60', '70'))
    print