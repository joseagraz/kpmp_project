import os
from typing import Tuple
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import AnnData
# import scvi
import omicverse as ov
import scvelo as scv
import logging
import torch
import concurrent.futures
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from scipy.sparse import csr_matrix, issparse
import openai
import cellrank as cr
# ------------------------------------------
# Script Information
__author__ = "Jose L. Agraz, PhD"
__status__ = "Prototype"
__email__ = "jose@agraz.email"
__credits__ = ["Jose L. Agraz", "Parker Wilson"]
__license__ = "MIT"
__version__ = "1.0"
# ------------------------------------------
# reference: 
# Complete single-cell RNAseq analysis walkthrough | Advanced introduction
# https://www.youtube.com/watch?v=uvyG9yLuNSE&t=635s
# ------------------------------------------
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
MINIMUM_GENES      = 500
MINIMUM_CELLS      = 100
RESULTS_DIR           = 'Results'
ROOT_PATH          = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR  = Path('Supporting_Files')
DATA_DIR           = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
METADATA_FILE_PATH = ROOT_PATH / SUPPORT_FILES_DIR / 'source_metadata.csv'
LOOKUP_TABLE       = { 'Healthy Reference': 0,'CKD': 1,'AKI': 2,'DM-R': 3,'None': 4 }
PROCESSED_DIR      = ROOT_PATH / Path('Pre-processed_and_Labeled_Samples')
OUTPUT_DIR         = ROOT_PATH / DATA_DIR / Path('0a0a_Results')
CELLULAR_5_MODEL   = ROOT_PATH / Path('cytotrace2/cytotrace2_python/cytotrace2_py/resources/5_models_weights')
CELLULAR_17_MODEL  = ROOT_PATH / Path('cytotrace2/cytotrace2_python/cytotrace2_py/resources/17_models_weights')
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------
def find_files(root_path:Path, file_name:str) -> List[Path]:
    """
    Find all instances of a file with a given name under the root path.

    Parameters:
    root_path (str): The root directory to start the search from.
    file_name (str): The name of the file to search for.

    Returns:
    list: A list of Paths where the file is found.
    """
    return [file for file in root_path.rglob(file_name)]
# ------------------------------------------   
def read_files_in_parallel(file_paths:List[Path]) -> List[AnnData]:
    """
    Reads multiple .h5ad files in parallel and returns a list of AnnData objects.

    Parameters:
    - file_paths (list of str): A list of paths to .h5ad files to be read.

    Returns:
    - A list of AnnData objects loaded from the provided file paths.
    """
    adata_array = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the file paths to the executor to read in parallel
        results = executor.map(sc.read_h5ad, file_paths)

        columns_to_keep = ['n_genes',           \
                           'n_genes_by_counts', \
                           'total_counts',      \
                           'total_counts_mt',   \
                           'pct_counts_mt',     \
                           'Sample_Name',       \
                           'Sample_Status',     \
                           'Sample_Type',       \
                           'Sample_Tissue_Type',\
                           'Sample_Sex',        \
                           'Sample_Age',        \
                           'Sample_Race',       \
                           'celltype']      
          
        for adata in results:
            # remove obs.columns that are not in columns_to_keep
            columns_to_drop = [col for col in adata.obs.columns if col not in columns_to_keep]
            adata.obs.drop(columns_to_drop, axis=1, inplace=True)
            # remove var.columns that are not in 'columns_to_keep'
            columns_to_drop = [col for col in adata.var.columns if col not in ['n_cells']]
            adata.var.drop(columns_to_drop, axis=1, inplace=True)

            # Convert the 'Sample_Age' column to integers by extracting the first two digits, handling NaN values
            adata.obs['Sample_Age'] = (
                adata.obs['Sample_Age']
                .str.extract(r'(\d{2})')  # Extract the first two digits
                .astype(float)            # Convert to float to handle NaNs
                .fillna(0)                # Replace NaN with 0
                .astype(int)              # Convert back to integer
            )            

            if issparse(adata.X):
                adata.X = adata.X.astype(np.int64)
            else:
                # For dense arrays
                adata.X = np.asarray(adata.X, dtype=np.int64)   
            adata_array.append(adata)
    
    return adata_array
# ------------------------------------------
def poor_cell_and_gene_removal(adata: AnnData) -> Tuple[AnnData, int, int]:
    """
    Removes cells with fewer than a specified number of genes and genes present in fewer than a specified number of cells.

    This function filters out low-quality cells and genes from an AnnData object based on specified thresholds. It is designed to improve the quality of the dataset for further analysis.

    Parameters:
    - adata (AnnData): The AnnData object containing single-cell gene expression data.

    Returns:
    - Tuple[AnnData, int, int]: A tuple containing the filtered AnnData object, the number of cells removed, and the number of genes removed.
    
    """
    # Count the number of cells and genes before filtering
    num_cells_before = adata.n_obs
    num_genes_before = adata.n_vars

    # Filter out cells with fewer than 200 genes and genes found in fewer than 100 cells
    print(f"Filtering genes < {MINIMUM_GENES} and cells < {MINIMUM_CELLS}..")
    sc.pp.filter_cells(adata, min_genes=MINIMUM_GENES)
    sc.pp.filter_genes(adata, min_cells=MINIMUM_CELLS)

    # Calculate the number of cells and genes removed
    num_cells_removed = num_cells_before - adata.n_obs
    num_genes_removed = num_genes_before - adata.n_vars

    print(f"Number of input genes:     {num_genes_before}")
    print(f"Number of genes removed:   {num_genes_removed}")
    print(f"Number of genes remaining: {adata.n_vars}")
    print(f"Number of input cells:     {num_cells_before}")
    print(f"Number of cells removed:   {num_cells_removed}")
    print(f"Number of cells remaining: {adata.n_obs}")
    
    return adata, num_cells_removed, num_genes_removed
# ------------------------------------------
def remove_duplicates(adata):
    data_dense = adata.X.toarray()
    data_df = pd.DataFrame(data_dense)
    duplicates = data_df.duplicated(keep='first')
    print(f"Number of duplicate rows: {duplicates.sum()}")
    unique_mask = ~duplicates.values
    adata = adata[unique_mask].copy()

    obs_duplicates = adata.obs.duplicated()
    print(f"Number of duplicate cells in .obs: {obs_duplicates.sum()}")

    # Check for duplicates in gene metadata
    var_duplicates = adata.var.duplicated()
    print(f"Number of duplicate genes in .var: {var_duplicates.sum()}")    

    # Remove duplicates if needed
    adata = adata[~obs_duplicates].copy()
    adata = adata[:, ~var_duplicates].copy()    
    return adata

        
if __name__ == "__main__":  
    ov.utils.ov_plot_set()  
    # scvi.settings.seed  = 0
    start_time = time.time()
    # print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    
    adata_array       = []
    found_files       = []
    file_of_interest  = 'filtered_feature_bc_matrix.h5_sample-filtered-and-labeled.h5ad'
    number_of_samples_to_keep=174
    
    SCANVI_LATENT_KEY = "Integration_scANVI"
    LEIDEN_RESOLUTION=.6
    input_file = OUTPUT_DIR / Path(f'{number_of_samples_to_keep}_samples_integration_{SCANVI_LATENT_KEY}_images_Leiden_resolution_{LEIDEN_RESOLUTION}')        
    # -------------------------------
    # csv file with valid dataset metadata
    metadata_df  = pd.read_csv(METADATA_FILE_PATH)
    dataset_name = list(set(metadata_df.sample_name.tolist()))

    number_of_samples_to_keep=174
    integrated_samples_file = OUTPUT_DIR / Path(f'{number_of_samples_to_keep}_samples_multiple_integrations_Leiden_resolution_{LEIDEN_RESOLUTION}')    
    adata                  = sc.read_h5ad(integrated_samples_file)
    adata.X = adata.layers['counts']
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.raw = adata.copy()
    # -------------------------------
    adata= remove_duplicates(adata)
    # -------------------------------
    random_indices = np.random.choice(adata.n_obs, 50000, replace=False)
    adata = adata[random_indices].copy() 
    # -------------------------------
    adata.X = adata.X.astype(np.float32)

    

    adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)

    # -------------------------------
    print('scale the adata.X')
    ov.pp.scale(adata)
    print('Dimensionality Reduction')
    ov.pp.pca(adata,layer='scaled',n_pcs=50)
    # -------------------------------
    ov.utils.cluster(adata,use_rep='scaled|original|X_pca',
                    method='GMM',
                    n_components=21)        
    adata.obs['label'] = adata.obs['gmm_cluster']
    sc.tl.tsne(adata, use_rep='X_pca', n_pcs=min(80, adata.obsm['X_pca'].shape[1]))

    adata.obsm['tsne'] = adata.obsm['X_tsne']
    sc.tl.pca(adata, svd_solver='arpack', n_comps=200)
    v0 = ov.single.pyVIA(adata=adata,adata_key='X_pca',adata_ncomps=80, basis='tsne',
                            clusters='label',knn=30,random_seed=4,root_user=[4823],)

    v0.run()
    PT_VCAM1_gene_list_magic = [
    "VCAM1",  # Vascular Cell Adhesion Molecule 1
    "CD133",  # Prominin-1
    "CD24",   # Signal Transducer CD24
    "AQP1",   # Aquaporin-1
    "MCP-1",  # Monocyte Chemoattractant Protein-1
    "TNF-alpha",  # Tumor Necrosis Factor-alpha
    "Nestin",  # Intermediate Filament Protein
    "CADM1"    # Cell Adhesion Molecule 1
    ]
    gene_list_magic = ['HAVCR1', 'VCAM1', 'PAX2', 'Vimentin', 'LCN2', 'HNF4A', 'CD79B', 'SPI1', 'CD34', 'CSF1R', 'ITGAX']
    fig,axs=v0.plot_clustergraph(gene_list=gene_list_magic[:4],figsize=(12,3),)







    ov.utils.embedding( adata,
                        basis='X_umap',
                        color=['gmm_cluster'],
                        frameon='small',
                        wspace=0.5,
                        legend_loc='on data')
    # -------------------------------
    fig, ax = plt.subplots(figsize=(4,4))
    ov.pl.embedding(
        adata,
        basis="X_umap",
        color=['celltype','CytoTRACE2_Score'],
        frameon='small',
        title="Celltypes",
        legend_loc='on data',
        legend_fontsize=14,
        legend_fontoutline=2,
        size=10,
        # ax=ax,
        #legend_loc=True, 
        add_outline=False, 
        #add_outline=True,
        outline_color='black',
        outline_width=1,
        show=False,
    )    
    # -------------------------------
    ov.pl.embedding(
        adata,
        basis="X_umap",
        color=['celltype','CytoTRACE2_Score'],
        frameon='small',
        title="Celltypes",
        legend_loc='on data',
        legend_fontsize=14,
        legend_fontoutline=2,
        size=10,
        # ax=ax,
        # legend_loc=True, 
        add_outline=False, 
        #add_outline=True,
        outline_color='black',
        outline_width=1,
        show=False,
        cmap='Reds'
    )        


    print('Clustering')

