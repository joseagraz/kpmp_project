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
# 12TB disk path
ROOT_PATH             = Path('/media/jagraz/8TB/KPMP_Data/Privately_Available_Data')
# NAS path
# ROOT_PATH           = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR     = Path('Supporting_Files')
DATA_DIR              = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
PLOT_FILE_PATH        = ROOT_PATH / SUPPORT_FILES_DIR / 'Plots'
CLUSTERING_MODEL_PATH = ROOT_PATH / SUPPORT_FILES_DIR / 'scVI_Models'
IMAGING_DIR           = ROOT_PATH / SUPPORT_FILES_DIR
METADATA_FILE_PATH    = ROOT_PATH / SUPPORT_FILES_DIR / 'metadata.csv'
# Testing
# samples_of_interest = ROOT_PATH / SUPPORT_FILES_DIR / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
SCVI_LATENT_KEY         = "X_scVI"
LEIDEN_RESOLUTION       = 0.4
NORMALIZATION_SCALE     = 1e4
NUMBER_OF_GENES         = 3000
MINIMUM_GENES           = 500
MINIMUM_CELLS           = 100
LEIDEN_CLUSTER_NAME     = 'leiden'
INTEGRATED_OUTPUT_FILE  = ROOT_PATH / DATA_DIR / 'integrated.h5ad'
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------  
def plot_data(adata,plot_type,default_color_map='viridis'):
    num_clusters   = adata.obs[plot_type].nunique()
    colormap       = plt.cm.get_cmap(default_color_map, num_clusters)
    custom_palette = [colormap(i) for i in range(num_clusters)]
    with rc_context({'figure.figsize': (8, 8)}):
        sc.pl.umap(adata, color=plot_type, palette=custom_palette)
    plot_path = PLOT_FILE_PATH / Path(f'Integration_resolution_{LEIDEN_RESOLUTION}_{plot_type}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    return     
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
if __name__ == "__main__":    

    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(16, 16))
    torch.set_float32_matmul_precision("high")
    # -------------------------------
    # csv file with valid dataset metadata
    metadata_df  = pd.read_csv(METADATA_FILE_PATH)
    dataset_name = list(set(metadata_df.sample_name.tolist()))

    # ---------------------------
    concatenated_file_name = ROOT_PATH / DATA_DIR / 'combined_10_samples_with_NO_cellbender.h5ad'
    adata = sc.read_h5ad(str(concatenated_file_name))
    adata.var_names_make_unique()
    # ---------------------------
    adata,_,_ = poor_cell_and_gene_removal(adata)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata,target_sum=NORMALIZATION_SCALE)
    sc.pp.log1p(adata)    
    adata.raw = adata

    # sc.pp.highly_variable_genes(adata, 
    #                             n_top_genes = NUMBER_OF_GENES, 
    #                             subset      = True, 
    #                             layer       = 'counts',                                
    #                             flavor      = 'seurat_v3',
    #                             batch_key   = 'Sample_Name',
    #                             span        = 0.6)

    # Integration model
    print('Setup model')
    # scvi.model.SCVI.setup_anndata(adata,
    #                               layer                      = "counts",
    #                               categorical_covariate_keys = ["Sample_Name"],
    #                               continuous_covariate_keys  = ['pct_counts_mt',    \
    #                                                             'total_counts',     \
    #                                                             'Sample_Gender',    \
    #                                                             'Sample_Ages',      \
    #                                                             'Sample_Disease'])    
    scvi.model.SCVI.setup_anndata(adata,
                                  layer                      = "counts",
                                  categorical_covariate_keys = ["Sample_Name", "Sample_Gender","Sample_Disease"], \
                                  continuous_covariate_keys  = ['pct_counts_mt','total_counts', 'Sample_Ages'])    
    model = scvi.model.SCVI(adata)    
    print(f"Train model")
    print('Traning Model')
    model.train() 

    # Number of cells, columns
    model.get_latent_representation().shape
    adata.obsm[SCVI_LATENT_KEY]     = model.get_latent_representation()
    adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size = NORMALIZATION_SCALE)
    # ---------------------------
    print('Calculating neighbors')
    sc.pp.neighbors(adata, use_rep = SCVI_LATENT_KEY)
    print('Creating umap')
    sc.tl.umap(adata)
    print('Creating Leiden')
    LEIDEN_RESOLUTION=0.8
    sc.tl.leiden(adata, resolution = LEIDEN_RESOLUTION) 
    # ---------------------------
    sc.pl.umap(adata,color='leiden',frameon=False, legend_loc='on data',title='Leiden Clusters')
    # sc.pl.umap(adata,color=['celltype'],legend_loc='on data',title='Cell Types',frameon=False, layer='scvi_normalized')

    directory_path = IMAGING_DIR / Path(f'{len(dataset_name)}_samples_integration_images')
    if not directory_path.exists():
        # Directory does not exist, so create it
        directory_path.mkdir(parents=True)    

    for cell_type in adata.obs['celltype'].unique():
        # Subset adata to only include cells of the current cell type
        adata_subset = adata[adata.obs['celltype'] == cell_type].copy()
        
        # Use Scanpy to plot UMAP for the subset
        sc.pl.umap(adata_subset, color='celltype', legend_loc='on data',title=f'Cell Type: {cell_type}', show=False)
        
        # Save the figure. Adjust the file path as needed
        file_path = os.path.join(directory_path, f'UMAP_{cell_type}.png')
        plt.savefig(file_path)
        plt.close()  


    adata_podo = adata[adata.obs['celltype'] == 'PODO'].copy()
    sc.pl.umap(adata_podo, color=['celltype'], legend_loc='on data', title='Cell Type: PODO', frameon=False, layer='scvi_normalized')


    sc.pl.umap(adata,color=['PODO'],frameon=False, layer='scvi_normalized')

    sc.pl.umap(adata,color='Sample_Name',frameon=False)

    # Check if the file exists and then delete it
    if INTEGRATED_OUTPUT_FILE.exists():
        INTEGRATED_OUTPUT_FILE.unlink()    
    adata.write_h5ad(INTEGRATED_OUTPUT_FILE)


    sc.tl.rank_genes_groups(adata,LEIDEN_CLUSTER_NAME,method='t-test')
    # Plot genes, but tough to look at 
    # sc.pl.rank_genes_groups(adata,n_genes=25,sharey=False)

    # ChatGPT4, https://www.youtube.com/watch?v=fkuLFlC2ZWk&t=914s
    print('Get df of top genes expressions')
    top_n_genes=5
    marker_genes_df=pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(top_n_genes)
    cell_type_dict={
    '0': 'GCP',  # General Cell Population
    '1': 'MC',   # Mesangial Cells or General Cell Population due to non-specific markers
    '2': 'MRC',  # Mitochondria-rich Cells
    '3': 'DCTC', # Distal Convoluted Tubule Cells
    '4': 'TAL',  # Thick Ascending Limb Cells
    '5': 'PC',   # Progenitor Cells
    '6': 'MSC',  # Metabolic Stress Cells
    '7': 'MRC',  # Mitochondria-rich Cells
    '8': 'IC',   # Immune Cells
    '9': 'SMMF', # Smooth Muscle/Myofibroblasts
    '10': 'ICCs', # Acid-Secreting InterCalated Cells
    '11': 'IC',   # Immune Cells
    '12': 'NLC',  # Neuronal or Nerve-like Cells
    '13': 'APC',  # Antigen Presenting Cells
    '14': 'PDC',  # Podocytes
    '15': 'MYC',  # Myocytes or Muscle-like Cells
    '16': 'APC',  # General Cell Population or Amyloid Precursor Cells
    '17': 'EPC'   # Exocrine Pancreatic Cells (Note: needs further investigation in kidney context)
    }

    # Cellbender to corre ct ofr ambient rna  contaminnation

    adata.obs['celltype']=adata.obs[LEIDEN_CLUSTER_NAME].map(cell_type_dict)

    sc.pl.umap(adata,color='celltype',legend_loc='on data',title='Cell Types',frameon=False)

    sc.pl.dotplot(adata, var_names=cell_type_dict, groupby = 'celltype', dendrogram = True)




    # Instead use a dataframe
    markers = sc.get.rank_genes_groups_df(adata, None)
    markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > .5)]

    df = model.differential_expression(groupby = LEIDEN_CLUSTER_NAME)  # takes 45mins

    number_of_genes_of_interest_per_cluster = 4
    automated_markers = {}
    for c in adata.obs.leiden.cat.categories:
        cell_df = df.loc[df.group1 == c]
        automated_markers[c] = cell_df.index.tolist()[:number_of_genes_of_interest_per_cluster]

    sc.pl.dotplot(adata, {'0':automated_markers.pop('0')}, groupby = LEIDEN_CLUSTER_NAME, swap_axes = True,
                use_raw = True, standard_scale = 'var', dendrogram = True)


    model.differential_expression(groupby = "Sample_Name", group1 = '0', group2 = '2')   


    #
    directory_path = CLUSTERING_MODEL_PATH / Path(f"Clustering_Model_{NUMBER_OF_GENES}_top_genes_Leiden_Resolution_{LEIDEN_RESOLUTION}")
    # Check if the file exists
    if directory_path.exists() and directory_path.is_dir():
    # Use shutil.rmtree() to delete the directory and all its contents
        shutil.rmtree(directory_path)
    # Save model 
    model.save(directory_path)
    #
    model.load(directory_path,adata)
    # Number of cells, columns
    model.get_latent_representation().shape
    # Save object representing data
    adata.obsm[SCVI_LATENT_KEY]     = model.get_latent_representation()
    adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size = NORMALIZATION_SCALE)

    sc.tl.rank_genes_groups(adata,LEIDEN_CLUSTER_NAME)
    # Plot genes, but tough to look at 
    # sc.pl.rank_genes_groups(adata,n_genes=20,sharey=False)
    # Instead use a dataframe
    markers = sc.get.rank_genes_groups_df(adata, None)
    markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > .5)]






    markers_scvi = model.differential_expression(groupby = LEIDEN_CLUSTER_NAME)  # takes 45mins
    # filter dataframe
    markers_scvi = markers_scvi[(markers_scvi['is_de_fdr_0.05']) & (markers_scvi.lfc_mean > .5)]
    sc.pl.umap(adata, color = [LEIDEN_CLUSTER_NAME], frameon = False, legend_loc = "on data",legend_fontsize=8,  show=False)

    # Starting with CD45 (Leukocytes)
    sc.pl.umap(adata, color = ['PTPRC', 'CD3E', 'CD8A'], frameon = False, layer = 'scvi_normalized')

    dot_plot_marker_list = markers_scvi.index.tolist()
    dot_plot_marker_list = ["PTPRC","CD3E","MS4A1","CD19","SDC1"]   
    
    # Sort axis
    adata.obs[LEIDEN_CLUSTER_NAME] = pd.Categorical(adata.obs[LEIDEN_CLUSTER_NAME])
    adata.obs[LEIDEN_CLUSTER_NAME] = adata.obs[LEIDEN_CLUSTER_NAME].cat.reorder_categories(
    sorted(adata.obs[LEIDEN_CLUSTER_NAME].unique()), ordered=True) 
    sc.pl.dotplot(adata, dot_plot_marker_list, groupby=LEIDEN_CLUSTER_NAME, dendrogram=False)
    #
    sc.pl.dotplot(adata, dot_plot_marker_list, groupby=LEIDEN_CLUSTER_NAME, dendrogram=True)


    plot_path = PLOT_FILE_PATH / Path(f'Leiden_Clusters_resolution_{LEIDEN_RESOLUTION}_{NUMBER_OF_GENES}_Genes_Parkers_Short_Gene_List.png')
    plt.savefig(plot_path, bbox_inches='tight')

    sc.tl.leiden(adata, resolution = .4)

    adata.var_names[adata.var_names.str.contains('PTPRC', case=False, regex=True)].tolist()

    sc.pl.umap(adata, color = ['EPCAM', 'MUC1'], frameon = False, layer = 'scvi_normalized', vmax = 5)

    markers[markers.names=='RAMP3']

    # Selecting specific values
    markers[(markers.names == 'RAMP3') & (markers.logfoldchanges > 1)]
    sc.pl.umap(adata, color = ['RAMP3'], frameon = False, layer = 'scvi_normalized', vmax = 5)



    print('Done')

