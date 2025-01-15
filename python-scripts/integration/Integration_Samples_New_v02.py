from typing import Tuple
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib.pyplot import rc_context
from pathlib import Path
import scvi
import os
import re
import logging
import torch
import seaborn as sns
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
ROOT_PATH             = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR     = Path('Supporting_Files')
RESULTS_DIR           = 'Results'
DATA_DIR              = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
PLOT_FILE_PATH        = ROOT_PATH / SUPPORT_FILES_DIR / 'Plots'
CLUSTERING_MODEL_PATH = ROOT_PATH / SUPPORT_FILES_DIR / 'scVI_Models'
IMAGING_DIR           = ROOT_PATH / SUPPORT_FILES_DIR / 'Imaging'
METADATA_FILE_PATH    = ROOT_PATH / SUPPORT_FILES_DIR / 'source_metadata.csv'
RIBO_LOOKUP_FILE_PATH = ROOT_PATH / Path(SUPPORT_FILES_DIR) / 'KEGG_RIBOSOME.v2023.2.Hs.txt'
OUTPUT_DIR            = ROOT_PATH / DATA_DIR / Path('0a0a_Results')
# Testing
# samples_of_interest = ROOT_PATH / SUPPORT_FILES_DIR / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
SCVI_LATENT_KEY         = "Integration_scVI"
LEIDEN_RESOLUTION       = 0.4
NORMALIZATION_SCALE     = 1e4
NUMBER_OF_GENES         = 3000
MINIMUM_GENES           = 200
MINIMUM_CELLS           = 3
MITOCHONDRIAL_THRESHOLD = 1
LEIDEN_CLUSTER_NAME     = 'leiden'
INTEGRATED_OUTPUT_FILE  = ROOT_PATH / DATA_DIR / 'integrated.h5ad'
INTEGRATED_OUTPUT_FILE  = ROOT_PATH / SUPPORT_FILES_DIR / 'Data_Integration/Nov_26_2024_integrated.h5ad'
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
def print_quality_control_plots1(df,value):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Paginate Sample_Name values
    samples_per_page=5
    page=1
    unique_samples = df['Sample_Name'].unique()
    start = (page - 1) * samples_per_page
    end = start + samples_per_page

    subset_samples = unique_samples[start:end]

    # Filter the dataframe for only the selected samples
    df_subset = df[df['Sample_Name'].isin(subset_samples)].copy()


    g = sns.FacetGrid(df_subset, row="Sample_Name", hue="Sample_Name", aspect=5, height=0.25, palette="tab20")

    g.map(sns.kdeplot, value, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, value, clip_on=False, color="w", lw=2)

    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, value)

    g.figure.subplots_adjust(hspace=-.6)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    for ax in g.axes.flat:
        ax.axvline(x=df[value].median(), color='r', linestyle='-')

    plt.show()    

# ------------------------------------------  
def print_quality_control_plots(df, value, page=1, samples_per_page=5):
    """
    Plot KDEs for a subset of Sample_Name categories in a DataFrame.

    Parameters:
    - df (DataFrame): The data to plot.
    - value (str): Column name in df for which to plot the KDE.
    - page (int): Page number for pagination (1-indexed).
    - samples_per_page (int): Number of Sample_Name categories per page.

    Displays:
    - KDE plots for the specified subset of Sample_Name categories.
    """
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Ensure valid input
    if value not in df.columns:
        raise ValueError(f"Column '{value}' not found in the DataFrame.")

    # Paginate Sample_Name values
    unique_samples = df['Sample_Name'].unique()
    start = (page - 1) * samples_per_page
    end = start + samples_per_page
    if start >= len(unique_samples):
        raise ValueError("Page number is out of range.")

    subset_samples = unique_samples[start:end]

    # Filter the dataframe for only the selected samples
    df_subset = df[df['Sample_Name'].isin(subset_samples)]

    # Determine the number of rows needed
    num_rows = min(samples_per_page, len(subset_samples))

    # Set an appropriate height for each row so the plot doesn't get too long
    height_per_row = 2  # You can adjust this value as needed for better visibility
    total_height = height_per_row * num_rows

    # Create the FacetGrid with the subset dataframe, with a more square aspect ratio
    g = sns.FacetGrid(df_subset, row="Sample_Name", hue="Sample_Name", height=total_height, aspect=10, palette="tab20")

    # KDE plots for the data
    g.map(sns.kdeplot, value, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, value, clip_on=False, color="w", lw=2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Adding labels
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)
    g.map(label, value)

    # Adjust layout
    g.figure.subplots_adjust(hspace=-0.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add a vertical line at the median value of the overall data
    for ax in g.axes.flat:
        ax.axvline(x=df[value].median(), color='r', linestyle='-')

    # Adding a subtitle to indicate the page
    plt.suptitle(f"Showing page {page} of Sample_Name categories", fontsize=16, y=1.05)

    # Show the plot
    plt.show()
def ensure_unique_indices(adata):
    """
    Ensures that the indices of the AnnData object are unique by appending their position if necessary.
    
    Parameters:
    adata : AnnData
        The AnnData object to modify.
        
    Returns:
    AnnData
        The AnnData object with guaranteed unique indices.
    """
    # Ensure unique cell names
    if not adata.obs_names.is_unique:
        adata.obs_names = [f"{name}-{i}" for i, name in enumerate(adata.obs_names)]
    
    # Ensure unique gene names
    if not adata.var_names.is_unique:
        adata.var_names = [f"{name}-{i}" for i, name in enumerate(adata.var_names)]

# ------------------------------------------
def decimate_adata(adata, decimation_factor):
    """
    Decimates the AnnData object by the specified factor.
    
    Parameters:
    adata : AnnData
        The AnnData object to decimate.
    decimation_factor : int
        The factor by which to decimate the AnnData object.
        
    Returns:
    AnnData
        The decimated AnnData object.
    """
    ensure_unique_indices(adata)  # Ensure indices are unique

    np.random.seed(0)  # For reproducibility
    # Decimate cells
    n_cells = adata.n_obs // decimation_factor
    sampled_cells = np.random.choice(adata.obs_names, n_cells, replace=False)
    
    # Decimate genes
    n_genes = adata.n_vars // decimation_factor
    sampled_genes = np.random.choice(adata.var_names, n_genes, replace=False)
    
    # Subset the adata object
    adata_decimated = adata[sampled_cells, sampled_genes].copy()
    return adata_decimated

def number_cluster_to_match_celltype(adata):
    # Create a mapping from cell type to cluster ID
    cell_types = adata.obs['celltype'].unique()
    celltype_to_cluster = {cell_type: i for i, cell_type in enumerate(cell_types)}
    # Apply the mapping to the 'clusters' column
    adata.obs['clusters'] = adata.obs['celltype'].map(celltype_to_cluster)    
    # Verify the changes
    print(adata.obs[['celltype', 'clusters']])    
    # sc.pl.umap(adata, color=["total_counts", "clusters"], legend_loc='on data',wspace=0.4)
    return adata
# ------------------------------------------
def get_sample_details(sample_name):


    df = pd.read_csv(METADATA_FILE_PATH)
    # Filter the dataframe to find the row matching the given sample_name
    sample_row = df[df['sample_name'] == sample_name]

    # If the sample is found, return the relevant details
    if not sample_row.empty:
        sample_details = sample_row[['Sample_Type', 'Sample_Tissue_Type', 'Sample_Sex', 'Sample_Age', 'Sample_Race']].iloc[0]
        sample_details = sample_details.fillna('None')
        # Convert age string to integer
        sample_details['Sample_Age'] = convert_sample_age(sample_details['Sample_Age'])

        return sample_details.tolist()
    else:
        return "Sample not found."
# ------------------------------------------
def convert_sample_age(age_str):
    # If the age_str is None or indicates missing data, return 0
    if pd.isna(age_str) or age_str == 'None':
        return 0

    # Use regex to find the starting digits
    match = re.match(r"(\d+)-\d+ Years", age_str)
    if match:
        return int(match.group(1))
    
    return 0        
if __name__ == "__main__":    

    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(10, 10))
    torch.set_float32_matmul_precision("high")
    # -------------------------------
    # csv file with valid dataset metadata
    metadata_df  = pd.read_csv(METADATA_FILE_PATH)
    dataset_name = list(set(metadata_df.sample_name.tolist()))
    # ---------------------------
    number_of_samples_to_keep=174
    # file_name              = f'combined_{number_of_samples_to_keep}_samples.h5ad'
    # concatenated_file_name = ROOT_PATH / DATA_DIR / file_name
    concatenated_file_name = ROOT_PATH / Path(DATA_DIR) / Path('0a0a_'+RESULTS_DIR) / f'{number_of_samples_to_keep}_samples_concatenated.h5ad'
    adata                  = sc.read_h5ad(concatenated_file_name)
    adata.raw              = adata
    adata.layers['counts'] = adata.X.copy() 
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    # ---------------------------    
    sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)  
    sc.pp.log1p(adata)
    # ---------------------------
    batch_counts = adata.obs['Sample_Name'].value_counts()
    threshold = 2  # You can adjust this as appropriate
    valid_batches = batch_counts[batch_counts > threshold].index
    adata = adata[adata.obs['Sample_Name'].isin(valid_batches)].copy()

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=3000,
        layer="counts",
        batch_key="Sample_Name",
        subset=True,
    )
    # ---------------------------
    sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
    # We subset to the highly variable genes so that each method has the same input.
    # adata = adata[:, adata.var.highly_variable].copy()
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    # ---------------------------
    number_of_samples = len(adata.obs.Sample_Name.unique())
    print(f"Number of samples found: {len(adata.obs.Sample_Name.unique())}")
    print(f"Processed: {adata.n_obs} observations and {adata.n_vars} features")   
    # ---------------------------
    # from harmony import harmonize

    # adata.obsm["Integration_Harmony"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key="Sample_Name")    
    # ---------------------------
    # import scanorama

    # # List of adata per batch
    # batch_cats = adata.obs.Sample_Name.cat.categories
    # adata_list = [adata[adata.obs.Sample_Name == b].copy() for b in batch_cats]
    # scanorama.integrate_scanpy(adata_list)

    # adata.obsm["Scanorama"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
    # for i, b in enumerate(batch_cats):
    #     adata.obsm["Scanorama"][adata.obs.Sample_Name == b] = adata_list[i].obsm["X_scanorama"]    
    # ---------------------------
    # Integration model
    # print('scVI Integration Setup model')
    scvi.model.SCVI.setup_anndata(adata,
                                  layer                      = 'counts',
                                  batch_key                  = 'Sample_Name',
                                  categorical_covariate_keys = ['Sample_Sex','Sample_Tissue_Type'], 
                                  continuous_covariate_keys  = ['pct_counts_mt','total_counts', 'Sample_Age'])    
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood='nb')
    print('Traning Model')
    model.train() 
    adata.obsm[SCVI_LATENT_KEY]     = model.get_latent_representation()
    # ---------------------------
    SCANVI_LATENT_KEY = "Integration_scANVI"
    
    lvae = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key="celltype",
        unlabeled_category="Unknown",
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)
    adata.obsm[SCANVI_LATENT_KEY] = lvae.get_latent_representation()

    # ---------------------------

    adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size = NORMALIZATION_SCALE)

    # sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
    # print('Creating umap')
    # sc.tl.umap(adata)
    # print('Creating Leiden')
    # LEIDEN_RESOLUTION=.6
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="clusters")     
    # for CPU acceleration vs GPU
    # adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[SCVI_LATENT_KEY], accelerator="cpu")
    # SCVI_MDE_KEY = "X_scANVI_MDE"
    # sc.pl.embedding(
    #     adata,
    #     basis='X_umap',
    #     color=["Sample_Name", "clusters"],
    #     frameon=False,
    #     ncols=1,
    # )    
    # ---------------------------
    print('Calculating neighbors')
    sc.pp.neighbors(adata, use_rep = SCANVI_LATENT_KEY)
    print('Creating umap')
    sc.tl.umap(adata)
    print('Creating Leiden')
    LEIDEN_RESOLUTION=.6
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="clusters") 
    # ---------------------------
    adata = number_cluster_to_match_celltype(adata)
    # sc.pl.umap(adata, color=["total_counts", "clusters"], legend_loc='on data',wspace=0.4, layer='scvi_normalized')
    # sc.pl.umap(adata, color=['celltype'],legend_loc='on data',title='Cell Types',frameon=False, layer='scvi_normalized')
    # sc.pl.embedding(adata, basis="X_umap", color=["celltype"], flegend_loc='on data',rameon=False, ncols=1)
    # --------------------------- plt.savefig(IMAGING_DIR/"umap_plot.png")
    # Sort cluster DCT split into DCT1  and DCT2 (
    dct_cluster_number   = (adata.obs[adata.obs['celltype'] == 'DCT' ]['clusters']).unique()[0]   
    dct1_cluster_number  = (adata.obs[adata.obs['celltype'] == 'DCT1']['clusters']).unique()[0]   
    dct2_cluster_number  = (adata.obs[adata.obs['celltype'] == 'DCT2']['clusters']).unique()[0]   

    adata_cluster_number  = adata[adata.obs['clusters'] == dct_cluster_number].copy()
    sc.pp.neighbors(adata_cluster_number)  # Compute the neighborhood graph
    sc.tl.leiden(adata_cluster_number, key_added='new_clusters', resolution=0.18)  # Adjust resolution based on the data

    # Verify the changes
    # sc.pl.umap(adata_cluster_number, color=["total_counts", "new_clusters"], legend_loc='on data',wspace=0.4, layer='scvi_normalized')
    new_cluster_labels = {'0':dct2_cluster_number, 
                          '1':dct1_cluster_number,
                          '2':dct1_cluster_number,
                          '3':dct2_cluster_number,
                          '4':dct1_cluster_number,
                          '5':dct1_cluster_number,
                          }
    adata_cluster_number.obs['new_clusters'] = adata_cluster_number.obs['new_clusters'].map(new_cluster_labels)

    for new_cluster in set(new_cluster_labels.values()):
        selected_barcodes = adata_cluster_number.obs_names[adata_cluster_number.obs['new_clusters'] == new_cluster]
        adata.obs.loc[selected_barcodes, 'clusters'] = new_cluster

    # catch any strays. usually only one, not clear why this happens
    adata.obs['clusters'].replace(dct_cluster_number, dct2_cluster_number, inplace=True)

    # update celltype
    cluster_to_celltype = {dct2_cluster_number:'DCT2',dct1_cluster_number:'DCT1'}
    adata.obs['celltype'] = adata.obs['clusters'].map(cluster_to_celltype).fillna(adata.obs['celltype'])
    # ---------------------------
    # sc.pl.umap(adata, color=['celltype'],legend_loc='on data',title='Cell Types',frameon=False, layer='scvi_normalized')
    # sc.pl.umap(adata, color=["total_counts", "clusters"], legend_loc='on data',wspace=0.4)
    # ---------------------------
    # Combine clusters 16 ICA & 17 IC, into a single cluster 16 ICA
    ic_cluster_number  = (adata.obs[adata.obs['celltype'] == 'IC']['clusters']).unique()[0]   
    ica_cluster_number = (adata.obs[adata.obs['celltype'] == 'ICA']['clusters']).unique()[0]   
    icb_cluster_number = (adata.obs[adata.obs['celltype'] == 'ICB']['clusters']).unique()[0]  

    adata_cluster_number  = adata[adata.obs['clusters'] == ic_cluster_number].copy()
    sc.pp.neighbors(adata_cluster_number)  # Compute the neighborhood graph     
    sc.tl.leiden(adata_cluster_number, key_added='new_clusters', resolution=0.05)  # Adjust resolution based on the data
    # Verify the changes
    # sc.pl.umap(adata_cluster_number, color=["total_counts", "new_clusters"], legend_loc='on data',wspace=0.4, layer='scvi_normalized')

    new_cluster_labels = {'0':ica_cluster_number, 
                          '1':icb_cluster_number,
                          '2':icb_cluster_number,}    
    adata_cluster_number.obs['new_clusters'] = adata_cluster_number.obs['new_clusters'].map(new_cluster_labels)
    for new_cluster in set(new_cluster_labels.values()):
        selected_barcodes = adata_cluster_number.obs_names[adata_cluster_number.obs['new_clusters'] == new_cluster]
        adata.obs.loc[selected_barcodes, 'clusters'] = new_cluster

    # catch any strays
    adata.obs['clusters'].replace(ic_cluster_number, ica_cluster_number, inplace=True)

    # update celltype
    cluster_to_celltype = {ica_cluster_number:'ICA',icb_cluster_number:'ICB'}
    adata.obs['celltype'] = adata.obs['clusters'].map(cluster_to_celltype).fillna(adata.obs['celltype'])

    # ---------------------------
    # Combine clusters 12 PT_VCAM1 & 7 PTVCAM1, into a single cluster 12 PT_VCAM1

    PTVCAM1_cluster_number  = (adata.obs[adata.obs['celltype'] == 'PTVCAM1']['clusters']).unique()[0]   
    PT_VCAM1_cluster_number = (adata.obs[adata.obs['celltype'] == 'PT_VCAM1']['clusters']).unique()[0]    

    adata.obs['clusters'].replace(PTVCAM1_cluster_number, PT_VCAM1_cluster_number, inplace=True)
    adata.obs['celltype'].replace('PTVCAM1', 'PT_VCAM1', inplace=True)
    # ---------------------------
    # Rename PT to PT_HEALTHY
    # cluster_of_interest = (adata.obs[adata.obs['celltype'] == 'PT']['clusters']).unique()[0]    
    # adata.obs['celltype'].replace('PT', 'PT_HEALTHY', inplace=True)result_file = OUTPUT_DIR / f'{len(found_files)}_samples_concatenated.h5ad'
    # what is gene PT, PT_MT, PT_PROM1, PT_VCAM1, and PT_
    # ---------------------------
    # Change FIB cluster to 
    fib_cluster_number      = (adata.obs[adata.obs['celltype'] == 'FIB']['clusters']).unique()[0]   
    fib_vsmc_cluster_number = (adata.obs[adata.obs['celltype'] == 'FIB_VSMC_MC']['clusters']).unique()[0]   
    adata.obs['clusters'].replace(fib_cluster_number, fib_vsmc_cluster_number, inplace=True)
    adata.obs['celltype'].replace('FIB', 'FIB_VSMC_MC', inplace=True)
    # sc.pl.umap(adata,color='leiden',frameon=False, legend_loc='on data',title='Leiden Clusters')
    # sc.pl.umap(adata, color=["total_counts", "clusters"], wspace=0.4)
    # sc.pl.umap(adata, color=["total_counts", "clusters"], legend_loc='on data',wspace=0.4)
    # sc.pl.umap(adata,color=['celltype'],legend_loc='on data',title='Cell Types',frameon=False, layer='scvi_normalized')
    # ---------------------------
    adata = number_cluster_to_match_celltype(adata)
    # Plot cluster and gene names
    directory_path = IMAGING_DIR / Path(f'{number_of_samples_to_keep}_samples_integration_images_Leiden_resolution_{LEIDEN_RESOLUTION}')
    if not directory_path.exists():
        # Directory does not exist, so create it
        directory_path.mkdir(parents=True)    

    for cell_type in adata.obs['celltype'].unique():
        plt.rcParams["figure.figsize"] = (4, 4)
        # Subset adata to only include cells of the current cell type
        adata_subset = adata[adata.obs['celltype'] == cell_type].copy()
        
        # Use Scanpy to plot UMAP for the subset
        sc.pl.umap(adata_subset, color='celltype', legend_loc='on data',title=f'Cell Type: {cell_type}', show=False)
        
        # Save the figure. Adjust the file path as needed
        file_path = os.path.join(directory_path, f'UMAP_{cell_type}.png')
        plt.savefig(file_path)
        plt.close()  
    
    print(f"Number of cell types found: {len(adata.obs['celltype'].unique())}")
    print(f"{adata.n_obs} observations and {adata.n_vars} features")
    # ---------------------------
    # Check if the file exists and then delete it    
    output_file = OUTPUT_DIR / Path(f'{number_of_samples_to_keep}_samples_integration_{SCANVI_LATENT_KEY}_images_Leiden_resolution_{LEIDEN_RESOLUTION}')        
    if output_file.exists():
        output_file.unlink()   
    print(f'Writing file: {output_file}') 
    adata.write_h5ad(output_file)    
    quit()
    # ---------------------------
    print('Done')

