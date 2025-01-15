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

def plot_gene_expression(df_ranked_genes,top_n,cell_type_of_interest,sample_sex):
    print(f'plot\nCell type: {cell_type_of_interest},\ntop:{top_n} genes:')
    # Filter out rows where 'score' is negative
    df_positive_scores = df_ranked_genes[df_ranked_genes['score'] > 0]
    # Sort the DataFrame by 'score' in descending order
    df_sorted = df_positive_scores.sort_values(by='score', ascending=False)    
    # Select top N genes for visualization
    top_genes = df_sorted.head(top_n).index.tolist()  # Adjust N as needed

    results = []
    
    subset = adata[(adata.obs['celltype']==cell_type_of_interest) & (adata.obs['Sample_Sex']==sample_sex)]

    # Extract data per gene and age group
    for gene in top_genes:

        # Get expression values
        expression_values = subset[:, gene].X.toarray().flatten()
        expression_values = expression_values[expression_values > 0.1]
        
        # Append to results
        for value in expression_values:
            results.append({
                'gene': gene,
                'sex': sample_sex,
                'expression_value': value,
                'celltype': cell_type_of_interest
            })
        if len(expression_values) > 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(expression_values)), expression_values, alpha=0.6)
            x = np.arange(len(expression_values))
            fit = np.polyfit(x, expression_values, deg=1)
            trend_line = np.polyval(fit, x)
            plt.plot(x, trend_line, color='red', label='Trend Line')        
            plt.title(f'Expression values for {gene} in {cell_type_of_interest}')
            plt.xlabel('Cell Index')
            plt.ylabel('Expression Value')
            plt.grid(True)
            plt.show()        
        print

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/media/jagraz/8TB/KPMP_Data/Privately_Available_Data/Supporting_Files/Plots/parker/csv_files/gene_expression_{cell_type_of_interest}_{sample_sex}_per_age.csv', index=False)

# ------------------------------------------
def dotplot_gene_expression(df_ranked_genes,top_n,cell_type_of_interest,gender):
    print(f'Dot plot\nCell type: {cell_type_of_interest},\ntop:{top_n} genes:')
    # Filter out rows where 'score' is negative
    df_positive_scores = df_ranked_genes[df_ranked_genes['score'] > 0]
    # Sort the DataFrame by 'score' in descending order
    df_sorted = df_positive_scores.sort_values(by='score', ascending=False)    
    # Select top N genes for visualization
    top_genes = df_sorted.head(top_n).index.tolist()  # Adjust N as needed

    # Filter adata_pt to include only the selected top genes
    adata_pt= adata[adata.obs.Sample_Sex==gender]
    adata_pt= adata_pt[adata_pt.obs.celltype==cell_type_of_interest]

    adata_pt_top_genes = adata_pt[:, adata_pt.var_names.isin(top_genes)]

    dotplot_file_name = plot_dir / f'{cell_type_of_interest}_top_{top_n}_genes_dotplot.png'
    # Create the dot plot for gene expression per age group
    ax=sc.pl.dotplot(adata_pt_top_genes, 
                  var_names=top_genes, 
                  groupby='age_group', 
                  standard_scale='var', 
                  color_map='viridis', 
                  dot_max=0.7, 
                  dot_min=0.2,
                  title=f'{cell_type_of_interest}: {gender} Gene Expression per Age Group',
                  )
    
    fig = plt.gcf()  
    fig.savefig(dotplot_file_name, bbox_inches='tight', dpi=300)
    plt.show()  # Close the plot figure to free up memory 
     
# ------------------------------------------
def barplot_gene_expression(df, top_n, cell_type_of_interest):
    print(f'Bat plot\nCell type: {cell_type_of_interest},\ntop:{top_n} genes:')

    # Filter the top N genes based on score
    top_genes = df.sort_values(by=['score'], ascending=False).head(top_n)
    top_genes = top_genes.sort_values(by=['group'], ascending=False)
    
    # Create a figure and axis object with matplotlib
    plt.figure(figsize=(12, 6))
    
    # Create a bar plot
    sns.barplot(x=top_genes.index, 
                y='logfoldchange', 
                hue='group', 
                data=top_genes, 
                dodge=False)
    
    # Adding labels and title
    plt.xlabel('Genes')
    plt.ylabel('Log Fold Change')
    plt.title(f'{cell_type_of_interest}: KPMP, 175 samples, Gene Expression Changes Per Age Group')
    plt.xticks(rotation=45)  # Rotate the gene names for better readability
    
    # Display legend
    plt.legend(title='Age Group', loc='upper right')
    
    # Show the plot
    # plt.show()
    barplot_file_name = plot_dir / f'{cell_type_of_interest}_top_{top_n}_genes_barplot.png'
    plt.savefig(barplot_file_name, format='png', dpi=300)  # Save the plot as a PNG file with high resolution
    plt.close()  # Close the plot figure to free up memory       

def stacked_violin_gene_expression(df_ranked_genes, top_n, cell_type_of_interest):
    print(f'Stacked Violin Plot\nCell type: {cell_type_of_interest},\nTop: {top_n} genes:')
    
    # Filter out rows where 'score' is negative
    df_positive_scores = df_ranked_genes[df_ranked_genes['score'] > 0]
    
    # Sort the DataFrame by 'score' in descending order
    df_sorted = df_positive_scores.sort_values(by='score', ascending=False)
    
    # Select top N genes for visualization
    top_genes = df_sorted.head(top_n).index.tolist()

    # Filter adata to include only the selected top genes and specific cell type
    adata_pt_top_genes = adata[:, adata.var_names.isin(top_genes)]
    adata_pt_top_genes = adata_pt_top_genes[adata_pt_top_genes.obs['celltype'] == cell_type_of_interest]

    # Create the stacked violin plot for gene expression per age group
    ax = sc.pl.stacked_violin(adata_pt_top_genes, 
                              var_names=top_genes, 
                              groupby='age_group', 
                              swap_axes=False,  # Change to True if you prefer swapped axes
                              standard_scale='var', 
                            #   color_map='viridis', 
                              title=f'{cell_type_of_interest}: Gene Expression per Age Group')
    
    violinplot_file_name = plot_dir / f'{cell_type_of_interest}_top_{top_n}_genes_stacked_violin.png'
    # Save the plot
    fig = plt.gcf()  # Get the current figure
    fig.savefig(violinplot_file_name, bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot figure to free up memory

def merge_dataframes(df_list):
    # Reset the index for each dataframe in the list to make 'gene' a column
    reset_dfs = [df.reset_index() for df in df_list]

    # Concatenate all dataframes along the columns
    merged_df = pd.concat(reset_dfs, axis=1)

    # Optional: Handle duplicate columns if necessary. This code assumes 'gene' is the column with gene names.
    # Assuming 'gene' column repeats, we can either drop duplicates or handle them in a specific way.
    # Here, we keep the first 'gene' column and remove subsequent ones that are duplicates.
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df = merged_df[merged_df['score'] > 1]

    return merged_df   

def filter_common_genes(df):
    # Group the dataframe by 'Cell_Type' and collect unique genes for each type
    grouped = df.groupby('Cell_Type')['gene'].apply(set)

    # Find the intersection of all sets of genes to get common genes across all Cell_Types
    common_genes = set.intersection(*grouped)

    # Filter the dataframe to only include rows with genes that are common to all Cell_Types
    filtered_df = df[df['gene'].isin(common_genes)]

    filtered_df = filtered_df[filtered_df['score'] > 10]

    return filtered_df 

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

    # cell_type_of_interest = 'PT'
    number_of_significant_genes = 30
    # ---------------------------    
    adata          = sc.read_h5ad(output_file)

    adata.X      = adata.layers['counts']
    adata.var_names_make_unique()
    adata.obs_names_make_unique()    
    
    # # Normalize the data to total counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    # # Logarithmize the data
    sc.pp.log1p(adata)
    
    # Convert and specify that the categories should be ordered
    # Ensure Sample_Ages is in an appropriate numeric format
    # adata.obs['Sample_Ages'] = adata.obs['Sample_Ages'].astype(str)
    age_categories = sorted(adata.obs['Sample_Ages'].unique())
    adata.obs['Sample_Ages'] = pd.Categorical(adata.obs['Sample_Ages'], categories=age_categories, ordered=True)
    
    # Define bins for the age groups; include a small buffer for the lower and upper edges if necessary
    bins = [19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5]
    # Labels for bins that correspond to the actual ages
    labels = ["20", "30", "40", "50", "60", "70"]

    # Apply pd.cut to categorize ages across all samples
    adata.obs['age_group'] = pd.cut(adata.obs['Sample_Ages'], bins=bins, labels=labels, right=True)
    adata.obs['age_group'] = adata.obs['age_group'].astype(str)

    # Ensure age_group is categorical and ordered
    adata.obs['age_group'] = pd.Categorical(adata.obs['age_group'], categories=labels, ordered=True)

    cell_types = adata.obs['celltype'].unique()
    list_of_dfs = []
    for cell_type_of_interest in cell_types:

        print(f'Processing Cell type of interest: {cell_type_of_interest}')

        plot_dir = PLOT_FILE_PATH / cell_type_of_interest 
        # Create the directory
        plot_dir.mkdir(parents=True, exist_ok=True)         

        # Subset to only 'PT' cells after adding 'age_group'
        adata_pt = adata[adata.obs['celltype'] == cell_type_of_interest].copy()

        # Now perform differential expression analysis
        sc.tl.rank_genes_groups(adata_pt, 'age_group', method='wilcoxon')

        # Visualize the results
        # sc.pl.rank_genes_groups(adata_pt, n_genes=number_of_significant_genes, sharey=False)
        # glidplot_file_name = plot_dir / f'{cell_type_of_interest}_top_{number_of_significant_genes}_genes_gridplot.png'
        # fig = plt.gcf() 
        # fig.savefig(glidplot_file_name, bbox_inches='tight', dpi=300)
        # plt.close() 

        # Access the ranked genes results
        print(adata_pt.uns['rank_genes_groups'].keys())  # Shows available keys
        # Sort adata_pt.obs by 'age_group' in ascending order
        adata_pt.obs.sort_values(by='age_group', inplace=True)
        
        # Create a dot plot for the top 10 ranked genes
        top_genes = adata_pt.uns['rank_genes_groups']['names'][0:10]
        # sc.pl.dotplot(adata_pt, var_names=top_genes, groupby='age_group', use_raw=True)


        # Initialize data structure
        ranked_genes = adata_pt.uns['rank_genes_groups']
        group_names = ranked_genes['names'].dtype.names
        data = []

        # Iterate through each group and extract the gene data
        for group in group_names:
            group_data = []
            for i in range(ranked_genes['names'][group].shape[0]):
                gene_data = {
                    'group':         group,
                    'gene':          ranked_genes['names'][group][i],
                    'score':         ranked_genes['scores'][group][i],
                    'pval':          ranked_genes['pvals'][group][i],
                    'logfoldchange': ranked_genes['logfoldchanges'][group][i],
                    'pval_adj':      ranked_genes['pvals_adj'][group][i],
                    'celltype':      cell_type_of_interest
                }
                data.append(gene_data)
            
        # Create a DataFrame from the gathered data
        df_ranked_genes = pd.DataFrame(data)
        df_ranked_genes = df_ranked_genes.sort_values('score', ascending=False).reset_index(drop=True)
        df_ranked_genes.set_index('gene', inplace=True)
        
        print(df_ranked_genes.head())

        # bar plot
        number_of_genes = 20
        gender='Male'

        dotplot_gene_expression(df_ranked_genes,number_of_genes, cell_type_of_interest,gender)
        plot_gene_expression(df_ranked_genes,number_of_genes, cell_type_of_interest,gender)
        # barplot_gene_expression(df_ranked_genes,number_of_genes, cell_type_of_interest)
        # dotplot_gene_expression(df_ranked_genes,number_of_genes, cell_type_of_interest,gender)
        # stacked_violin_gene_expression(df_ranked_genes, number_of_genes, cell_type_of_interest)
  
        df_ranked_genes['Cell_Type'] = cell_type_of_interest
        list_of_dfs.append(df_ranked_genes)
    print('Concatenating dataframes')

    all_df_ranked_genes        = merge_dataframes(list_of_dfs)
    all_df_ranked_common_genes = filter_common_genes(all_df_ranked_genes)
   

    print
