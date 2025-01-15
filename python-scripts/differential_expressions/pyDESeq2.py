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
MINIMUM_GENES           = 500
MINIMUM_CELLS           = 200
LEIDEN_CLUSTER_NAME     = 'leiden'
INTEGRATED_OUTPUT_FILE  = ROOT_PATH / DATA_DIR / 'integrated.h5ad'
LOOKUP_TABLE            = { 'Healthy Reference': 0,'CKD': 1,'AKI': 2,'DM-R': 3,'None': 4 }
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------  
def filter_low_count_cells(adata: AnnData, minimum_number_of_cells: int=30) -> AnnData: 
    cell_counts  = adata.obs['Sample_Name'].value_counts()
    good_samples = cell_counts[cell_counts >= minimum_number_of_cells].index
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
# ------------------------------------------
def a_thing(x):
    if x < 0:
        return 1
    else:
        return x
    
def save_dataframes_side_by_side(dataframes, filename):
  """
  Saves a list of DataFrames side-by-side to a single CSV file.

  Args:
      dataframes: A list of pandas DataFrames.
      filename: The name of the output CSV file.
  """
  with open(filename, 'w', newline='') as f:
    # Write header row
    header = ','.join([df.columns[0] for df in dataframes])
    f.write(header + '\n')

    # Write data rows, iterating through each dataframe
    for df in dataframes:
      df.to_csv(f, sep=',', header=False, index=False)   

    num_rows = dataframes[0].shape[0]

    return all(df.shape[0] == num_rows for df in dataframes[1:])

# ------------------------------------------    
if __name__ == "__main__":    

    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(8, 8))
    torch.set_float32_matmul_precision("high")
    inference = DefaultInference(n_cpus=20)
    # ---------------------------
    number_of_samples_to_keep = 175
    directory_path = IMAGING_DIR / Path(f'{number_of_samples_to_keep}_samples_integration_images_Leiden_resolution_{LEIDEN_RESOLUTION}')
    output_file    = directory_path / INTEGRATED_OUTPUT_FILE
    adata          = sc.read_h5ad(output_file)
    adata.X        = adata.layers['counts']
    # sc.pl.umap(adata, color=["total_counts", "clusters"], legend_loc='on data',wspace0.4)
    # sc.pl.umap(adata,color=['celltype'],legend_loc='on data',title='Cell Types',frameon=False, layer='scvi_normalized')
    print(f'Integrated dataset shape: {adata.shape}')

    # Subset data      
    cell_types  = adata.obs['celltype'].unique().tolist()  # All cell types
    cluster_list = adata.obs['clusters'].unique().tolist()

    dataframe_list=[]
    for cluster_number in cluster_list:
        cell_type      = adata.obs[adata.obs['clusters'] == cluster_number]['celltype'].unique()[0]
        print(f'Cell type: {cell_type}, Cluster: {cluster_number}')

        # make directories
        output_path    = PLOT_FILE_PATH / Path(f'cluster_{cluster_number}_{cell_type}')
        output_path.mkdir(parents=True, exist_ok=True)
     
        cell_subset = adata[(adata.obs['celltype'] == cell_type)].copy() 
        # cell_subset = adata[(adata.obs['clusters']       == cluster_number) & 
        #                     (adata.obs['Sample_Disease'] == LOOKUP_TABLE['Healthy Reference']) |
        #                     (adata.obs['Sample_Disease'] == LOOKUP_TABLE['CKD']) |  
        #                     (adata.obs['Sample_Disease'] == LOOKUP_TABLE['AKI']) | 
        #                     (adata.obs['Sample_Disease'] == LOOKUP_TABLE['DM-R'])| 
        #                     (adata.obs['Sample_Disease'] == LOOKUP_TABLE['None'])
        #                     ].copy()

        print(f'Cell subset before shape: {cell_subset.shape}')
        cell_subset = filter_low_count_cells(cell_subset)
        print(f'Cell subset after shape: {cell_subset.shape}')

        # pathway enrichment analisys or gene set enrichment analysis
        # for R -> cluster profiler see python equivalent
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

        pb = sc.concat(pbs)

        counts_df = pd.DataFrame(pb.X.astype(np.int32) , index=pb.obs.index, columns=pb.var.index)    
        metadata  = pb.obs.copy()

        condition_list = ['Sample_Ages']
        # Filter out samples with missing data
        for condition in condition_list:        
            samples_to_keep = ~metadata[condition].isna()
            counts_df       = counts_df.loc[samples_to_keep]
            metadata        = metadata.loc[samples_to_keep]

        dds = DeseqDataSet(
            counts         = counts_df,
            metadata       = metadata,
            design_factors = condition_list,
            refit_cooks    = True,
            inference      = inference
            )
        dds.deseq2()

        Max_Age = str(dds.obs['Sample-Ages'].astype(np.int32).max())
        Min_Age = str(dds.obs['Sample-Ages'].astype(np.int32).min())

        stat_res = DeseqStats(dds, contrast=["Sample-Ages", Max_Age, Min_Age], inference=inference)
        stat_res.summary()
        res_df   = stat_res.results_df
        # Add Gene name column
        res_df.index.rename('Gene', inplace=True)
        # Only genes that are of value
        res_df   = res_df[res_df.baseMean >= 0]
        res_df   = res_df.reset_index()
        # Only significant values    
        # sigs_df  = res_df[(res_df.padj < 0.05) & (abs(res_df.log2FoldChange) > 0.5)]
        sigs_df  = res_df[(res_df.padj < 2) ]
        print(sigs_df.head())
        # Add Gene Rank column
        sigs_df['Rank'] = -np.log10(sigs_df.padj)*sigs_df.log2FoldChange
        # Sort by Rank
        sigs_df         = sigs_df.sort_values('Rank', ascending = False).reset_index(drop = True)
        # New dataframe with only Gene and Rank columns
        ranking = sigs_df[['Gene', 'Rank']]

        dataframe_list.append(ranking)

    dataframes_prefixed = [
    df.add_prefix(f"cluster_{i}_")
    for i, df in enumerate(dataframe_list)
    ]

    concatenated_df = pd.concat(dataframes_prefixed, axis=1)
    concatenated_df.to_csv(PLOT_FILE_PATH / 'rankings.csv', index=False)

    # user set of genes
    # ranking['weights'] = ranking.Rank.map(a_thing)
    # a_list = random.choices(ranking.Gene.values, k = 100, weights = ranking.weights.values)
    # user_set = {'A' : a_list}
    if not len(ranking):
        print('No significant genes found. Cluster:{cluster_number}, Cell type:{cell_type}')
        quit()

    pre_res = gp.prerank(rnk             = ranking.set_index('Gene'), 
                         gene_sets       = 'GO_Biological_Process_2021', 
                         seed            = 6,
                         verbose         = True, # see what's going on behind the scenes
                         min_size=1, # ignore sets smaller than 10
                         )
    # ------------------------------------------

    plot_title = f'Cell type:{cell_type}, Cluster:{cluster_number}'
    ax = dotplot(pre_res.res2d,
                column="FDR q-val",
                title=plot_title,
                cmap=plt.cm.viridis,
                size=6, # adjust dot size
                figsize=(15,8), 
                cutoff=0.25, 
                show_ring=False)
    fig = ax.figure 
    fig.tight_layout()
    fig.savefig(output_path / 'gsea_dotplot.png')  # Saves the plot as a PNG file
    # ------------------------------------------
    terms = pre_res.res2d.Term
    axs   = pre_res.plot(terms=terms[1]) # v1.0.5
    fig   = axs.figure 
    fig.set_size_inches(10, 8) 
    fig.tight_layout()

    fig.savefig(output_path / 'gsea_pathway_plot.png')  # Saves the plot as a PNG file
    # ------------------------------------------
    terms = pre_res.res2d.Term
    axs = pre_res.plot(terms[:5], show_ranking=False, legend_kws={'loc': (.05, 0)}, )
    fig   = axs.figure 
    fig.set_size_inches(20, 8) 
    fig.suptitle(plot_title, fontsize=16)
    fig.tight_layout()    
    fig.savefig(output_path / 'gsea_hill_plot.png')  # Saves the plot as a PNG file    
    # ------------------------------------------
    # Network Visualization
    nodes, edges = enrichment_map(pre_res.res2d)
    # build graph
    G = nx.from_pandas_edgelist(edges,
                                source='src_idx',
                                target='targ_idx',
                                edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes'])
    # fig, ax = plt.subplots(figsize=(8, 8))
    # init node cooridnates
    pos     = nx.layout.spiral_layout(G)
    #node_size = nx.get_node_attributes()
    # draw node
    nx.draw_networkx_nodes(G,
                           pos        = pos,
                           cmap       = plt.cm.RdYlBu,
                           node_color = list(nodes.NES),
                           node_size  = list(nodes.Hits_ratio *1000))
    # draw node label
    nx.draw_networkx_labels(G,
                            pos    = pos,
                            labels = nodes.Term.to_dict())    
    edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
    nx.draw_networkx_edges(G,
                           pos   = pos,
                           width = list(map(lambda x: x*10, edge_weight)),
                           edge_color='#CDDBD4')
    # plt.show()
    # fig.suptitle(plot_title, fontsize=16)
    # fig.tight_layout()    
    plt.savefig(output_path / 'gsea_network_plot.png', format="png", dpi=300)    
    quit()




    i = 2
    genes = pre_res.res2d.Lead_genes[i].split(";")
    # Make sure that ``ofname`` is not None, if you want to save your figure to disk
    ax = heatmap(df = pre_res.heatmat.loc[genes], z_score=0, title=terms[i], figsize=(14,4))
    fig   = ax.figure 
    fig.set_size_inches(20, 8) 
    fig.tight_layout()    
    fig.savefig(PLOT_FILE_PATH / 'gsea_heatmap_plot.png')  # Saves the plot as a PNG file



# generate gene list
# feed that to


# scanpy ran