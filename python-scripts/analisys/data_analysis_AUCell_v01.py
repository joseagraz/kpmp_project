import requests
import re
from typing import Tuple
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import anndata as ad
from anndata import AnnData
# import scvi
from gget import enrichr
import omicverse as ov
import scvelo as scv
import logging
import torch
import csv
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from scipy.sparse import csr_matrix, issparse
import cellrank as cr
from xml.etree import ElementTree as ET
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
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
#
# Enrichment libraries
# https://maayanlab.cloud/Enrichr/#libraries
# ------------------------------------------
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
RESULTS_DIR           = 'Results'
ROOT_PATH          = Path('/media/KPMP_Data/Privately_Available_Data')
SUPPORT_FILES_DIR  = Path('Supporting_Files')
ENRICHMENT_ANALYSIS_GENE_SETS = ROOT_PATH / SUPPORT_FILES_DIR / Path('enrichment_analysis_gene_set')
DATA_DIR           = Path('Original_Download_KPMP_S3_Bucket_Oct_16_2023')
METADATA_FILE_PATH = ROOT_PATH / SUPPORT_FILES_DIR / 'source_metadata.csv'
LOOKUP_TABLE       = { 'Healthy Reference': 0,'CKD': 1,'AKI': 2,'DM-R': 3,'None': 4 }
PROCESSED_DIR      = ROOT_PATH / Path('Pre-processed_and_Labeled_Samples')
OUTPUT_DIR         = ROOT_PATH / DATA_DIR / Path('0a0a_Results')
CELLULAR_5_MODEL   = ROOT_PATH / Path('cytotrace2/cytotrace2_python/cytotrace2_py/resources/5_models_weights')
CELLULAR_17_MODEL  = ROOT_PATH / Path('cytotrace2/cytotrace2_python/cytotrace2_py/resources/17_models_weights')
CELLPHONE_DB_DIR   = ROOT_PATH / SUPPORT_FILES_DIR / Path('cellphone_db')
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
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
# ------------------------------------------
def get_gene_database_text(pathway_name):
    kgml_url= f'http://rest.kegg.jp/get/{pathway_name}/kgml'

    # Download the KGML file
    print(f"Downloading KGML file from {kgml_url}...")
    response = requests.get(kgml_url)
    response.raise_for_status()  # Raise an error for HTTP issues
    kgml_content = response.text

    # Parse the KGML content
    root = ET.fromstring(kgml_content)
    return root
# ------------------------------------------

def extract_kegg_genes_to_txt(pathway_name):
    """
    Extracts gene IDs and their names from a KEGG KGML file URL and saves them to a CSV file.
    If the output file already exists, the function will bail out.

    Parameters:
        kgml_url (str): URL to the KEGG KGML file.
        output_file (str): Path to the output CSV file for saving gene IDs and names.

    Returns:
        None
    """
    output_file = ENRICHMENT_ANALYSIS_GENE_SETS / f'{pathway_name}_genes.txt'
    # Parse the KGML content
    root = get_gene_database_text(pathway_name)

    # Extract gene IDs and names
    gene_data = []
    for entry in root.findall("entry"):
        hsa_string = entry.get("name")
        gene_names = entry.find("graphics").get("name")
        print(f"Gene ID: {hsa_string}, Gene Name: {gene_names}")
        if gene_names is None or hsa_string == 'undefined':
            continue

        if hsa_string.count(":") > 1:
            parts = [part for part in hsa_string.split() if part.startswith("hsa:")]
            hsa_string=parts[-1] if parts else ""

        if not hsa_string.startswith("hsa:"):
            continue

        gene_names=gene_names.replace("...", "")
        gene_data.append([hsa_string,'\t',gene_names])
    unique_gene_data = [list(item) for item in set(tuple(sublist) for sublist in gene_data)]

    # Save gene data to the CSV file
    with open(output_file, "w") as file:
        for gene in unique_gene_data:
            file.write("\t".join(gene) + "\n")

    print(f"Gene list extracted and saved to {output_file}")

    # kegg_gene_set = pd.read_csv(output_file, sep='\t', header=None)
    return str(output_file)

# ------------------------------------------                
if __name__ == "__main__":  
    ov.utils.ov_plot_set()  
    ov.utils.download_pathway_database()
    ov.utils.download_geneid_annotation_pair()    
    # scvi.settings.seed  = 0
    start_time = time.time()
    # print("Last run with scvi-tools version:", scvi.__version__)
    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    
    adata_array       = []
    found_files       = []
    number_of_samples_to_keep=174
    
    SCANVI_LATENT_KEY = "Integration_scANVI"
    LEIDEN_RESOLUTION=.6
    # -------------------------------
    integrated_samples_file = OUTPUT_DIR / Path(f'{number_of_samples_to_keep}_samples_multiple_integrations_Leiden_resolution_{LEIDEN_RESOLUTION}')    
    adata                  = sc.read_h5ad(integrated_samples_file)
    adata.X = adata.layers['counts']
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.raw = adata.copy()
    # -------------------------------
    adata= remove_duplicates(adata)
    # -------------------------------
    # random_indices = np.random.choice(adata.n_obs, 100000, replace=False)
    # adata = adata[random_indices].copy() 
    # -------------------------------        
    
    selected_cell_types = ['PT','PT_MT','PT_VCAM1','PT_PROM1']
    filtered_adata = adata[adata.obs['celltype'].isin(selected_cell_types)].copy()

    filtered_adata=adata.copy()

    expression_matrix = filtered_adata.X 
    gene_names = filtered_adata.var_names 
    # -------------------------------

    sc.pp.filter_cells(filtered_adata, min_genes=200)
    sc.pp.filter_genes(filtered_adata, min_cells=3)
    adata1=sc.AnnData(filtered_adata.X,obs=pd.DataFrame(index=filtered_adata.obs.index),
                            var=pd.DataFrame(index=filtered_adata.var.index))
    adata1.write_h5ad(ENRICHMENT_ANALYSIS_GENE_SETS/'norm_log.h5ad',compression='gzip')

    df_meta = pd.DataFrame(data={'Cell':list(filtered_adata[adata1.obs.index].obs.index),
                                'celltype':[ i for i in filtered_adata[adata1.obs.index].obs['celltype']]
                                })
    df_meta.set_index('Cell', inplace=True)
    df_meta.to_csv(ENRICHMENT_ANALYSIS_GENE_SETS/'meta.tsv', sep = '\t')

    cpdb_file_path = CELLPHONE_DB_DIR /'cellphonedb.zip'
    meta_file_path = ENRICHMENT_ANALYSIS_GENE_SETS / 'meta.tsv'
    counts_file_path = ENRICHMENT_ANALYSIS_GENE_SETS / 'norm_log.h5ad'
    microenvs_file_path = None
    active_tf_path = None
    out_path = CELLPHONE_DB_DIR / 'test_cellphone'

    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path = cpdb_file_path,                 # mandatory: CellphoneDB database zip file.
        meta_file_path = meta_file_path,                 # mandatory: tsv file defining barcodes to cell label.
        counts_file_path = counts_file_path,             # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
        counts_data = 'hgnc_symbol',                     # defines the gene annotation in counts matrix.
        active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
        microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
        score_interactions = True,                       # optional: whether to score interactions or not. 
        iterations = 1000,                               # denotes the number of shufflings performed in the analysis.
        threshold = 0.1,                                 # defines the min % of cells expressing a gene for this to be employed in the analysis.
        threads = 10,                                     # number of threads to use in the analysis.
        debug_seed = 42,                                 # debug randome seed. To disable >=0.
        result_precision = 3,                            # Sets the rounding for the mean values in significan_means.
        pvalue = 0.05,                                   # P-value threshold to employ for significance.
        subsampling = False,                             # To enable subsampling the data (geometri sketching).
        subsampling_log = False,                         # (mandatory) enable subsampling log1p for non log-transformed data inputs.
        subsampling_num_pc = 100,                        # Number of componets to subsample via geometric skectching (dafault: 100).
        subsampling_num_cells = 1000,                    # Number of cells to subsample (integer) (default: 1/3 of the dataset).
        separator = '|',                                 # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        debug = False,                                   # Saves all intermediate tables employed during the analysis in pkl format.
        output_path = out_path,                          # Path to save results.
        output_suffix = None                             # Replaces the timestamp in the output files by a user defined string in the  (default: None).
        )

    ov.utils.save(cpdb_results, Path(out_path) /'gex_cpdb_test.pkl')
    cpdb_results=ov.utils.load(Path(out_path) /'gex_cpdb_test.pkl')

    interaction=ov.single.cpdb_network_cal(adata = filtered_adata,
        pvals = cpdb_results['pvalues'],
        celltype_key = "celltype",)   

    interaction['interaction_edges'].head()

    ov.plot_set()

    fig, ax = plt.subplots(figsize=(4,4)) 
    ov.pl.cpdb_heatmap(filtered_adata,interaction['interaction_edges'],celltype_key='celltype',
                    fontsize=11,
            ax=ax,legend_kws={'fontsize':12,'bbox_to_anchor':(5, -0.9),'loc':'center left',})


    fig, ax = plt.subplots(figsize=(2,4)) 
    ov.pl.cpdb_heatmap(filtered_adata,interaction['interaction_edges'],celltype_key='celltype',
                    source_cells=selected_cell_types,
            ax=ax,legend_kws={'fontsize':12,'bbox_to_anchor':(5, -0.9),'loc':'center left',})

    ov.pl.cpdb_chord(filtered_adata,interaction['interaction_edges'],celltype_key='celltype',
            count_min=60,fontsize=12,padding=50,radius=100,save=None,)
    fig.show()

    fig, ax = plt.subplots(figsize=(4,4)) 
    ov.pl.cpdb_network(filtered_adata,interaction['interaction_edges'],celltype_key='celltype',
                counts_min=60,
                nodesize_scale=5,
                    ax=ax)    
    
    ov.single.cpdb_plot_network(adata=filtered_adata,
                    interaction_edges=interaction['interaction_edges'],
                    celltype_key='celltype',
                    nodecolor_dict=None,title='EVT Network',
                    edgeswidth_scale=25,nodesize_scale=10,
                    pos_scale=1,pos_size=10,figsize=(6,6),
                    legend_ncol=3,legend_bbox=(0.8,0.2),legend_fontsize=10)    



    ov.pl.cpdb_interacting_network(adata=adata,
                            celltype_key='celltype',
                                means=cpdb_results['means'],
                                source_cells=['PT'],
                                target_cells=['PT_MT','PT_VCAM1','PT_PROM1'],
                                means_min=1,
                                means_sum_min=1,        
                                nodecolor_dict=None,
                                ax=None,
                                figsize=(6,6),
                                fontsize=10)








    gene_list = str(ENRICHMENT_ANALYSIS_GENE_SETS / "gene_list.txt")

    gene_names.to_series().to_csv(gene_list, index=False, header=False)

    # -------------------------------    
    kegg_gene_set = enrichr("gene_list.txt", str(ENRICHMENT_ANALYSIS_GENE_SETS/"KEGG_2021_Human.txt"))

    pathway_dict=ov.utils.geneset_prepare(kegg_gene_set,organism='Human')    
    # -------------------------------
    random_indices = np.random.choice(adata.n_obs, 50000, replace=False)
    adata = adata[random_indices].copy() 
    # -------------------------------
    adata.X = adata.X.astype(np.float32)

    sc.pp.normalize_total(PT_MT, target_sum=1e4)
    sc.pp.log1p(PT_MT)    

    ##Assest one geneset
    geneset_names=list(pathway_dict.keys())
    ov.single.geneset_aucell(PT_MT,
                                geneset_name=geneset_names,
                                geneset=pathway_dict)
    sc.pl.embedding(adata,
                basis='umap',
                color=[i+'_aucell' for i in geneset_names])
    

    adata_aucs=ov.single.pathway_aucell_enrichment(adata,
                                                  pathways_dict=pathway_dict,
                                                  num_workers=22)





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

