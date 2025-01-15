import scanpy as sc
import pandas as pd
from matplotlib.pyplot import rc_context
import numpy as np
from pathlib import Path
import scvi
import logging
import concurrent.futures
import os
import torch
import tempfile
from collections import Counter
import seaborn as sns
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,
)
# ------------------------------------------
# Script Information
__author__ = "Jose L. Agraz, PhD"
__status__ = "Prototype"
__email__ = "jose@agraz.email"
__credits__ = ["Jose L. Agraz", "Parker Wilson"]
__license__ = "MIT"
__version__ = "1.0"
# ------------------------------------------
sc.set_figure_params(dpi=100)
torch.set_float32_matmul_precision("high")
# 12TB disk path
root_path           = Path('/mnt/12TB_Disk/KPMP_Data/Privately_Available_Data')
# NAS path
# root_path           = Path('/media/jagraz/KPMP_Data/Privately_Available_Data')
support_files_dir   = 'Supporting_Files'
data_dir            = 'Original_Download_KPMP_S3_Bucket_Oct_16_2023'
results_dir         = 'Results'
sample_name         = '0a8d4f18-84ca-4593-af16-3aaf605ca328'
source_data_path    = Path('cellranger_output/outs/filtered_feature_bc_matrix')
data_location       = root_path / Path(data_dir) / Path(sample_name) / source_data_path
MODEL_DIR           = root_path / Path(results_dir) / "scVI-model"
ADATA_FILE_PATH     = root_path / Path(results_dir) / "175_samples.h5ad"
# Testing
# samples_of_interest = root_path / Path(support_files_dir) / 'list_of_samples_processed_using_cellranger_short_list.txt'
# Full list
SAMPLES_OF_INTEREST_DIR = root_path / Path(data_dir)
SCVI_LATENT_KEY         = "X_scVI"
# ------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()])
# ------------------------------------------
def mitochondrial_and_ribo_genes_removal(adata):
    UPPER_QUANTILE   = 0.98
    LOWER_QUANTILE   = 0.02
    MT_COUNT_LIMIT   = 20
    RIBO_COUNT_LIMIT = 2
    ribo_url       = "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt"

    adata.var['mt'] = adata.var_names.str.contains('^MT-', 
                                                   case=False, 
                                                   regex=True)  # annotate the group of mitochondrial genes as 'mt'
    ribo_genes = pd.read_table(ribo_url, skiprows=2, header = None)
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)
    sc.pp.calculate_qc_metrics(adata, 
                               qc_vars=['mt','ribo'], 
                               percent_top=None, 
                               log1p=False, 
                               inplace=True)    
    # Plot results
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

    # sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    #instead of picking subjectively, you can use quanitle
    adata.var.sort_values('n_cells_by_counts')
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, UPPER_QUANTILE)
    lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, LOWER_QUANTILE)
    print(f'{lower_lim} to {upper_lim}')

    adata = adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]
    adata = adata[adata.obs.pct_counts_mt   < MT_COUNT_LIMIT]
    adata = adata[adata.obs.pct_counts_ribo < RIBO_COUNT_LIMIT]

    #normalize every cell to 10,000 UMI
    # sc.pp.normalize_total(adata, target_sum=NORMALIZATION_SCALE) 

    #change to log counts
    # sc.pp.log1p(adata) 
    return(adata)     
# ------------------------------------------
def doublet_removal(adata_original):
    DOUBLETS_PREDICTION = 'prediction'
    DOUBLETS            = 'doublet'
    SINGLET             = 'singlet'

    adata          = adata_original.copy()
    adata_doublets = adata_original.copy()
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes = 3000,
        subset      = True,
        flavor      = "seurat_v3"
    )
    # Model to predict doublets using scVI
    scvi.model.SCVI.setup_anndata(adata)
    doublets_model = scvi.model.SCVI(adata) 
    doublets_model.train()
    # Pass the scVI model to SOLO model
    solo           = scvi.external.SOLO.from_scvi_model(doublets_model)
    solo.train()
    #
    # Convert doublet preditions to dataframe
    df                      = solo.predict()
    df[DOUBLETS_PREDICTION] = solo.predict(soft = False)

    df.groupby('prediction').count()
    # Fine tune doublet labeling, by deleting dif>1 in distribution
    df['dif'] = df.doublet - df.singlet
    # Plot doublet label distribution
    # sns.displot(df[df.prediction == 'doublet'], x = 'dif')
    doublets                      = df[(df.prediction == 'doublet') & (df.dif > 1)]
    adata_doublets.obs['doublet'] = adata_doublets.obs.index.isin(doublets.index)
    adata_doublets                = adata_doublets[~adata_doublets.obs.doublet]
    
    return(adata_doublets)
# ------------------------------------------
def find_duplicate_var_names(adata):
    """
    Find and return duplicate var_names in an AnnData object.

    Parameters:
    adata (AnnData): The AnnData object to check for duplicate var_names.

    Returns:
    list: A list of duplicate var_names.
    """
    # Count occurrences of each var_name
    var_name_counts = Counter(adata.var_names)

    # Filter out var_names that occur more than once
    duplicates = {var_name: count for var_name, count in var_name_counts.items() if count > 1}

    return duplicates
# ------------------------------------------
def read_10x_data(path):
    path
    adata                      = read_mtx(   path / 'matrix.mtx.gz').T
    genes                      = pd.read_csv(path / 'features.tsv.gz', header=None, sep='\t')    
    adata.var_names            = genes[1].values
    adata.var['gene_ids']      = genes[0].values
    adata.var['feature_types'] = genes[2].values
    adata.obs_names            = pd.read_csv(path / 'barcodes.tsv.gz', header=None)[0].values
    return adata
# ------------------------------------------
def process_sample(data_location):

    # adata = read_10x_data(data_location)  
    data_location = data_location.with_suffix('.h5')
    adata = sc.read_10x_h5(data_location)

    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=20) #get rid of genes that are found in fewer than 3 cells

    adata = mitochondrial_and_ribo_genes_removal(adata)

    adata = doublet_removal(adata)

    return(adata)
# ------------------------------------------
def read_and_process_data(sample_name, root_path, data_dir, source_data_path):
    try:
        # logging.info(f"processing sample: {sample_name.strip()}")
        data_location = root_path / Path(data_dir) / Path(sample_name.strip()) / source_data_path
        # logging.info(f"Path: {data_location}")
        adata = process_sample(data_location)   
        adata.obs['Sample_Name'] = sample_name.strip()
        return adata
    except Exception as e:
        logging.error(f"Error processing sample {sample_name}: {e}")
        return None
# ------------------------------------------   
def find_subdirectories(root_path):

    target_path = 'cellranger_output'
    sample_names = []

    for path in root_path.rglob(target_path):
        # Assuming sample name is the directory immediately before 'cellranger_output'
        sample_name = path.parts[-2]
        sample_names.append(sample_name)

    return sample_names 
# ------------------------------------------   
if __name__ == "__main__":    

    NORMALIZATION_SCALE = 1e4
    scvi.settings.seed  = 0
    print("Last run with scvi-tools version:", scvi.__version__)

    sc.set_figure_params(figsize=(4, 4))
    torch.set_float32_matmul_precision("high")
    save_dir = tempfile.TemporaryDirectory()

    adata_array       = []
    sample_names      = find_subdirectories(SAMPLES_OF_INTEREST_DIR)

    # Comment line before after done testing!!!
    # sample_names=[sample_names[0]]
    # sample_names=sample_names[0:3]

    total_samples     = len(sample_names)
    completed_samples = 0

    test=read_and_process_data(sample_names[44], root_path, data_dir, source_data_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sample_name in sample_names:
            logging.info(f"Starting processing for sample {sample_name} ({len(futures) + 1}/{total_samples})")
            future = executor.submit(read_and_process_data, sample_name, root_path, data_dir, source_data_path)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed_samples += 1
            if result is not None:
                adata_array.append(result)
                logging.info(f"Completed processing a sample ({completed_samples}/{total_samples})")
            else:
                logging.error(f"A sample failed to process or returned no data ({completed_samples}/{total_samples})")

    logging.info(f"Concatenating {total_samples} samples")
    adata = sc.concat(adata_array, index_unique='_')

    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

    adata.layers["counts"] = adata.X.copy() 
    #normalize every cell to 10,000 UMI
    sc.pp.normalize_total(adata, target_sum=NORMALIZATION_SCALE) 

    #change to log counts
    sc.pp.log1p(adata) 

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="Sample_Name", # sample, seq tech, gender,
    )

    scvi.model.SCVI.setup_anndata(adata,
                                  layer = "counts",
                                  categorical_covariate_keys=["Sample_Name"],
                                  continuous_covariate_keys=['total_counts'])

    logging.info(f"Define model")
    model = scvi.model.SCVI(adata)    
    logging.info(f"Train model")
    model.train() 

    model.save(MODEL_DIR, overwrite=True)    
    model = scvi.model.SCVI.load(MODEL_DIR, adata=adata)    

    latent = model.get_latent_representation()
    adata.obsm[SCVI_LATENT_KEY] = latent
    latent.shape

    adata.write(ADATA_FILE_PATH)

    # run PCA then generate UMAP plots 
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=100)
    sc.tl.umap(adata, min_dist=0.3)

    sc.pl.umap(
        adata,
        color=["Sample_Name"],
        frameon=False,
    )
    sc.pl.umap(
        adata,
        color=["Sample_Name"],
        ncols=2,
        frameon=False,
    )

    adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size = NORMALIZATION_SCALE)

    sc.pp.neighbors(adata, use_rep = SCVI_LATENT_KEY)

    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution = 0.5)

    with rc_context({'figure.figsize': (8, 8)}):
        sc.pl.umap(adata, color = ['Sample_Name'])

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5) #these are default values

    #save raw data before processing values and further filtering
    # adata.raw = adata 

    #filter highly variable
    adata = adata[:, adata.var.highly_variable] 

    #Regress out effects of total counts per cell and the pe
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt']) 

    #scale each gene to unit variance
    sc.pp.scale(adata, max_value=10) 

    sc.tl.pca(adata, svd_solver='arpack')

    sc.pl.pca_variance_ratio(adata, log=True)

    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.pp.neighbors(adata, n_neighbors=1000, n_pcs=30)
    sc.tl.umap(adata)
    sc.pl.umap(adata)
    sc.tl.leiden(adata, resolution = 0.25)
    sc.pl.umap(adata, color=['leiden'])

    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)    

    print('Done')