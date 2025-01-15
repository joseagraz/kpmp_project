import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import SpatialDE
import NaiveDE
# Tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
# Introduction to spatial sequencing data analysis
# https://www.youtube.com/watch?v=86uR01mwLIQ&t=1s

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

adata = sc.read_visium('/home/jagraz/Downloads/Visium_data')
# adata spots x gene names
adata.var_names_make_unique()

# Images located at:
adata.uns['spatial']['MGI3779_D1_070722_446AJGE']['images']['hires']
# Each row correcponds of a barcode. it will be filtered
adata.obsm['spatial']
# adding a column to look at where the spots are mapped on the tissue
adata.obs['thing'] = 'a'
# plot tissue vs spots gene probes
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, color = 'thing')

# count mitocondira reads. THere may not be any mitochondria reads
adata.var["mt"] = adata.var_names.str.startswith("MT")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])

plt.show()

sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 2000], kde=False, bins=40)
sc.pp.filter_cells(adata, min_counts = 1000)
sc.pp.filter_cells(adata, max_counts=40000) #based on the following QC this should be reduced
adata = adata[adata.obs["pct_counts_mt"] < 20]

sc.pp.filter_genes(adata, min_cells=3)

# plot tissue vs spots gene probes
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, color = 'thing')

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "leiden"], wspace=0.4)

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"])

sc.pl.spatial(adata, img_key="hires", color="leiden", size=1.5)

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

#convert to dataframe
results = adata.uns['rank_genes_groups']
('0', '1', '2', '3', '4')

out = np.array([[0,0,0,0,0]])
for group in results['names'].dtype.names:
    out = np.vstack((out, np.vstack((results['names'][group],
                                     results['scores'][group],
                                     results['pvals_adj'][group],
                                     results['logfoldchanges'][group],
                                     np.array([group] * len(results['names'][group])).astype('object'))).T))



markers = pd.DataFrame(out[1:], columns = ['Gene', 'scores', 'pval_adj', 'lfc', 'cluster'])

markers = markers[(markers.pval_adj < 0.05) & (abs(markers.lfc) > 1)]

markers[markers.cluster == '2']

sc.pl.rank_genes_groups_dotplot(adata,n_genes=10)
sc.pl.rank_genes_groups_heatmap(adata, groups="5",n_genes=25, groupby="leiden")
sc.pl.spatial(adata, img_key="hires", color=["SPP1","IGFBP7"], ncols=2)

counts      = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
sample_info = pd.DataFrame(adata.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata.obs_names)
sample_info['total_counts'] = adata.obs.total_counts.values.astype(int)

norm_expr = NaiveDE.stabilize(counts.T).T
resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T

sample_resid_expr = resid_expr.sample(n=1000, axis=1, random_state=1)

X = sample_info[['x', 'y']].values
results = SpatialDE.run(X, sample_resid_expr)

results.sort_values('qval').head(10)[['g', 'l', 'qval']]

# pattern 0
for i, g in enumerate(['NPNT', 'EIF3L', 'GJA1']):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=norm_expr[g])
    plt.title(g)
    plt.axis('equal')
    plt.colorbar(ticks=[])
# pattern 1
for i, g in enumerate(['GANAB', 'TCP1', 'ARMCX3']):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=norm_expr[g])
    plt.title(g)
    plt.axis('equal')
    plt.colorbar(ticks=[])

for i, g in enumerate(['VIM', 'PODN', 'GYPC']):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=norm_expr[g])
    plt.title(g)
    plt.axis('equal')
    plt.colorbar(ticks=[])    

#          g           l  qval
# 517  TRNP1  228.796708   0.0
# 569  MSRB1  421.233371   0.0
# 573  KITLG  421.233371   0.0
# 575  ALDOB  421.233371   0.0
# 528   TRAC  228.796708   0.0
# 589  ITGB2  421.233371   0.0
# 592  G3BP2  421.233371   0.0
# 594  CCND2  421.233371   0.0
# 563  HYOU1  421.233371   0.0
# 566  SFRP4  421.233371   0.0

plt.yscale('log')
plt.scatter(results['FSV'], results['qval'], c='black')
plt.axhline(0.05, c='black', lw=1, ls='--')
plt.gca().invert_yaxis()
plt.xlabel('Fraction spatial variance')
plt.ylabel('Adj. P-value')

# Automatic expression histology

sign_results = results.query('qval < 0.05')
sign_results['l'].value_counts()
histology_results, patterns = SpatialDE.aeh.spatial_patterns(X, resid_expr, sign_results, C=3, l=1.8, verbosity=1)

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=patterns[i])
    plt.axis('equal')
    plt.title('Pattern {} - {} genes'.format(i, histology_results.query('pattern == @i').shape[0] ))
    plt.colorbar(ticks=[])

for i in histology_results.sort_values('pattern').pattern.unique():
    print('Pattern {}'.format(i))
    print(', '.join(histology_results.query('pattern == @i').sort_values('membership')['g'].tolist()))
    print()    

print
