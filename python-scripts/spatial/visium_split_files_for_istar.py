import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from pathlib import Path
# Tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
# Introduction to spatial sequencing data analysis
# https://www.youtube.com/watch?v=86uR01mwLIQ&t=1s

visium_directory           = Path('/media/jagraz/8TB/iStar_Web_App/KPMP_HK2873_ST')
output_directory           = visium_directory / 'istar'
counts_file                = output_directory / 'genexp-counts.tsv'
histology_file             = output_directory / 'histology.jpg'
spatial_coordinates_file   = output_directory / 'spot-locations.tsv'
pixel_size_raw_file        = output_directory / 'pixel-size-raw.txt'
radius_raw_file            = output_directory / 'radius-raw.txt'

output_directory.mkdir(exist_ok=True)

adata            = sc.read_visium(visium_directory)

# Extracting the required metadata from the AnnData object
sample_key            = list(adata.uns['spatial'].keys())[0]  # Use the appropriate sample key
scalefactors          = adata.uns['spatial'][sample_key]['scalefactors']
spot_diameter_fullres = scalefactors['spot_diameter_fullres']
tissue_hires_scalef   = scalefactors['tissue_hires_scalef']
tissue_lowres_scalef  = scalefactors['tissue_lowres_scalef']

# File 1: pixel size raw
pixel_size_raw        =  8000 / 2000 * tissue_hires_scalef
with open(pixel_size_raw_file, "w") as file:
    file.write(f'{pixel_size_raw}')
print(f"Pixel size raw: {pixel_size_raw}")
# ------------------------------------------------
# File 2: radius raw size
radius_raw            = spot_diameter_fullres * 0.5
with open(radius_raw_file, "w") as file:
    file.write(f'{radius_raw}')
print(f"Radius raw: {radius_raw}")
# ------------------------------------------------
# File 3: genexp-counts.tsv
# Extract counts matrix and convert to a DataFrame
counts         = adata.to_df().astype(int)
counts['spot'] = [str(row) + 'x' + str(col) for row, col in zip(adata.obs['array_row'], adata.obs['array_col'])]
# Reorder columns
counts         = counts[['spot'] + [col for col in counts.columns if col != 'spot']]
# Save to TSV file
counts.to_csv(counts_file, sep='\t', index=False)
# ------------------------------------------------
# File 4: histology.jpg
# Extract the histology image
# The key `V1_Human_Lymph_Node` should be replaced with the correct key for your dataset
sample_key = list(adata.uns['spatial'].keys())[0]
image = adata.uns['spatial'][sample_key]['images']['hires']
# Convert to an image format and save as JPEG
Image.fromarray((image * 255).astype(np.uint8)).save(histology_file)
# ------------------------------------------------
# File 5: spot-locations.tsv
# Extract spatial coordinates
locations = adata.obsm['spatial']
# Create a DataFrame with spot, x, and y columns
spot_locations = pd.DataFrame(locations, columns=['x', 'y'])
# Assuming `obs` has columns `array_row` and `array_col` or similar for spot positions
spot_locations['spot'] = [str(row) + 'x' + str(col) for row, col in zip(adata.obs['array_row'], adata.obs['array_col'])]
# Reorder columns
spot_locations = spot_locations[['spot', 'x', 'y']]
# Save to TSV file
spot_locations.to_csv(spatial_coordinates_file, sep='\t', index=False)