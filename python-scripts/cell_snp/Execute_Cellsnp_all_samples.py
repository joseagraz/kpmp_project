import csv
import time
from pathlib import Path
import subprocess

# Define the paths and filenames
chromosome_of_interest = "chrX"
vcf_file_name          = "chrX.ucsc_filtered_dbsnp_chrX.vcf.gz"
bam_file               = "possorted_genome_bam.bam"
barcode_file           = "barcodes.tsv"
# Execution @ HPC
hpc_root_dir           = Path("/home/jagraz/KPMP_Data/Privately_Available_Data")
# Execution @ PC
root_dir               = Path("/media/jagraz/PMACS")
# root_dir               = Path("/home/jagraz")
support_files_dir      = hpc_root_dir / Path("Supporting_Files")
data_dir               = hpc_root_dir / Path("Original_Download_KPMP_S3_Bucket_Oct_16_2023")
vcf_dir                = support_files_dir / Path("xiao_references")
datalogs_dir           = support_files_dir / Path("Datalogs")
# Path to the CSV file containing the sample list
germline_dir_name      = f"germline_het_{chromosome_of_interest}_{Path(bam_file).stem}"

csv_file_path          = support_files_dir / 'list_of_samples_processed_using_cellranger.txt'
csv_file_path          = root_dir / Path(*csv_file_path.parts[3:])

cellsnp_dir            = Path("cellsnp-dbsnp") / Path(germline_dir_name)
bash_script_dir        = support_files_dir / Path("Bash-Files/PMACS_Bash_Job_Files/cellsnp-lite_scripts/batch-dump")
# Adjust path to mounted directory
bash_script_dir        = root_dir / Path(*bash_script_dir.parts[3:])

# Read the sample list from the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        sample_name = row[0]  

        # Define paths based on the sample name
        target_dir    = data_dir   / Path(sample_name) / Path("cellranger_output/outs")
        bam_file_path = target_dir / bam_file
        vcf_file_path = vcf_dir    / vcf_file_name   

        outputdir     = target_dir / cellsnp_dir
        source_barcode_file = target_dir / Path(f"filtered_feature_bc_matrix/{barcode_file}.gz")
        target_barcode_file = outputdir  / barcode_file

        # Create the bash script content
        bash_script_content = f"""#!/bin/bash
#BSUB -J cellsnp_job            # LSF job name
#BSUB -o {datalogs_dir}/cellsnp_job.%J.out
#BSUB -e {datalogs_dir}/cellsnp_job.%J.error
#BSUB -M 120000
#BSUB -n 4
#BSUB -R "rusage [mem=120000] span[hosts=1]"
#
# -------------------
echo "Removing directory: {outputdir.stem}"
rm -rf {outputdir}
echo "Creating directory: {outputdir.stem}"
mkdir -p {outputdir}
echo "Uncompressing {source_barcode_file.stem}"
gzip --stdout --decompress --force "{source_barcode_file}" > "{target_barcode_file}"
echo "Running conda"
conda run -n cellsnp-lite_env_p310 \\
    cellsnp-lite \\
    -s "{bam_file_path}" \\
    -b "{target_barcode_file}" \\
    -O "{outputdir}" \\
    -R "{vcf_file_path}" \\
    -p 6 \\
    --gzip
"""

        # Write the bash script to a file
        bash_file_path = bash_script_dir / Path(f"cellsnp_sample_{sample_name}.sh")
        with open(bash_file_path, 'w') as bash_file:
            bash_file.write(bash_script_content)
            print(f"Bash script for sample '{sample_name}' written to {bash_file_path}")
        print
        # subprocess.Popen(['bsub <  ', bash_file_path])
        # time.sleep(5)