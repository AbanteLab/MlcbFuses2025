#!/bin/bash

##########################################################
# Generates vcf files and filters for genes in lookuptab #
##########################################################

# Activate Conda environment with plink2 dependencies
source activate enrollhd_jupyter

# Directories
enrolldir="../../../../../../projects/abante_lab/ao_prediction_enrollhd_2024/enroll_hd/vcfs/"
outdir="../../../../../../projects/abante_lab/ao_prediction_enrollhd_2024/enroll_hd/vcfs/"

# Iterate over all chromosomes
for chr in {1..22}
do
    # Extract the unique name by removing the file extension
    filename="${enrolldir}gwa12345.mis4.9064.chr${chr}"
    out="${enrolldir}gwa12345.mis4.9064.chr${chr}"

    # Print filename
    echo "Input file: $filename"
    echo "Output file: $out"

    # Run plink command
    plink2 --bfile "$filename" --recode vcf --out "$out"

done