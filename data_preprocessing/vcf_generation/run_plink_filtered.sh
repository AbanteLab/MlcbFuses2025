#!/bin/bash

##########################################################
# Generates vcf files and filters for genes in lookuptab #
##########################################################

# Activate Conda environment with plink2 dependencies
source activate enrollhd_jupyter

# Directories
enrolldir="../../data/enroll_hd/"
snptabdir="../../data/biomart/"
outdir="../../data/enroll_hd/filtered_vcfs/"
featuredir="../../data/features/"

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

    # Get headers of each vcf file to check same sample order across files
    grep '^#CHROM' "${enrolldir}gwa12345.mis4.9064.chr${chr}.vcf" | cut -f 10- > ${enrolldir}headers/header${chr}.txt

    # Get rows of vcf of positions in snp lookup tab abd drop innecessary cols
    awk -v chr_name="$chr" '$1 == chr_name {print $2}' ${snptabdir}revised_filtered_snp_gene_lookup_tab.txt | grep -wf - "${enrolldir}gwa12345.mis4.9064.chr${chr}.vcf" | cut -f -2,10- > "${outdir}gwa12345.mis4.9064.chr${chr}.model.vcf"

    # Change allele encoding
    sed -i 's/0\/0/0/g; s/0\/1/1/g; s/1\/0/1/g; s/1\/1/2/g' "${outdir}gwa12345.mis4.9064.chr${chr}.model.vcf"

    # Delete original vcf and log files, keep the filtered vcf version
    rm "${enrolldir}gwa12345.mis4.9064.chr${chr}.vcf" "${enrolldir}gwa12345.mis4.9064.chr${chr}.log"

done

# Concatenate all filtered files 
cat ${outdir}* > "${outdir}chrompos_snps_model3.txt"

# Separate by numeric matrix and row identification
cut -f 1-2 "${outdir}chrompos_snps_model3.txt" > "${outdir}chrompos_model3.txt"
cut -f 3- "${outdir}chrompos_snps_model3.txt" > "${outdir}snps_numeric_model3.txt"

# Transpose numeric matrix (check input file name of cpp script)
./transpose_matrix
# (Generates ${outdir}Tsnps_numeric_model3.txt)

# Get rs IDs from lookup table
awk 'NR==FNR{arr[$1,$2]=$3; next} {print $1,$2,arr[$1,$2]}' OFS='\t' "${snptabdir}revised_filtered_snp_gene_lookup_tab.txt" "${outdir}chrompos_model3.txt" > "${outdir}chrompos_refIDs_model3.txt"

# Extract refIDs column and make it a row, delete last tabulator
cut -f 3 "${outdir}chrompos_refIDs_model3.txt" | tr '\n' '\t' | sed 's/\t$//' > "${outdir}refIDs_row.txt"

# Add final \n to row file
echo >> "${outdir}refIDs_row.txt"

# Add ref IDs on top of the transposed matrix
cat "${outdir}refIDs_row.txt" "${outdir}Tsnps_numeric_model3.txt" > "${outdir}T_refIDs_snps_model3.txt"

# Add sample name, sex and CAG length as first three columns (obtained from gwa12345.mis4.9064.sample.info.190805.txt)
paste "${enrolldir}id_sex_cag_columns.txt" "${outdir}T_refIDs_snps_model3.txt" > "${featuredir}feature_matrix_m3.txt"