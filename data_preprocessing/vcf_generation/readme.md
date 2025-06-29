Genotype models only use the SNPs falling inside coding regions of the subset of 2774 genes selected. We need features encoded as a table, where each sample is an observation contained in a row, and each feature is a column. The raw enroll data has samples in columns, and SNPs in rows. The first part of the script `run_plink_filtered.sh` subsets the vcf files taking only those rows which represent an SNP contained in the look-up table checking by chromosome and position. This look-up table is generated by `biomart_snpid_gene_retrieval_v2.R`.

Then these subsetted rows are translated from the allele encoding of type 0/0 to an integer 0, 1 or 2:

| Old allele encoding | New allele encoding |
|:-------------------:|:-------------------:|
| 0/0 | 0 |
| 0/1 | 1 |
| 1/1 | 2 |

Outside the loop that iterates through chromosomes the script generates intermediate files for each action:
1. Concatenates the subsets of all chromosomes.
2. Transposes the numeric part of the matrix to have samples in columns (by running the build generated by `transpose_matrix.cpp`).
3. Translate the SNPs from chromosome and position to the reference SNP ID.
4. Finally writes the sample names and adds sex and CAG repeat length of each sample as the two first features. These rows are obtained from the metadata enroll file.

`alternative_prevalence_colsuming.py` filters the input matrix based on a minimum alternative frequency (MAF) threshold.
