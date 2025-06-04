### Preprocessing of GWAS data for ML applications.

Enroll-HD data was transformed into .vcf format using `plink2`. For the genotype models, we used the script `run_plink_filtered.sh`. For expression models, we used the script 'run_plink_v2.sh`, followed by filtering steps to only keep SNPs inside the promoters and enhancers of our gene set.

The creation of the target label and different structures for the prediction models are inside ./data_preprocessing.
