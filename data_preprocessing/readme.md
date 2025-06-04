### Preprocessing of GWAS data for ML applications.

Enroll-HD data was transformed into .vcf format using _plink2_. For the genotype models, we used the script `run_plink_filtered.sh`. For expression models, we used the script `run_plink_v2.sh`, followed by filtering steps to only keep SNPs inside the promoters and enhancers of our gene set.

Since our goal is to identify genes that reduce the unexplained variability after accounting for CAG length, we trained our models on the residuals of a linear model that predicts the AO from CAG length alone. This linear model is in `ao_quadratic_residuals.py`. `ao_binning.ipynb` contains the visualization and discretization of the residuals into 5 balanced groups.

Different models require different data structures, created in ./data_preprocessing.
