For expression features, we need the genotype of the enhancers and promoters of our gene set. 

#### Plink2

`run_plink_v2.sh` converts GWAS data to vcf format, without filtering.

#### Enhancer filtering

`filtering_enhancer_score.py` visualizes how many SNPs we would have as a function of enhancer score, to select the most adequate enhancer score threshold (in our case we took 0.5). `regulatory_regions` filters the _Genehancer_ dataset to contain only enhancers above the selected threshold and regulating genes in our gene set. It also takes the gene promoters, established as 4k bp centered at the gene TSS.

`regulatory_maf.py` is an exploratory script to select the minimum alternative frequency, finally set at 0.01.

`regulatory_vcfs.py` finally filters the vcfs according the enhancers and MAF selected.