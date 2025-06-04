ML models need the data to be in table format, while the GNN models take tensors as input. `vae_snps_matrix.py` builds a feature table with SNPs genotype or gene-VAE embeddings (depending of the total number of SNPs of each gene). `vae_snps_gene_tensor.py` does the same but assmebling it into a 3d tensor, where the first dimension are samples, the second are genes and the third are gene embedding, either SNPs genotype or VAE embeddings, zero-padded up to 30.

`ppi_net_fromdownload.py` assembles the gene protein-protein graph by subsetting the link table from [StringDB](https://string-db.org/cgi/download). 

Tensor, graph, metadata as sex and CAG length and the target label are all assembled into a torch_geometric Data object in `binao_sex_cag_vae_snps_tensor.py` named AOGraphDataset. 