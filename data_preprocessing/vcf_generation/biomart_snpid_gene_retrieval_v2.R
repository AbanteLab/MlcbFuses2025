#!/usr/bin/env Rscript

###############################################################################################################
# RETRIEVE SNP ID AND GENE NAME FROM SNP COORDINATES
# Note: for biomart to work it requires dplyr v2.3.4 - the last working version. Otherwise we get the error
#     Error in `collect()`: ! Failed to collect lazy table. Caused by error in `db_collect()`: ...
###############################################################################################################

# deps
require(biomaRt)
require(data.table)

# Directory to files with SNP positions per chromosome
data.dir <- "/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/data/"
# data.dir <- "/Users/jordiabante/petra/cfuses_gnn_enrollhd_2024/data/"
snp.dir <- paste(data.dir, "/SNPs/", sep = "")
gene.dir <- paste(data.dir, "/genes/", sep = "")

# Output directory
out.dir <- paste(data.dir, "/biomart/", sep = "")

# 1. get gene list
gene.file <- paste(gene.dir, "core_genes_names.txt", sep = "")
gene.lst <- fread(gene.file)$Gene

# 2. get gene coordinates

# Connect to the Ensembl BioMart database
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl", "https://feb2014.archive.ensembl.org")

# Retrieve gene symbols
gene.symb <- getBM(
      attributes = c("ensembl_gene_id", "external_gene_id", "chromosome_name", "start_position", "end_position", "strand"),
      filters = "hgnc_symbol", values = gene.lst, mart = ensembl
)

# assign column names
colnames(gene.symb) <- c("ensembl_gene_stable_id", "gene_name", "chr", "st", "end", "strand")
# gene.symb[1:20,1:4]

# filter chromosomes
gene.symb <- gene.symb[gene.symb$chr %in% c("X", "Y", paste(1:22)), ]

# 3. get SNPs in gene windows

# Connect to dataset
mart <- useMart("ENSEMBL_MART_SNP", dataset = "hsapiens_snp", "https://feb2014.archive.ensembl.org")

# Empty data frame to store ensembl outputs
snp.tab <- data.frame()

for (i in 1:nrow(gene.symb)) {
      print(paste("Retrieving SNPs for ", gene.symb$gene_name[i], " (", i, "/", nrow(gene.symb), ")", sep = ""))

      # Create coords vector as required for getBM
      query <- paste(gene.symb$chr[i], gene.symb$st[i], gene.symb$end[i], sep = ":")
      # query <- "22:16954364:16974993"

      # set maximum and current number of attempts
      max_attempts <- 5
      current_attempt <- 1

      # try X times per SNP set
      while (current_attempt <= max_attempts) {
            tryCatch(
                  {
                        print(paste("Attempt ", current_attempt, " of gene ", i, "...", sep = ""))

                        # Get table from ensembl
                        sub.snp.tab <- getBM(
                              attributes = c("chr_name", "chrom_start", "refsnp_id", "allele"),
                              filters = c("chromosomal_region"), values = query, mart = mart
                        )

                        # If the command succeeds, break out of the loop
                        break
                  },
                  error = function(err) {
                        # Print the error message
                        cat(paste("Attempt", current_attempt, "failed with error:", conditionMessage(err), "\n"))

                        # Increment the attempt counter
                        current_attempt <- current_attempt + 1

                        # If it's the last attempt, stop trying
                        if (current_attempt > max_attempts) {
                              stop("Max attempts reached. Exiting.")
                        }
                  }
            )
      }

      # skip if number of SNPs is 0
      if (nrow(sub.snp.tab) == 0) next

      # assign gene name
      sub.snp.tab$gene <- gene.symb$gene_name[i]

      # Append to general snp tab
      snp.tab <- rbind(snp.tab, sub.snp.tab)
}

# store table
write.table(snp.tab, file = paste0(out.dir, "snp_gene_lookup_tab_grch37.txt"), quote = FALSE, sep = "\t", row.names = FALSE)
