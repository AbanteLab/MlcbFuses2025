#!/bin/bash

# download example data from SRA

# define ENCODE ID
ENC_ID="SRP074904_putamen"

# define project directory
PROJECT_DIR='/pool01/projects/abante_lab/genomic_llms/human/rna/sra/${ENC_ID}/'

# define remote urls
# URL_REP1='https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR3500573/SRR3500570'
# URL_REP2='https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR3500573/SRR3500571'
# URL_REP3='https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR3500572/SRR3500572'
# URL_REP4='https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR3500573/SRR3500573'
URL_REP3='https://duffel.rail.bio/recount3/human/data_sources/sra/base_sums/04/SRP074904/72/sra.base_sums.SRP074904_SRR3500572.ALL.bw'
URL_REP4='https://duffel.rail.bio/recount3/human/data_sources/sra/base_sums/04/SRP074904/73/sra.base_sums.SRP074904_SRR3500573.ALL.bw'

# define ENCODE file IDs
# FILE_REP1='SRR3500570'
# FILE_REP2='SRR3500571'
FILE_REP3='SRR3500572'
FILE_REP4='SRR3500573'

# create folder for bigwig files
# mkdir -p "${PROJECT_DIR}rep1"
# mkdir -p "${PROJECT_DIR}rep2"
mkdir -p "${PROJECT_DIR}rep3"
mkdir -p "${PROJECT_DIR}rep4"

# # download bigwig files; rep1
# if [ -f "${PROJECT_DIR}rep1/$FILE_REP1.bigWig" ]; then
#   echo "example RNA-seq data already downloaded (rep 1)."
# else
#   wget $URL_REP1 -O "${PROJECT_DIR}rep1/$FILE_REP1.bigWig"
# fi

# # download bigwig files; rep2
# if [ -f "${PROJECT_DIR}rep2/$FILE_REP2.bigWig" ]; then
#   echo "example RNA-seq data already downloaded (rep 2)."
# else
#   wget $URL_REP2 -O "${PROJECT_DIR}rep2/$FILE_REP2.bigWig"
# fi

# download bigwig files; rep3
if [ -f "${PROJECT_DIR}rep3/$FILE_REP3.bigWig" ]; then
  echo "example RNA-seq data already downloaded (rep 3)."
else
  wget $URL_REP3 -O "${PROJECT_DIR}rep3/$FILE_REP3.bigWig"
fi

# download bigwig files; rep4
if [ -f "${PROJECT_DIR}rep4/$FILE_REP4.bigWig" ]; then
  echo "example RNA-seq data already downloaded (rep 4)."
else
  wget $URL_REP4 -O "${PROJECT_DIR}rep4/$FILE_REP4.bigWig"
fi