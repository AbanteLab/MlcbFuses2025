#!/bin/bash

# merge bigwig replicates, generate .w5 files and run qc

data_dir="/pool01/projects/abante_lab/genomic_llms/"

# define ENCODE ID
# ENC_ID='SRP074904_putamen'
ENC_ID='SRP074904_caudate'

# define ENCODE file IDs
# FILE_REP1='SRR3500570'
# FILE_REP2='SRR3500571'
# FILE_REP3='SRR3500572'
# FILE_REP4='SRR3500573'
FILE_REP1='SRR3500562'
FILE_REP2='SRR3500563'
FILE_REP3='SRR3500564'
FILE_REP4='SRR3500565'

# # create folder for merged replicate files
mkdir -p "${data_dir}human/rna/sra/$ENC_ID/summary"

# step 1: generate per-replicate .w5 files

# rep1
if [ -f "${data_dir}human/rna/sra/$ENC_ID/rep1/$FILE_REP1.w5" ]; then
  echo "example RNA-seq .w5 already exists (rep 1)."
else
  bw_h5.py -z "${data_dir}human/rna/sra/$ENC_ID/rep1/$FILE_REP1.bigWig" "${data_dir}human/rna/sra/$ENC_ID/rep1/$FILE_REP1.w5"
fi

# rep2
if [ -f "${data_dir}human/rna/sra/$ENC_ID/rep2/$FILE_REP2.w5" ]; then
  echo "example RNA-seq .w5 already exists (rep 2)."
else
  bw_h5.py -z "${data_dir}human/rna/sra/$ENC_ID/rep2/$FILE_REP2.bigWig" "${data_dir}human/rna/sra/$ENC_ID/rep2/$FILE_REP2.w5"
fi

# rep3
if [ -f "${data_dir}human/rna/sra/$ENC_ID/rep3/$FILE_REP3.w5" ]; then
  echo "example RNA-seq .w5 already exists (rep 3)."
else
  bw_h5.py -z "${data_dir}human/rna/sra/$ENC_ID/rep3/$FILE_REP3.bigWig" "${data_dir}human/rna/sra/$ENC_ID/rep3/$FILE_REP3.w5"
fi

# rep4
if [ -f "${data_dir}human/rna/sra/$ENC_ID/rep4/$FILE_REP4.w5" ]; then
  echo "example RNA-seq .w5 already exists (rep 4)."
else
  bw_h5.py -z "${data_dir}human/rna/sra/$ENC_ID/rep4/$FILE_REP4.bigWig" "${data_dir}human/rna/sra/$ENC_ID/rep4/$FILE_REP4.w5"
fi


# step 2: merge replicates

if [ -f "${data_dir}human/rna/sra/$ENC_ID/summary/coverage.w5" ]; then
  echo "example RNA-seq .w5 already exists (merged)."
else
# left caudate rep 1 out as w5 couldn't be generated
  w5_merge.py -w -s mean -z "${data_dir}human/rna/sra/$ENC_ID/summary/coverage.w5" "${data_dir}human/rna/sra/$ENC_ID/rep1/$FILE_REP1.w5" "${data_dir}human/rna/sra/$ENC_ID/rep2/$FILE_REP2.w5" "${data_dir}human/rna/sra/$ENC_ID/rep3/$FILE_REP3.w5" "${data_dir}human/rna/sra/$ENC_ID/rep4/$FILE_REP4.w5"
fi


# step 3: run qc on each replicate and the merged file

if [ -f "${data_dir}human/rna/sra/$ENC_ID/summary/covqc/means.txt" ]; then
  echo "qc statistics already exist."
else
  # rep1
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "${data_dir}human/rna/sra/$ENC_ID/rep1/covqc" "${data_dir}human/rna/sra/$ENC_ID/rep1/$FILE_REP1.w5"

  # rep2
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "${data_dir}human/rna/sra/$ENC_ID/rep2/covqc" "${data_dir}human/rna/sra/$ENC_ID/rep2/$FILE_REP2.w5"

  # rep3
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "${data_dir}human/rna/sra/$ENC_ID/rep3/covqc" "${data_dir}human/rna/sra/$ENC_ID/rep3/$FILE_REP3.w5"

  # rep4
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "${data_dir}human/rna/sra/$ENC_ID/rep4/covqc" "${data_dir}human/rna/sra/$ENC_ID/rep4/$FILE_REP4.w5"

  # summary
  w5_qc.py -b "$BORZOI_HG38/blacklist/blacklist_hg38_all.bed" -o "${data_dir}human/rna/sra/$ENC_ID/summary/covqc" "${data_dir}human/rna/sra/$ENC_ID/summary/coverage.w5"
fi

