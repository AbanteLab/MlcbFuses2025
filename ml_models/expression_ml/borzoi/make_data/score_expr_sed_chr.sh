#!/bin/sh

# Iterate over chromosomes
for chr in $(seq 1 22)
do

    echo "Job for chromosome $chr"

    # Create output directory
    OUT_DIR="../../../../../../../projects/abante_lab/genomic_llms/borzoi/test_outputs/chr${chr}_snp_sed/f0c0"
    mkdir -p $OUT_DIR

    # Run
    borzoi_sed.py -o $OUT_DIR --rc --stats logSED,logD2 -t ../../../../../../../projects/abante_lab/genomic_llms/human/rna/sra/norm_targets_human.txt ../train_model/mini_models/human_all/params.json ../train_model/mini_models/human_all/f0/model0_best.h5 ../../../../../../../projects/abante_lab/ao_prediction_enrollhd_2024/enroll_hd/regulatory_vcfs/gwa12345.mis4.9064.hg38.cisreg0.5.mad0.01.chr${chr}.vcf > ${OUT_DIR}/sortida${chr}.txt 2> ${OUT_DIR}/errors${chr}.txt &

done