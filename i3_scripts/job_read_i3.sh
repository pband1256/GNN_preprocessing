#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -o /mnt/scratch/lehieu1/log/data/read_i3_%a.log
#SBATCH --job-name=read_i3

INPUT="/mnt/scratch/lehieu1/data/11900/Sunflower_240m/BaseReco"
#/mnt/scratch/lehieu1/data/21901/baseproc/
OUTPUT="/mnt/home/lehieu1/IceCube/plot/iceprod/11900_hist"

script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/read_i3.py

$script -i $INPUT -o ${OUTPUT}

#$script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
