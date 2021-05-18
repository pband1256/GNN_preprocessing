#!/bin/bash
#SBATCH -t 3:59:00
#SBATCH -n 1
#SBATCH --mem=50G
#SBATCH -o /mnt/scratch/lehieu1/log/data/create_pkl_job_%a.log
#SBATCH --job-name=create_pkl

script=/mnt/home/lehieu1/IceCube/code/GNN/create_train_val_test.py

NB_TRAIN=700
NB_VAL=100
NB_TEST=100
NB_SPLIT=1

INDIR=/mnt/scratch/lehieu1/training_files/11900_FinalLevel
OUTDIR=/mnt/scratch/lehieu1/training_files/11900_FinalLevel/processed
mkdir -p ${OUTDIR}
#EMIN=0
#EMAX=500
FLAT=0

hostname

echo python $script -t ${NB_TRAIN} -v ${NB_VAL} -e ${NB_TEST} -n ${NB_SPLIT} -i ${INDIR} -o ${OUTDIR} --flat ${FLAT}
python $script -t ${NB_TRAIN} -v ${NB_VAL} -e ${NB_TEST} -n ${NB_SPLIT} -i ${INDIR} -o ${OUTDIR} --flat ${FLAT}
