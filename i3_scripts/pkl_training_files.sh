#!/bin/bash
#SBATCH -t 3:59:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -o /mnt/scratch/lehieu1/log/data/GNN_pkl_job_%a.log
#SBATCH --job-name=GNN_pkl

module unload
#eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
# Arguments --array=0-999

FILE_NR=`expr $SLURM_ARRAY_TASK_ID`
FILE_NR=`printf "%03d\n" $FILE_NR`

#i3env=/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/V00-00-03/env-shell.sh
script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/pkl_training_files.py

# set 21217
#INDIR=/mnt/scratch/lehieu1/data/21217/0000000-0000999/processed/
#OUTDIR=/mnt/scratch/lehieu1/training_files/21217_IC_hit_filtered_muE/
#INNAME=processed_Level2_IC86.2016_NuMu.021217.000${FILE_NR}.i3.zst
#GCD_FILE=/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz

# set 11374
INDIR=/mnt/scratch/lehieu1/data/11374/00000-00999/clsim-base-4.0.3.0.99_eff/processed
OUTDIR=/mnt/scratch/lehieu1/training_files/11374_IC_hit_filtered_muE/
INNAME=processed_Level2_IC86.2012_nugen_numu.011374.000${FILE_NR}.clsim-base-4.0.3.0.99_eff.i3.bz2
GCD_FILE=/mnt/scratch/lehieu1/data/11374/00000-00999/clsim-base-4.0.3.0.99_eff/GeoCalibDetectorStatus_2012.56063_V1.i3.gz

OUTNAME=$(date +"%m%d%y")_000${FILE_NR}_training.pkl

hostname

#$i3env python $script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
$script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
