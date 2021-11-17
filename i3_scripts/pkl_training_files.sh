#!/bin/bash
#SBATCH -t 11:59:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -o /mnt/scratch/lehieu1/log/data/GNN_pkl_job_%a.log
#SBATCH --job-name=GNN_pkl

module unload
#eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
# Arguments --array=0-999

FILE_NR=`expr $SLURM_ARRAY_TASK_ID`
FILE_NR=`printf "%03d\n" $FILE_NR`

script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/src/pkl_training_files.py

# set 11900
INDIR=/mnt/scratch/lehieu1/data/11029/0${ID}000-0${ID}999
INNAME=FinalLevel_nugen_numu_IC86.2012.011029.00${ID}${FILE_NR}.i3.bz2
OUTDIR=/mnt/scratch/lehieu1/training_files/11029_FinalLevel
OUTNAME=$(date +"%m%d%y")_00${ID}${FILE_NR}_training.pkl
GCD_FILE=/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz

echo ${INDIR}/${INNAME}
echo ${OUTDIR}/${OUTNAME}

hostname

$script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
