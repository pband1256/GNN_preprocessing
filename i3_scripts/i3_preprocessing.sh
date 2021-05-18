#!/bin/bash
#SBATCH -t 3:59:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -o /mnt/scratch/lehieu1/log/data/I3_preprocessing_%a.log
#SBATCH --job-name=I3_preprocessing

# Preprocessing requires muon propagation must be done on py2 combo
module unload
# Arguments --array=0-999

FILE_NR=`expr $SLURM_ARRAY_TASK_ID`
FILE_NR=`printf "%03d\n" $FILE_NR`

script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/i3_preprocessing.py

#INDIR='/mnt/scratch/lehieu1/data/21217/0004000-0004999/'
#INNAME=Level2_IC86.2016_NuMu.021217.004${FILE_NR}.i3.zst
#GCD_FILE=/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz

ID=4
INDIR=/mnt/scratch/lehieu1/data/11374/0${ID}000-0${ID}999/clsim-base-4.0.3.0.99_eff/
INNAME=Level2_IC86.2012_nugen_numu.011374.00${ID}${FILE_NR}.clsim-base-4.0.3.0.99_eff.i3.bz2
GCD_FILE=/mnt/scratch/lehieu1/data/11374/0${ID}000-0${ID}999/clsim-base-4.0.3.0.99_eff/GeoCalibDetectorStatus_2012.56063_V1.i3.gz

# mkdir if doesn't exist
if [ ! -d "${INDIR}processed/" ]; then
  mkdir ${INDIR}processed/
fi

echo "INFILE NAME : " ${INNAME} 

$script -n ${INDIR}${INNAME} --gcdfile ${GCD_FILE}
