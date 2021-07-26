#!/bin/bash
#SBATCH -t 3:59:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -o /mnt/scratch/lehieu1/log/data/I3_preproc_%a.log
#SBATCH --job-name=I3_preproc

# Preprocessing requires muon propagation must be done on py2 combo
module unload
# Arguments --array=0-999

FILE_NR=`expr $SLURM_ARRAY_TASK_ID`
FILE_NR=`printf "%03d\n" $FILE_NR`

script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/src/i3_preprocessing.py

#ID=4
SETNUM=11900
INDIR=/mnt/scratch/lehieu1/data/${SETNUM}/BaseProc
mkdir -p ${INDIR}/preproc
INNAME=${SETNUM}_MUONGUN_.000${FILE_NR}_Sunflower_240m_calibrated.i3.bz2
GCD_FILE=/mnt/scratch/lehieu1/data/11900/BaseProc/IceCubeHEX_Sunflower_240m_v3_ExtendedDepthRange.GCD.i3.bz2

echo "INFILE NAME : " ${INNAME} 
echo "$script -n ${INDIR}/${INNAME} --gcdfile ${GCD_FILE}"

$script -n ${INDIR}/${INNAME} --gcdfile ${GCD_FILE}
