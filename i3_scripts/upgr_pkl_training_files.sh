#!/bin/bash
#SBATCH -t 3:59:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -o /mnt/scratch/lehieu1/log/data/GNN_pkl_%a.log
#SBATCH --job-name=GNN_pkl

module unload
# Arguments --array=0-999

FILE_NR=`expr $SLURM_ARRAY_TASK_ID`
FILE_NR=`printf "%03d\n" $FILE_NR`

script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/src/upgr_pkl_training_files.py

# set 11900
SETNUM=11900
INDIR=/mnt/scratch/lehieu1/data/${SETNUM}/Sunflower_240m/BaseReco
#INNAME=${SETNUM}_MUONGUN_.000${FILE_NR}_Sunflower_240m_calibrated.i3.bz2
INNAME=${SETNUM}_MUONGUN_.000${FILE_NR}_Sunflower_240m_recos.i3.bz2
OUTDIR=/mnt/scratch/lehieu1/training_files/${SETNUM}_SplineMPE
mkdir -p ${OUTDIR}
OUTNAME=$(date +"%m%d%y")_000${FILE_NR}_training.pkl
GCD_FILE=/mnt/scratch/lehieu1/data/11900/IceCubeHEX_Sunflower_240m_v3_ExtendedDepthRange.GCD.i3.bz2

echo ${INDIR}/${INNAME}
echo ${OUTDIR}/${OUTNAME}

hostname

#$i3env python $script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
$script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
