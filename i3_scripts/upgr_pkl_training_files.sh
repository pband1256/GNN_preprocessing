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
#FILE_NR=9
FILE_NR=`printf "%03d\n" $FILE_NR`

#i3env=/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/V00-00-03/env-shell.sh
#script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/upgr_pkl_training_files.py
script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/mult_upgr_pkl_training_files.py

# set 11900
ID=0
SETNUM=11900
INDIR=/mnt/scratch/lehieu1/data/${SETNUM}/BaseProc
INNAME=${SETNUM}_MUONGUN_.00${ID}${FILE_NR}_Sunflower_240m_calibrated.i3.bz2
OUTDIR=/mnt/scratch/lehieu1/training_files/${SETNUM}_FinalLevel
mkdir -p ${OUTDIR}
OUTNAME=$(date +"%m%d%y")_00${ID}${FILE_NR}_training.pkl
GCD_FILE=/mnt/scratch/lehieu1/data/11900/BaseProc/IceCubeHEX_Sunflower_240m_v3_ExtendedDepthRange.GCD.i3.bz2

echo ${INDIR}/${INNAME}
echo ${OUTDIR}/${OUTNAME}

hostname

#$i3env python $script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
$script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
