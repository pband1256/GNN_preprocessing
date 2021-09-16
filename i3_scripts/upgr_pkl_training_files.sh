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

script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/src/upgr_reco_mdom_pkl_training_files.py

# set 11900
#SETNUM=11900
#INDIR=/mnt/scratch/lehieu1/data/${SETNUM}/Sunflower_240m/BaseReco
#INNAME=${SETNUM}_MUONGUN_.000${FILE_NR}_Sunflower_240m_recos.i3.bz2
#OUTDIR=/mnt/scratch/lehieu1/training_files/${SETNUM}_SplineMPE
#OUTNAME=$(date +"%m%d%y")_000${FILE_NR}_training.pkl

INDIR=/mnt/home/lehieu1/IceCube/code/GNN/iceprod/data/multi_sensor_muongun_overhead
INNAME=Sunflower_240m_mDOM_2.2x_MuonGun.020016.000000_baseproc.i3.bz2
OUTDIR=/mnt/scratch/lehieu1/training_files/iceprod_test
OUTNAME=$(date +"%m%d%y")_000000_training.pkl
mkdir -p ${OUTDIR}

GCD_FILE=/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v3.2_ExtendedDepthRange_mDOM.GCD.i3.bz2

echo ${INDIR}/${INNAME}
echo ${OUTDIR}/${OUTNAME}

hostname

$script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE}
