#!/bin/bash
#SBATCH -t 11:59:00
#SBATCH -n 20
#SBATCH --mem=40G
#SBATCH -o /mnt/scratch/lehieu1/log/data/GNN_pkl_%a.log
#SBATCH --job-name=GNN_pkl

module unload
# Arguments --array=0-999

ARRAY=`expr $SLURM_ARRAY_TASK_ID`
#ARRAY=0
script=/mnt/home/lehieu1/IceCube/code/GNN/i3_scripts/src/upgr_reco_mdom_pkl_training_files.py
GCD_FILE=/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v3.2.2_ExtendedDepthRange_mDOM.GCD.i3.bz2
SET=21922
OUTDIR=/mnt/scratch/lehieu1/training_files/${SET}_complete
mkdir -p ${OUTDIR}

END=999

# Submit 20 tasks x 1000 jobs (limit)
for ((i=${ARRAY};i<=${END};((i=i+1000))))
do
  FILE_NR=`printf "%05d\n" ${i}`
  NUM=`expr $FILE_NR / 1000`
  NUM=`printf "%02d\n" $NUM`

  INDIR=/mnt/scratch/lehieu1/data/${SET}/00${NUM}000-00${NUM}999
  INNAME=Sunflower_240m_mDOM_2.2x_MuonGun.0${SET}.0${FILE_NR}_baseproc.i3.bz2
  OUTNAME=${SET}_0${FILE_NR}_training.pkl

  echo ${INDIR}/${INNAME}
  echo ${OUTDIR}/${OUTNAME}

  $script -i ${INDIR}/${INNAME} -o ${OUTDIR}/${OUTNAME} -g ${GCD_FILE} &
done

hostname
wait
