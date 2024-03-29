#!/bin/bash

#SBATCH --account=deyoungbuyin
#SBATCH --job-name=IceCube_GNN
#SBATCH --output=/mnt/scratch/lehieu1/log/GNN/GNN_%A_%a.out
#SBATCH --time=2-23:59:00
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

# Dataset
INDIR='/mnt/scratch/lehieu1/training_files/11900_regr_logE_IC86'
TRAINFILE=( ${INDIR}/processed/train_file* )
VALFILE=${INDIR}/processed/val_file.pkl
TESTFILE=${INDIR}/processed/test_file.pkl
echo ${TRAINFILE}
echo ${TESTFILE}
echo ${VALFILE}
NB_FILE=10
NB_TRAIN=147000
NB_VAL=20000
NB_TEST=20000

# Experiment
export SLURM_TIME_FORMAT='%y%m%d'
DATE=$(squeue -j ${SLURM_JOB_ID} -o "%V")
NAME="${DATE: -6}_11900_regr_logE_IC86"
RUN="$SLURM_ARRAY_TASK_ID"

PATIENCE=200
NB_EPOCH=200
LRATE=0.05
BATCH_SIZE=32
#REGR_MODE="direction"
REGR_MODE="energy"
EVAL=""
#EVAL="--evaluate"

# Network hyperparameters
NB_LAYER=10
NB_HIDDEN=64

# Modify parameters to fit multi-file submission
NB_TRAIN=`expr ${NB_TRAIN} / ${NB_FILE}`
NB_EPOCH=`expr ${NB_EPOCH} \* ${NB_FILE}`
TRAINFILE_SLICED=${TRAINFILE[@]:0:${NB_FILE}}

# Entering arguments
PYARGS="--name $NAME --run $RUN --train_file ${TRAINFILE_SLICED[@]} --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --lrate $LRATE --patience $PATIENCE --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN --regr_mode $REGR_MODE $EVAL"

echo -e "\nStarting experiment with name $NAME...\n"

module load powertools
source /mnt/home/lehieu1/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/ufs18/home-105/lehieu1/load_conda_env/GNN_conda

python /mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/src/gen2/multi_main.py $PYARGS

# Printing job statistics
js -j ${SLURM_JOB_ID}
