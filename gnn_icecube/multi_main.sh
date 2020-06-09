#!/bin/bash

#SBATCH --account=deyoungbuyin
#SBATCH --job-name=IceCube_GNN
#SBATCH --output=/mnt/scratch/lehieu1/log/GNN/GNN_%A_%a.out
#SBATCH --time=1-23:59:00
#SBATCH --gres=gpu:k80:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

#mkdir -p slurm_out

# Dataset
TRAINFILE=( /mnt/scratch/lehieu1/training_files/processed/test/train_file* )
VALFILE='/mnt/scratch/lehieu1/training_files/processed/test/val_file.pkl'
TESTFILE='/mnt/scratch/lehieu1/training_files/processed/test/test_file.pkl'

NB_FILE=5
NB_TRAIN=10000
NB_VAL=1000
NB_TEST=1000

# Experiment
export SLURM_TIME_FORMAT='%m%d%y'
DATE=$(squeue -j ${SLURM_JOB_ID} -o "%V")
NAME="${DATE: -6}_test"
RUN="$SLURM_ARRAY_TASK_ID"

PATIENCE=200
NB_EPOCH=200
LRATE=0.05
BATCH_SIZE=32

# Network hyperparameters
NB_LAYER=10
NB_HIDDEN=64

# Modify parameters to fit multi-file submission
NB_TRAIN=`expr ${NB_TRAIN} / ${NB_FILE}`
PATIENCE=`expr ${PATIENCE} \* ${NB_FILE}`
NB_EPOCH=`expr ${NB_EPOCH} \* ${NB_FILE}`
TRAINFILE=${TRAINFILE[@]:0:${NB_FILE}}

# Entering arguments
PYARGS="--name $NAME --run $RUN --train_file ${TRAINFILE[@]} --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --lrate $LRATE --patience $PATIENCE --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo -e "\nStarting experiment with name $NAME...\n"

module load powertools
source /mnt/home/lehieu1/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/home/lehieu1/load_conda_env

time python /mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/src/multi_main.py $PYARGS

# Printing job statistics
js -j ${SLURM_JOB_ID}
