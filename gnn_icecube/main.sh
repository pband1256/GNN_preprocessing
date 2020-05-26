#!/bin/bash

#SBATCH --job-name=IceCube_GNN
#SBATCH --output=/mnt/scratch/lehieu1/log/GNN/GNN_%A_%a.out
#SBATCH --time=23:59:00
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

#mkdir -p slurm_out
# Dataset
TRAINFILE='/mnt/scratch/lehieu1/training_files/processed/train_file.pkl'
VALFILE='/mnt/scratch/lehieu1/training_files/processed/val_file.pkl'
TESTFILE='/mnt/scratch/lehieu1/training_files/processed/test_file.pkl'

NB_TRAIN=100000
NB_VAL=10000
NB_TEST=10000

# Experiment
NAME="aa_test_updates"
RUN="$SLURM_ARRAY_TASK_ID"
NB_EPOCH=100
BATCH_SIZE=64

# Network hyperparameters
NB_LAYER=10
NB_HIDDEN=64

OPTIONS=""

PYARGS="--name $NAME --run $RUN --train_file $TRAINFILE --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo -e "\nStarting experiment with name $NAME...\n"

module load powertools
source /mnt/home/lehieu1/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/home/lehieu1/load_conda_env

python /mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/src/main.py $PYARGS
