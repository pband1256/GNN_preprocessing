#!/bin/bash

#SBATCH --account=deyoungbuyin
#SBATCH --job-name=IceCube_GNN
#SBATCH --output=/mnt/scratch/lehieu1/log/GNN/GNN_%A_%a.out
#SBATCH --time=6-23:59:59
#SBATCH --gres=gpu:k80:1
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

# Dataset
TRAINFILE=( /mnt/scratch/lehieu1/training_files/21901_complete/processed/train_file*.pkl* )
VALFILE='/mnt/scratch/lehieu1/training_files/21901_complete/processed/val_file.pkl'
TESTFILE='/mnt/scratch/lehieu1/training_files/21901_complete/processed/test_file.pkl'

NB_FILE=10
NB_TRAIN=760000
NB_VAL=76000
NB_TEST=95000

# Experiment
export SLURM_TIME_FORMAT='%y%m%d'
DATE=$(squeue -j ${SLURM_JOB_ID} -o "%V")
NAME="${DATE: -6}_21901ex_Ereg_mDOM"
#NAME="211019_21901_DIRreg_mDOM"
#NAME="TEST"
RUN="$SLURM_ARRAY_TASK_ID"

PATIENCE=20
NB_EPOCH=200
LRATE=0.05
BATCH_SIZE=16
REGR_MODE="energy"
#REGR_MODE="direction"
EVAL='None'
#OLD_RECO="None"
#EVAL="model_epoch_147.pkl"
OLD_RECO="/mnt/home/lehieu1/IceCube/plot/iceprod/11900_hist.pkl"

# Network hyperparameters
NB_LAYER=10
NB_HIDDEN=16

# Modify parameters to fit multi-file submission
NB_TRAIN=`expr ${NB_TRAIN} / ${NB_FILE}`
NB_EPOCH=`expr ${NB_EPOCH} \* ${NB_FILE}`
TRAINFILE_SLICED=${TRAINFILE[@]:0:${NB_FILE}}

# Entering arguments
PYARGS="--name $NAME --run $RUN --train_file ${TRAINFILE_SLICED[@]} --val_file $VALFILE --test_file $TESTFILE --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --lrate $LRATE --patience $PATIENCE --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN --regr_mode $REGR_MODE --evaluate $EVAL --old_reco_file $OLD_RECO"

echo -e "\nStarting experiment with name $NAME...\n"

module load powertools
source /mnt/home/lehieu1/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/ufs18/home-105/lehieu1/load_conda_env/GNN_conda

python /mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/src/gen2_mdom/multi_main.py $PYARGS

# Printing job statistics
#js -j ${SLURM_JOB_ID}
