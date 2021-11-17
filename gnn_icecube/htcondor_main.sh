#!/bin/bash

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
RUN=$1
PROJECT="Illume"

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

source /home/users/hieule/anaconda3/etc/profile.d/conda.sh
conda activate GNN_conda
#nvidia-smi 
python /home/users/hieule/code/GNN/gnn_icecube/src/gem2_mdom/multi_main.py $PYARGS
