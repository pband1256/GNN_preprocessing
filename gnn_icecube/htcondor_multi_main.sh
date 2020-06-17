#!/bin/bash

# Dataset
TRAINFILE=( /data/icecube/hieule/training_files/processed/500GeV_min_cuts_multi/train_file* )
VALFILE='/data/icecube/hieule/training_files/processed/500GeV_min_cuts_multi/val_file.pkl'
TESTFILE='/data/icecube/hieule/training_files/processed/500GeV_min_cuts_multi/test_file.pkl'

NB_FILE=10
NB_TRAIN=1000000
NB_VAL=100000
NB_TEST=100000

# Experiment
DATE=$(date +'%m%d%y')
NAME="${DATE}_500gev_mincuts_10x100k_20patience"
RUN=$1

PATIENCE=20
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
TRAINFILE_SLICED=${TRAINFILE[@]:0:${NB_FILE}}

# Entering arguments
PYARGS="--name $NAME --run $RUN --train_file ${TRAINFILE_SLICED[@]} --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --lrate $LRATE --patience $PATIENCE --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo "$PYARGS"
echo -e "\nStarting experiment with name $NAME...\n"

source /home/users/hieule/anaconda3/etc/profile.d/conda.sh
conda activate GNN_conda
python /home/users/hieule/code/GNN/gnn_icecube/src/multi_main.py $PYARGS
