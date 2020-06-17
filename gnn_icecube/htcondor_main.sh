#!/bin/bash

CONDOR_ID=$(Cluster)_$(Process)

# Dataset
TRAINFILE='/data/icecube/hieule/training_files/processed/weight1_energy/train_file.pkl'
VALFILE='/data/icecube/hieule/training_files/processed/weight1_energy/val_file.pkl'
TESTFILE='/data/icecube/hieule/training_files/processed/weight1_energy/test_file.pkl'

NB_TRAIN=100000
NB_VAL=10000
NB_TEST=10000

# Experiment
NAME="${MONTH}${DAY}${YEAR :-2}_patience_200_100k"
RUN="${Step}"

NB_EPOCH=200
LRATE=0.05
BATCH_SIZE=32

# Network hyperparameters
NB_LAYER=10
NB_HIDDEN=64

PYARGS="--name $NAME --run $RUN --train_file $TRAINFILE --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --lrate $LRATE --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo -e "\nStarting experiment with name $NAME...\n"

source /home/users/hieule/anaconda3/etc/profile.d/conda.sh
conda activate GNN_conda

python /home/users/hieule/code/GNN/gnn_icecube/src/main.py $PYARGS
