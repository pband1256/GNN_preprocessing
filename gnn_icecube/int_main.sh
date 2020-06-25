#!/bin/bash

#SBATCH --account=deyoungbuyin
#SBATCH --job-name=bench_GNN
#SBATCH --output=/mnt/scratch/lehieu1/log/GNN/bench_%A_%a.out
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:k80:1
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

# Dataset
TRAINFILE=( /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file* )
VALFILE='/mnt/scratch/lehieu1/training_files/processed/nocuts_multi/val_file.pkl'
TESTFILE='/mnt/scratch/lehieu1/training_files/processed/nocuts_multi/test_file.pkl'

NB_FILE=10
NB_TRAIN=1000000
NB_VAL=100000
NB_TEST=100000

# Experiment
DATE=$(date +'%m%d%y')
NAME="${DATE}_5epoch_benchmarking"
RUN="$SLURM_ARRAY_TASK_ID"

PATIENCE=20
NB_EPOCH=1
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

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo -e "\nStarting experiment with name $NAME...\n"

source /mnt/home/lehieu1/anaconda3/etc/profile.d/conda.sh
conda activate GNN_conda
nvidia-smi
#python /mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/src/multi_main.py $PYARGS
python /mnt/home/lehieu1/IceCube/code/GNN/gnn_icecube/src/multi_main.py --name 061920_5epoch_job_profiler_benchmarking --run 0 --train_file /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_1.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_10.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_2.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_3.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_4.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_5.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_6.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_7.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_8.pkl /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/train_file_9.pkl --val_file /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/val_file.pkl --test_file /mnt/scratch/lehieu1/training_files/processed/nocuts_multi/test_file.pkl --nb_train 100000 --nb_val 100000 --nb_test 100000 --batch_size 32 --nb_epoch 5 --lrate 0.05 --patience 200 --nb_layer 10 --nb_hidden 64
