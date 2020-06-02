#!/bin/bash

#SBATCH --account=deyoungbuyin
#SBATCH --job-name=pkl
#SBATCH --output=/mnt/scratch/lehieu1/log/pkl.out
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

module load powertools
source /mnt/home/lehieu1/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/home/lehieu1/load_conda_env

python create_train_val_test.py -t 50 -v 5 -e 5 -i /mnt/scratch/lehieu1/training_files/with_energy/ -o /mnt/scratch/lehieu1/training_files/processed/masked_coords_we/
