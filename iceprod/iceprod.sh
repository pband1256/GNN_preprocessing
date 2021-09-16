#!/bin/bash

#SBATCH --account=deyoungbuyin
#SBATCH --job-name=iceprod_test
#SBATCH --output=/mnt/scratch/lehieu1/log/iceprod_%A.out
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=lehieu1

FILE="multi_sensor_muongun_overhead"

rm -r /mnt/home/lehieu1/IceCube/code/GNN/iceprod/data/${FILE}
mkdir -p /mnt/home/lehieu1/IceCube/code/GNN/iceprod/data/${FILE}
cd /mnt/home/lehieu1/IceCube/code/GNN/iceprod/data/${FILE} 

start_time="$(date -u +%s.%N)"

eval $(/cvmfs/icecube.opensciencegrid.org/iceprod/master/setup.sh)
python -m iceprod.core.i3exec --offline -f /mnt/home/lehieu1/IceCube/code/GNN/iceprod/json/${FILE}.json -d

end_time="$(date -u +%s.%N)"

elapsed="$(bc <<<"$end_time-$start_time")"
echo "Total of $elapsed seconds elapsed for process"
