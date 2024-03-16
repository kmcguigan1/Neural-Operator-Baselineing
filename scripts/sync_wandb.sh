#!/bin/bash

#SBATCH --nodes 1

#SBATCH --tasks-per-node=1

#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance

#SBATCH --mem=32G

#SBATCH --time=00:10:00 # time (HH:MM:SS)

#SBATCH --output=/home/kmcguiga/projects/def-sirisha/kmcguiga/computeCanadaOutput/%j.out

#SBATCH --account=def-sirisha

#SBATCH --mail-user=kmcguiga@uwaterloo.ca

#SBATCH --mail-type=BEGIN

#SBATCH --mail-type=END

#SBATCH --mail-type=FAIL

#SBATCH --mail-type=REQUEUE

module purge

echo "loading python module"
module load python/3.11

echo "loading virtual environment"
source /home/kmcguiga/projects/def-sirisha/kmcguiga/environments/torch/bin/activate

echo "syncing wandb"
wandb sync --sync-all --clean logs/wandb
echo "Finished running"
