#!/bin/bash

#SBATCH --nodes 1

#SBATCH --gpus-per-node=1 # request a GPU

#SBATCH --tasks-per-node=1

#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance

#SBATCH --mem=32G

#SBATCH --time=03:00:00 # time (HH:MM:SS)

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

echo "starting program"

expKind=${1:-None}
expName=${2:-None}
dataFile=${3:-None}
echo "experiment kind $expKind"
echo "experiment name $expName"

wandb online
python main.py --exp-kind=$expKind --exp-name=$expName --run-wandb

echo "Finished running"
