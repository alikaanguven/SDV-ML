#!/bin/bash 

# -----------------> GPU MEDIUM 
# Usage: Use only with submit_train.sh script


#SBATCH --job-name=ml_train
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G 
#SBATCH --nodes=1-1 
#SBATCH --partition=g 
#SBATCH --qos=g_medium
#SBATCH --gpus=1
#SBATCH --time=24:00:00 

echo -----------------------------------------------
echo "COMMAND: $*"
exec "$@"