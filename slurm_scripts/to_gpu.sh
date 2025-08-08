#!/bin/bash 

# Usage: sbatch to_gpu.sh "python ParT_customised.py"


#SBATCH --job-name=ParT_customised.py
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G 
#SBATCH --nodes=1-1 
#SBATCH --partition=g 
#SBATCH --qos=g_short 
#SBATCH --gpus=1
#SBATCH --time=07:59:00 
echo ----------------------------------------------- 
echo "COMMAND: $1" 
$1