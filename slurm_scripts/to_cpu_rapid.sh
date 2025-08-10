#!/bin/bash 

# Usage: sbatch to_gpu.sh "python ParT_customised.py"


#SBATCH --job-name=ML_thingy
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3G 
#SBATCH --nodes=1-1 
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --time=01:00:00 
echo ----------------------------------------------- 
echo "COMMAND: $1" 
$1