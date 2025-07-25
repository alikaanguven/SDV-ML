#!/bin/bash 

# Usage: sbatch to_gpu.sh "python ParT_customised.py"


#SBATCH --job-name=ML_thingy
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G 
#SBATCH --nodes=1-1 
#SBATCH --partition=c
#SBATCH --qos=c_short 
#SBATCH --time=07:59:00 
echo ----------------------------------------------- 
$1
