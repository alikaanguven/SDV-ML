#!/bin/bash 

# Usage: sbatch to_gpu.sh "python ParT_customised.py"


#SBATCH --job-name=fill_vtx
#SBATCH --output=/scratch-cbe/users/alikaan.gueven/job_outs/job_%j.out 
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G 
#SBATCH --nodes=1-1 
#SBATCH --partition=c
#SBATCH --qos=c_short
#SBATCH --time=08:00:00
echo ----------------------------------------------- 
$1
