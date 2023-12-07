#!/bin/bash
#SBATCH -a 10,21,23-24,31,38,44,49,51,61-66,70
#SBATCH -J nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p newnodes
#SBATCH --time=0-12:00
#SBATCH --exclude=node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 0 0