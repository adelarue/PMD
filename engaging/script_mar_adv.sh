#!/bin/bash
#SBATCH -a 0-399
#SBATCH -J nmar_out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-00:00
#SBATCH --exclude=node1294,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 1
