#!/bin/bash
#SBATCH -a 30-49 #52685858
#SBATCH -J s_nn_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
#SBATCH --exclude=node1294,node1333

srun julia comparison_full.jl $SLURM_ARRAY_TASK_ID 1 3