#!/bin/bash
#SBATCH -a 21,23,33-49 #52686040
#SBATCH -J s_nn_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
#SBATCH --exclude=node1294,node1333

srun julia comparison_full.jl $SLURM_ARRAY_TASK_ID 0 3