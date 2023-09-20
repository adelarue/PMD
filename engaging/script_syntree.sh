#!/bin/bash
#SBATCH -a 0-49
#SBATCH -J s_tree_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-00:30
srun julia synthetic_discrete.jl $SLURM_ARRAY_TASK_ID 1 2