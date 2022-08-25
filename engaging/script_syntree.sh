#!/bin/bash
#SBATCH -a 30-49
#SBATCH -J syn_tree_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-12:00
srun julia comparison.jl $SLURM_ARRAY_TASK_ID 1 2