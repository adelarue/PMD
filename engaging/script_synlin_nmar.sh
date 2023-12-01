#!/bin/bash
#SBATCH -a 0-49
#SBATCH -J s_lin_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-00:00
srun julia comparison_full.jl $SLURM_ARRAY_TASK_ID 0 1