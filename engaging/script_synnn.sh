#!/bin/bash
#SBATCH -a 21-49
#SBATCH -J s_nn_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
srun julia comparison_full.jl $SLURM_ARRAY_TASK_ID 1 3