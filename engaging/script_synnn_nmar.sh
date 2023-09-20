#!/bin/bash
#SBATCH -a 21-49
#SBATCH -J s_nn_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-00:30
srun julia synthetic_discrete.jl $SLURM_ARRAY_TASK_ID 0 3