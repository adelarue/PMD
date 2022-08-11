#!/bin/bash
#SBATCH -a 0-0
#SBATCH -J mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-06:00
srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 0
