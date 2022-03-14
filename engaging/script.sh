#!/bin/bash
#SBATCH -a 400-780
#SBATCH -J rmv
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
srun julia fakey.jl $SLURM_ARRAY_TASK_ID
