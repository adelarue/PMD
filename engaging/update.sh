#!/bin/bash
#SBATCH -a 0-9
#SBATCH -J pkg
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-00:00
srun julia pkg_updates.jl $SLURM_ARRAY_TASK_ID
