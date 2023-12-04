#!/bin/bash
#SBATCH -a 0-19
#SBATCH -J pkg
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-00:20
#SBATCH --exclude=node1294
srun julia pkg_updates.jl $SLURM_ARRAY_TASK_ID
