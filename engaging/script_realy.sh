#!/bin/bash
#SBATCH -a 23,24,27,43-46,49,68
#SBATCH -J realy
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-12:00
#SBATCH --exclude=node1294,node1333

srun julia realy_scripts.jl $SLURM_ARRAY_TASK_ID
