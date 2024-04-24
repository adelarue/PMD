#!/bin/bash
#SBATCH -a 0-43,45-48,50-70
#SBATCH -J realy
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia realy_scripts.jl $SLURM_ARRAY_TASK_ID
