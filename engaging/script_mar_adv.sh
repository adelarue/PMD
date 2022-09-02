#!/bin/bash
#SBATCH -a 4,5,10,21,23,24,31,38,43,49,61,62,63,64,65,66,67,68,70
#SBATCH -J nmar_out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-06:00
srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 1
