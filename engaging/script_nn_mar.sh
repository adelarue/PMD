#!/bin/bash
#SBATCH -a 23,24,47,52
#SBATCH -J nn_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 0 "nn"
