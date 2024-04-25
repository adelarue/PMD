#!/bin/bash
#SBATCH -a 24,43,47,67,95,114,118,138,185,189,209,260,280,331,351,379,402,422,469,473,493,544,564,615,635,686
#SBATCH -J lin_nmar_out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 1 "linear"
