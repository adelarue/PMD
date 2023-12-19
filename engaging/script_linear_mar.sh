#!/bin/bash
#SBATCH -a 23-24,31,94-95,102,165-166,236-237,307-308,378-379,386,449-450,520-521,567,591-592,662-663
#SBATCH -J lin_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 0 "linear"
