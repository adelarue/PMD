#!/bin/bash
#SBATCH -a 23-24,31,70,94-95,102,115,165-166,173,186,191,205-206,236-237,257,279,328,333,348,376,378,386,416,449,470,490,499,541,589,612,641,683
#SBATCH -J lin_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 0 "linear"
