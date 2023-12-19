#!/bin/bash
#SBATCH -a 23-24,42,70,94-95,165-166,212,236-237,244,307-308,378-379,425,449-450,457,520-521,528,572,591-592,628,662-663,670
#SBATCH -J lin_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 0 0 "linear"