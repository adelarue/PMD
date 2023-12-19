#!/bin/bash
#SBATCH -a 23-24,31,70,94-95,102,141,165-166,234,236-237,276,279-280,307-308,315,335,378-379,449-450,457,520-521,528,591-592,628,662-663
#SBATCH -J lin_nmar_out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 1 "linear"
