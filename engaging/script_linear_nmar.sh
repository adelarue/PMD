#!/bin/bash
#SBATCH -a 4,23,42,49,62-63,70,94-95,115,141,146,165-166,186,212,223,236-237,244,257,307,328,354,378,399,425,449-450,457,470,496,520-521,528,541,548,558,572-573,591,612,628,662,670
#SBATCH -J lin_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 0 0 "linear"