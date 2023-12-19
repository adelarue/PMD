#!/bin/bash
#SBATCH -a 23-24,31,70,83,94-95,165-166,173,179,236-237,276,278,307-308,335,347,378-379,419,425,443,449-450,457,464,477,496,520-521,538,563,591-592,610,638,652,662-663,670,699
#SBATCH -J nn_mar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 0 "nn"
