#!/bin/bash
#SBATCH -a 23-24,47,95,114,118,165-166,189,237,256,260,308,327,331,378-379,402,449-450,473,520-521,544,591-592,615,662-663
#SBATCH -J nn_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 0 0 "nn"