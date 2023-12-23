#!/bin/bash
#SBATCH -a 24,47,95,118,166,189,237,260,308,331,379,402,450,473,521,544,592,615,663
#SBATCH -J nn_nmar
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 0 0 "nn"