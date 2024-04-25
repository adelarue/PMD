#!/bin/bash
#SBATCH -a 47,114,118,166,185,189,237,256,260,327,331,402,473,540,544,611,615,682,686
#SBATCH -J nn_nmar_out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 1 "nn"
