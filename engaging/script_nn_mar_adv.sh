#!/bin/bash
#SBATCH -a 83,94-95,132,136-137,147,165,226,236,251,271,307,363,378-379,397,442-443,449-450,457,504,520-521,591-592,662-663
#SBATCH -J nn_nmar_out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia fakey.jl $SLURM_ARRAY_TASK_ID 1 1 "nn"
