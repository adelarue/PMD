#!/bin/bash
#SBATCH -a 0-99
#SBATCH -J realy
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-12:00
srun julia syndata.jl $SLURM_ARRAY_TASK_ID 1
