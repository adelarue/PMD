#!/bin/bash
#SBATCH -a 0-389
#SBATCH -J rmv
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-02:00
srun julia realy_scripts.jl $SLURM_ARRAY_TASK_ID
