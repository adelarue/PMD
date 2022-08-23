#!/bin/bash
#SBATCH -a 2-70
#SBATCH -J realy
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-00:00
srun julia realy_scripts.jl $SLURM_ARRAY_TASK_ID
