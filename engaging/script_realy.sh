#!/bin/bash
#SBATCH -a 10,24,27,34,38,45,46,61,62,63,64,65,67,66,69
#SBATCH -J realy
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-12:00
srun julia realy_scripts.jl $SLURM_ARRAY_TASK_ID
