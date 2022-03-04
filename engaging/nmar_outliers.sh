#!/bin/bash
#SBATCH -a 0-0
#SBATCH -J flow
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-01:00
srun julia spBed_Assignement.jl $SLURM_ARRAY_TASK_ID
