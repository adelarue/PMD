#!/bin/bash
#SBATCH -a 43,49,114,120,185,191,256,262,327,333,398,404,469,475,540,546,611,617,682,688
#SBATCH -J realy
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH --exclude=node1111,node1333

srun julia realy_scripts.jl $SLURM_ARRAY_TASK_ID
