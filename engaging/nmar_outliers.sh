#!/bin/bash
#SBATCH -a 500-789
#SBATCH -J rmv
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-04:00
srun julia nmar_outliers.jl $SLURM_ARRAY_TASK_ID

# Total experiments: 59 * (11 + 6) = 1003
# Total experiments: 71 * (11) = 781
