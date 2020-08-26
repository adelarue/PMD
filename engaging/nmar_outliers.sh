#!/bin/bash
#SBATCH -a 1
#SBATCH -J rmv
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p sched_mit_sloan_batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adelarue@mit.edu
#SBATCH --output=logs/nmar_outliers_\%a.log
#SBATCH --time=0-04:00
module load sloan/julia/1.0.0
module load sloan/gurobi/9.0.1
srun julia nmar_outliers.jl $SLURM_ARRAY_TASK_ID

# Total experiments: 59 * (11 + 6) = 1003