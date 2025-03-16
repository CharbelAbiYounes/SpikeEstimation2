#!/bin/bash
#SBATCH --job-name=cyounes-job  # Job name
#SBATCH --account=amath  # Account to charge
#SBATCH --partition=gpu-rtx6k     # Partition to use
#SBATCH --nodes=1               # Request 1 node
#SBATCH --cpus-per-task=30      # Request CPUs
#SBATCH --mem=300G              # Memory allocation
#SBATCH --time=72:00:00         # Maximum runtime
#SBATCH --output=cyounes-job-%j.out  # Standard output log
#SBATCH --error=cyounes-job-%j.err   # Standard error log

module load elbert/julia/1.10.2/1.10.2 

srun julia Example2R.jl > Radoutput.log 2>&1