#!/bin/bash
#SBATCH --job-name=cyounes-job  # Job name
#SBATCH --account=amath  # Account to charge
#SBATCH --partition=cpu-g2     # Partition to use
#SBATCH --nodes=1               # Request 1 node
#SBATCH --cpus-per-task=90      # Request CPUs
#SBATCH --mem=1200G              # Memory allocation
#SBATCH --time=72:00:00         # Maximum runtime
#SBATCH --output=cyounes-job-%j.out  # Standard output log
#SBATCH --error=cyounes-job-%j.err   # Standard error log

module load elbert/julia/1.10.2/1.10.2 

srun julia Example2.jl > output.log 2>&1