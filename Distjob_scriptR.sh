#!/bin/bash
#SBATCH --job-name=cyounes-job  # Job name
#SBATCH --account=randommatrix  # Account to charge
#SBATCH --partition=compute     # Partition to use
#SBATCH --nodes=4               # Request 1 node
#SBATCH --cpus-per-task=40      # Request CPUs
#SBATCH --mem=175G              # Memory allocation
#SBATCH --time=48:00:00         # Maximum runtime
#SBATCH --output=cyounes-job-%j.out  # Standard output log
#SBATCH --error=cyounes-job-%j.err   # Standard error log

module load elbert/julia/1.10.2/1.10.2 

scontrol show hostnames > hosts.txt
cat hosts.txt  

srun julia Example2R.jl > Betaoutput.log 2>&1