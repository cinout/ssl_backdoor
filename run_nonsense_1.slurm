#!/bin/bash

#SBATCH --partition=sapphire

#SBATCH --job-name="non1"
#SBATCH --account=punim1623
#SBATCH --time=0-00:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

### SBATCH --mail-user=haitianh@student.unimelb.edu.au
### SBATCH --mail-type=BEGIN

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python3 mytest.py

##Log this job's resource usage stats### 
my-job-stats -a -n -s
##