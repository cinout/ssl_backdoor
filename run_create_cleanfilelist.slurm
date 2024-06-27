#!/bin/bash

#SBATCH --partition=sapphire

#SBATCH --job-name="eval_bb"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

### SBATCH --mail-user=haitianh@student.unimelb.edu.au
### SBATCH --mail-type=BEGIN

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

cd poison-generation
cd scripts

python create_dataset_filelist.py \
  --data_root ../../dataset/Imagenet100 \
  --output_file_root ../../dataset \


##Log this job's resource usage stats### 
my-job-stats -a -n -s
##