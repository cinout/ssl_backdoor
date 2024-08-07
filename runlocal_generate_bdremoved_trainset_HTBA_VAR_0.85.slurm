#!/bin/bash

#SBATCH --partition=sapphire

#SBATCH --job-name="eval_bb"
#SBATCH --account=punim1623
#SBATCH --time=0-03:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

### SBATCH --mail-user=haitianh@student.unimelb.edu.au
### SBATCH --mail-type=BEGIN

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python generate_bdremoved_trainset.py \
    --pred_scores_file pred_scores_trigger_HTBA_default_detector_variance_aug_crop_nviews_128_bs_32_sd_20.npy \
    --output_folder poison-generation/data/HTBA_trigger_10_targeted_n02106550/train \
    --cutoff_quantile 0.85 \

##Log this job's resource usage stats### 
my-job-stats -a -n -s
##