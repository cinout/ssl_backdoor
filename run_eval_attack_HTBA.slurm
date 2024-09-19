#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="htba"
#SBATCH --account=punim1623
#SBATCH --time=0-12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=150G

# export WORLD_SIZE=2   ### update world size: nodes x ntasks-per-node
# export MASTER_PORT=28400
# echo ">>> NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

cd moco
python eval_linear.py \
    --arch moco_resnet18 \
    --weights ./HTBA_bs64_world4/HTBA_bs64_world4/mocom0.999_contr1tau0.2_mlp_aug+_cos_b64_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --resume ./HTBA_bs64_world4/HTBA_bs64_world4/mocom0.999_contr1tau0.2_mlp_aug+_cos_b64_lr0.06_e120,160,200/linear/checkpoint_0199.pth.tar/checkpoint.pth.tar \
    --pretrained_frequency_model "" \
    \
    --poisoned_train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True_filelist.txt \
    --val_file ../dataset/imagenet100_val_clean_filelist.txt \
    --val_poisoned_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
    --probe_file ../dataset/imagenet100_train_clean_filelist_0.01_balanced_sd20.txt \
    \
    --target_class 26 \
    --batch-size 128 \
    --num_views 64 \
    --evaluate \
    --load_cache \
    --seed 42 \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    --ignore_probe_channel_num 15 \
    \
    --minority_criterion ss_score \
    --minority_percent_lower_bound 0.000 \
    --minority_percent_upper_bound 0.010 \
    \
    --use_frequency_detector \
    --frequency_detector_epochs 800 \
    \
    --ignore_probe_channels \
    --note "0.000-0.010, ep=800, channel_num=40, ignore_probe_channel_num=15" \
    
##Log this job's resource usage stats###
my-job-stats -a -n -s
