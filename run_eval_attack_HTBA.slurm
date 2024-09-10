#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="mocov2"
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
    \
    --poisoned_train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True_filelist.txt \
    --val_file ../dataset/imagenet100_val_clean_filelist.txt \
    --val_poisoned_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
    \
    --batch-size 128 \
    --evaluate \
    --load_cache \
    --seed 42 \
    --detect_trigger_channels \
    --channel_num 1 2 4 8 12 16 20 24 \
    --num_views 64 \
    # --channel_num 1 2 3 4 6 8 10 12 14 16 20 24 \

##Log this job's resource usage stats###
my-job-stats -a -n -s
