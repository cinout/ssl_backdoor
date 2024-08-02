#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="det_eval"
#SBATCH --account=punim1623
#SBATCH --time=0-12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=100G

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
python eval_detector.py \
    --arch moco_resnet18 \
    --trigger_type BLEND \
    --weights ./BLEND_alpha02/blend_attack/mocom0.999_contr1tau0.2_mlp_aug+_cos_b64_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --train_file ../poison-generation/data/BLEND/train/alpha_0.20_rate_0.50_targeted_True_filelist.txt \
    --batch_size 128 \
    --num_views 64 \
    --detector InterViews \
    --interview_task spectral_signature \
    --aug_type crop_plus_perspective \
    --ss_option max \
    --seed 20 \
    

##Log this job's resource usage stats###
my-job-stats -a -n -s
##