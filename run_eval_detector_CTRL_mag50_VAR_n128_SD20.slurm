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
#SBATCH --time=0-20:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=80G

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
    --trigger_type CTRL_p0.01_mag50 \
    --weights ./CTRL_bs64_world4_poirate_0.01_mag50/CTRL_bs64_world4_poirate_0.01_mag50/mocom0.999_contr1tau0.2_mlp_aug+_cos_b64_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --train_file ../poison-generation/data/CTRL_0.01_mag_50/train/rate_1.00_targeted_True_filelist.txt \
    --batch_size 32 \
    --num_views 128 \
    --detector InterViews \
    --interview_task variance \
    --aug_type crop \
    --rrc_scale_min 0.2 \
    --rrc_scale_max 0.5 \
    --seed 20 \
    

##Log this job's resource usage stats###
my-job-stats -a -n -s
##