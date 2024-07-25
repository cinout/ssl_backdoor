#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="bld_atk"
#SBATCH --account=punim1623
#SBATCH --time=2-00:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=80G

export WORLD_SIZE=4   ### update world size: nodes x ntasks-per-node
export MASTER_PORT=28400
echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

cd moco
srun python -u main_moco.py \
    -a resnet18 \
    --lr 0.06 \
    --batch-size 64 \
    --aug-plus \
    --mlp \
    --cos \
    --moco-align-w 0 \
    --moco-unif-w 0 \
    --moco-contr-w 1 \
    --moco-contr-tau 0.2 \
    --save-folder-root haitian \
    --experiment-id blend_attack \
    --data ../poison-generation/data/BLEND/train/alpha_0.20_rate_0.50_targeted_True_filelist.txt \

# --batch-size 256 \
# TODO: update DDP setting, --batch_size, --time

##Log this job's resource usage stats###
my-job-stats -a -n -s
##