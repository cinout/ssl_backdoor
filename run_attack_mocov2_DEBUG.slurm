#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="bld_atk"
#SBATCH --account=punim1623
#SBATCH --time=0-00:50:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

export WORLD_SIZE=2   ### update world size: nodes x ntasks-per-node
export MASTER_PORT=28400
echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

cd moco
CUDA_VISIBLE_DEVICES=0,1 python main_moco.py \
    -a resnet18 \
    --lr 0.06 \
    --batch-size 256 \
    --multiprocessing-distributed \
    --world-size 2 \
    --rank 0 \
    --aug-plus \
    --mlp \
    --cos \
    --moco-align-w 0 \
    --moco-unif-w 0 \
    --moco-contr-w 1 \
    --moco-contr-tau 0.2 \
    --dist-url tcp://localhost:10005 \
    --save-folder-root haitian \
    --experiment-id blend_attack \
    --data ../poison-generation/data/BLEND/train/alpha_0.20_rate_0.50_targeted_True_filelist.txt \
    --workers 3 \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##