#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="mocov2"
#SBATCH --account=punim1623
#SBATCH --time=0-00:30:00
###TODO: change time

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

export WORLD_SIZE=2   ### FIXME: update world size: nodes x ntasks-per-node
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
    --world-size 1 \
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
    --experiment-id trytrykan \
    --data ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True_filelist.txt \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##