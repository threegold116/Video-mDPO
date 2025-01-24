#!/bin/bash
source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate llavadpo
echo "start"
export WANDB_MODE=offline
cd /share/home/jfliang/Project/Hall/Video-mDPO/llavaov
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
#     train_mdpo.py \
#     /share/home/jfliang/Project/Hall/Video-mDPO/llavaov/config_mdpo_loss.yaml

accelerate launch --config_file=/share/home/jfliang/Project/Hall/Video-mDPO/scripts/multi_gpu.yaml\
    /share/home/jfliang/Project/Hall/Video-mDPO/llavaov/train_mdpo.py\
    /share/home/jfliang/Project/Hall/Video-mDPO/llavaov/config_mdpo_loss.yaml