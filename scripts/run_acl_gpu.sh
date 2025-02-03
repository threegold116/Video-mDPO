#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate llavadpo
echo "start"
export WANDB_MODE=offline
cd /data/scir/sxjiang/project/Video-mDPO/llavahound
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
export CUDA_VISIBLE_DEVICES=4,5,6,7
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
#     train_mdpo.py \
#     /share/home/jfliang/Project/Hall/Video-mDPO/llavaov/config_mdpo_loss.yaml
export PYTHONPATH=/data/scir/sxjiang/project/Video-mDPO/:$PYTHONPATH
accelerate launch --config_file=/data/scir/sxjiang/project/Video-mDPO/scripts/multi_gpu.yaml\
    /data/scir/sxjiang/project/Video-mDPO/llavahound/train_mdpo.py\
    /data/scir/sxjiang/project/Video-mDPO/llavahound/config_mdpo_loss_per_crop_frames.yaml