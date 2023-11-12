#!/bin/bash

module purge
module load python/3.7
module load singularity/3.9.7

cd /scratch/paop01a16/FedMPI

export GLOO_SOCKET_IFNAME=eno1

echo "Client received server_port >> $server_port"
echo "Client received server_ip >> $server_ip"

singularity exec ../FLenv.sif torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --rdzv_id=0 --rdzv_endpoint=$server_ip:$server_port --master_addr=$server_ip --master_port=$server_port Start.py --round=$round --batch_size=32 --local_epochs=5 --lr=$lr --selection_ratio=$selection_ratio --target_acc=$target_acc --num_threads=$num_threads --dataset=$dataset --iid=$iid --split=$split --method=$method --d=$d --cluster_type=KISTI --repeat=$repeat
