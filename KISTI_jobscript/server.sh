#!/bin/bash

module purge
module load python/3.7
module load singularity/3.9.7

cd /scratch/paop01a16/FedMPI

echo "Server received server_port is >> $server_port"
echo "Server received server_ip is >> $server_ip"

echo "Server real ip is >> `hostname -I | awk '{print $1}'`"


export GLOO_SOCKET_IFNAME=eno1


singularity exec ../FLenv.sif torchrun --nnodes=$nnodes --nproc_per_node=$((nproc_per_node+1)) --node_rank=0 --rdzv_id=0 --rdzv_endpoint=$server_ip:$server_port --master_addr=$server_ip --master_port=$server_port Start.py --selection_ratio=$selection_ratio --round=$round --batch_size=32 --local_epochs=5 --lr=$lr --target_acc=$target_acc --num_threads=$num_threads --dataset=$dataset --iid=$iid --split=$split --method=$method --d=$d --wandb_on=True --project=SemiAsync --entity=yumiri --group=test --name=$method --cluster_type=KISTI --repeat=$repeat

