#!/bin/bash

module purge
module load python/3.7
module load singularity/3.9.7

cd /scratch/hpc117a02/FedMPI_K30

export GLOO_SOCKET_IFNAME=eno1

echo "Client received server_port >> $server_port"
echo "Client received server_ip >> $server_ip"

singularity exec ../FLenv.sif torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --rdzv_id=0 --rdzv_endpoint=$server_ip:$server_port --master_addr=$server_ip --master_port=$server_port Start.py --selection_ratio=0.1 --round=$round --batch_size=32 --local_epochs=5 --lr=$lr --target_acc=$target_acc --num_threads=$num_threads --dataset=$dataset --iid=$iid --split=$split --method=$method --d=$d --wandb_on=True --project=FedMPI --entity=yumiri --group=$dataset --name=$method --cluster_type=KISTI --repeat=$repeat
