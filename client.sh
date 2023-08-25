#!/bin/bash

cd /home/$USER/miri/FedMPI/

echo $nnodes
echo $nproc_per_node
echo $node_rank
echo $server_ip
echo $server_port
echo $lr
echo $num_threads
echo $dataset
echo $iid
echo $split

#torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank --rdzv_id=0 --rdzv_endpoint=$server_ip:$server_port --master_addr=$server_ip --master_port=$server_port Start.py --selection_ratio=0.1 --batch_size=32 --local_epochs=5 --lr=$lr --target_acc=0.9 --omp_num_threads=$num_threads --dataset=$dataset --iid=$iid --split=$split
