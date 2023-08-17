#!/bin/bash


cd /home/$USER/miri/FedMPI

torchrun --nnodes=$nnodes --nproc_per_node=$((nproc_per_node+1)) --node_rank=0 --rdzv_id=0 --rdzv_endpoint=$server_ip:$server_port --master_addr=$server_ip --master_port=$server_port start.py --selection_ratio=0.1 --round=30 --batch_size=32 --local_epochs=5 --lr=$lr --target_acc=0.9 --omp_num_threads=$num_threads --dataset=$dataset --iid=$iid --split=$split --wandb_on=False --project=FedMPI --entity=yumiri --group=test --name=FedAvg
