#!/bin/bash

server_port="$((RANDOM%55535+10000))"
num_clients=30
num_threads=3
nnodes=2
lr=0.001
dataset="CIFAR10"
iid="False"
split="uniform"

nproc_per_node=()
if [ $((num_clients%nnodes)) -eq 0 ]; then
        for ((i=1;i<=nnodes;i++)); do
                nproc_per_node+=($((num_clients/nnodes)))
        done
fi
if [ $((num_clients%nnodes)) -ne 0 ]; then
        for ((i=1;i<=nnodes-1;i++)); do
                nproc_per_node+=($(($((num_clients/nnodes))+1)))
        done
        nproc_per_node+=($[num_clients-$[$[$[num_clients/nnodes]+1]*$[nnodes-1]]])
fi

cnt=0

NODEFILE=("210.107.197.167" "210.107.197.182" "210.107.197.190" "210.107.197.213" "210.107.197.212" "210.107.197.188")

for i in "${NODEFILE[@]}"; do
        if [ ${cnt} -eq 0 ]; then
                export server_ip=$i
                ssh $i server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split /home/$USER/miri/FedMPI/server.sh &
        fi
        if [ ${cnt} -ne 0 ]; then
                ssh $i server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split node_rank=$cnt /home/$USER/miri/FedMPI/client.sh &
        fi
        cnt=$((cnt+1))
done
