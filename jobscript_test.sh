#!/bin/bash

server_port=29603
num_clients=20
num_threads=3
nnodes=5
lr=0.01
dataset="MNIST"
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



server_ip="210.107.197.167"

sshpass -p wise!1448 ssh wise@210.107.197.167 "source /home/wise/anaconda3/etc/profile.d/conda.sh;source /home/wise/anaconda3/bin/activate FL;export server_ip=$server_ip;export server_port=$server_port;export num_clients=$num_clients;export nproc_per_node=${nproc_per_node[$cnt]};export num_threads=$num_threads;export nnodes=$nnodes;export lr=$lr;export dataset=$dataset;export iid=$iid;export split=$split;export node_rank=$cnt;source /home/$USER/miri/FedMPI/server.sh;" &
cnt=$((cnt+1))
sshpass -p wise ssh euler@210.107.197.188 "source /home/euler/anaconda3/etc/profile.d/conda.sh;source /home/euler/anaconda3/bin/activate FL;export server_ip=$server_ip;export server_port=$server_port;export num_clients=$num_clients;export nproc_per_node=${nproc_per_node[$cnt]};export num_threads=$num_threads;export nnodes=$nnodes;export lr=$lr;export dataset=$dataset;export iid=$iid;export split=$split;export node_rank=$cnt;export GLOO_SOCKET_IFNAME=eno1;source /home/euler/miri/FedMPI/client.sh;" &
cnt=$((cnt+1))
sshpass -p eB3pY32r ssh dijkstra@210.107.197.212 "source /home/dijkstra/anaconda3/etc/profile.d/conda.sh;source /home/dijkstra/anaconda3/bin/activate FL;export server_ip=$server_ip;export server_port=$server_port;export num_clients=$num_clients;export nproc_per_node=${nproc_per_node[$cnt]};export num_threads=$num_threads;export nnodes=$nnodes;export lr=$lr;export dataset=$dataset;export iid=$iid;export split=$split;export node_rank=$cnt;export GLOO_SOCKET_IFNAME=eno1;source /home/dijkstra/miri/FedMPI/client.sh;" &
cnt=$((cnt+1))
sshpass -p asdf7946 ssh hdoop@210.107.197.213 "source /home/hdoop/anaconda3/etc/profile.d/conda.sh;source /home/hdoop/anaconda3/bin/activate FL;export server_ip=$server_ip;export server_port=$server_port;export num_clients=$num_clients;export nproc_per_node=${nproc_per_node[$cnt]};export num_threads=$num_threads;export nnodes=$nnodes;export lr=$lr;export dataset=$dataset;export iid=$iid;export split=$split;export node_rank=$cnt;export GLOO_SOCKET_IFNAME=eno1;source /home/hdoop/miri/FedMPI/client.sh;" &
cnt=$((cnt+1))
sshpass -p wise!1448 ssh wise@210.107.197.182 "source /home/wise/anaconda3/etc/profile.d/conda.sh;source /home/wise/anaconda3/bin/activate FL;export server_ip=$server_ip;export server_port=$server_port;export num_clients=$num_clients;export nproc_per_node=${nproc_per_node[$cnt]};export num_threads=$num_threads;export nnodes=$nnodes;export lr=$lr;export dataset=$dataset;export iid=$iid;export split=$split;export node_rank=$cnt;export GLOO_SOCKET_IFNAME=eno1;source /home/wise/miri/FedMPI/client.sh;" &




