#!/bin/bash

server_port=$((RANDOM%55535+10000))
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

NODEFILE=("210.107.197.167" "210.107.197.182" "210.107.197.190" "210.107.197.213" "210.107.197.212" "210.107.197.188")



export server_ip=210.107.197.167

sshpass -p wise!1448 ssh wise@210.107.197.167 "source /home/wise/anaconda3/etc/profile.d/conda.sh;source /home/wise/anaconda3/bin/activate FL;source /home/$USER/miri/FedMPI/server.sh server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split node_rank=$cnt;" &

sshpass -p wise ssh euler@210.107.197.188 "source /home/euler/anaconda3/etc/profile.d/conda.sh;source /home/euler/anaconda3/bin/activate FL;source /home/euler/miri/FedMPI/client.sh server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split node_rank=$cnt;" &

sshpass -p eB3pY32r ssh dijkstra@210.107.197.212 "source /home/dijkstra/anaconda3/etc/profile.d/conda.sh;source /home/dijkstra/anaconda3/bin/activate FL;source /home/dijkstra/miri/FedMPI/client.sh server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split node_rank=$cnt;" &

sshpass -p asdf7946 ssh hdoop@210.107.197.213 "source /home/hdoop/anaconda3/etc/profile.d/conda.sh;source /home/hdoop/anaconda3/bin/activate FL;source /home/hdoop/miri/FedMPI/client.sh; server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split node_rank=$cnt" &

sshpass -p wise!1448 ssh wise@210.107.197.182 "source /home/wise/anaconda3/etc/profile.d/conda.sh;source /home/wise/anaconda3/bin/activate FL;source /home/wise/miri/FedMPI/client.sh; server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=${nproc_per_node[$cnt]} num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split node_rank=$cnt" &


