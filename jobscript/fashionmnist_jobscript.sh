#!/bin/bash
#PBS -V
#PBS -N FCHAFL
#PBS -q normal
#PBS -l select=2:ncpus=68
#PBS -A etc
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR

module purge
module load singularity/3.9.7

cd /scratch/paop01a16/FedMPI

server_port="$((RANDOM%55535+10000))"
server_ip=`hostname -I | awk '{print $1}'`
num_clients=30

num_threads_list=(1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 8 8 8 8 8 8)
shuffled_num_threads_list=($(shuf -e "${num_threads_list[@]}"))

nnodes=2
lr=0.001
dataset="FashionMNIST"
iid="False"
split="gaussian"
target_acc=1.0

round=500
method="CHAFL"
d=30
repeat=2

node_rank=0

arr_idx=0
s=0
e=0
cnt=0

for i in `cat $PBS_NODEFILE`; do
        cnt=0
        s=$arr_idx

        while [ $arr_idx -ne 30 ]; do
                cnt=$((cnt+shuffled_num_threads_list[arr_idx]))
                if [ $((cnt+shuffled_num_threads_list[arr_idx+1])) -gt 67 ]; then
                        arr_idx=$((arr_idx+1))
                        break
                fi
                arr_idx=$((arr_idx+1))
        done

        e=$((arr_idx-1))
        subset=("${shuffled_num_threads_list[@]:$s:$((e-s+1))}")
        num_threads_per_node="${subset[@]}"

        if [ ${node_rank} -eq 0 ]; then
                temp="1 $num_threads_per_node"
                num_threads=`echo $temp | tr -d ' '`
                ssh $i server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=$((e-s+1)) num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split method=$method d=$d target_acc=$target_acc round=$round repeat=$repeat /scratch/paop01a16/FedMPI/server.sh &
        fi
        if [ ${node_rank} -ne 0 ]; then
                num_threads=`echo $num_threads_per_node | tr -d ' '`
                ssh $i server_ip=$server_ip server_port=$server_port num_clients=$num_clients nproc_per_node=$((e-s+1)) num_threads=$num_threads nnodes=$nnodes lr=$lr dataset=$dataset iid=$iid split=$split method=$method d=$d target_acc=$target_acc round=$round repeat=$repeat node_rank=$node_rank /scratch/paop01a16/FedMPI/client.sh &
        fi
        node_rank=$((node_rank+1))
done
wait