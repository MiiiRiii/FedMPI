#!/bin/bash
server_port="$((RANDOM%55535+10000))"
server_ip=`hostname -I | awk '{print $1}'`
num_clients=30

num_threads_list=(1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 8 8 8 8 8 8)
shuffled_num_threads_list=($(shuf -e "${num_threads_list[@]}"))
echo ${shuffled_num_threads_list[@]}
nnods=2
lr=0.001
dataset="MNIST"
iid="False"
split="gaussian"
target_acc=1.0

round=200
method="FedAvg"
d=30
repeat=3

node_rank=0

arr_idx=0
s=0
e=0
cnt=0



for i in {0..1} ; do
        cnt=0
        s=$arr_idx

        while [ $arr_idx -ne 30 ]; do
                cnt=$((cnt+shuffled_num_threads_list[arr_idx]))
                if [ $((cnt+shuffled_num_threads_list[arr_idx+1])) -gt 54 ]; then
                        arr_idx=$((arr_idx+1))
                        break
                fi
                arr_idx=$((arr_idx+1))
        done

        e=$((arr_idx-1))
        subset=("${shuffled_num_threads_list[@]:$s:$((e+1))}")
        num_threads_per_node="${subset[@]}"
	temp="1 $num_threads_per_node"
	echo $temp
done

