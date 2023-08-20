#!/bin/bash

array=(1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 7 7 7 7 7 7)

shuffled=($(shuf -e "${array[@]}"))


arr_idx=0
s=0
e=0
echo ${shuffled[@]}
for i in {1..2}; do
	cnt=0
	s=$arr_idx

	while [ $arr_idx -ne 30 ]; do
		cnt=$((cnt+shuffled[arr_idx]))
		if [ $((cnt+shuffled[arr_idx+1])) -gt 68 ]; then 
	
			arr_idx=$((arr_idx+1))
			break
		fi
		arr_idx=$((arr_idx+1))

	done

	e=$((arr_idx-1))
	
	subset=("${shuffled[@]:$s:$((e+1))}")
	num_threads_per_node="${subset[@]}"
	
	echo "0 $num_threads_per_node"
	echo "cnt: $cnt"
	echo "s: $s"
	echo "e: $e"
done


num_thread=(1 2 3 4 5)
#for item in "${num_thread[@]}"; do
#	num_thread_string="${num_thread_string}${item}"
#done
#echo $num_thread_string
num_thread_string="${num_thread[@]}"
echo "$num_thread_string"
torchrun --nnodes=1 --nproc_per_node=6 --node_rank=0 --rdzv_id=0 --rdzv_endpoint=210.107.197.167:29603 --master_addr=210.107.197.167 --master_port=29603 Start.py --selection_ratio=0.5 --round=1500 --batch_size=32 --local_epochs=5 --lr=0.001 --target_acc=0.9 --dataset=CIFAR10 --iid=False --split=gaussian --method=CHAFL --d=30 --wandb_on=True --project=FedMPI --entity=yumiri --group=CIFAR10 --name=CHAFL --repeat=9 --cluster_type=KISTI --num_threads="0 $num_thread_string"

