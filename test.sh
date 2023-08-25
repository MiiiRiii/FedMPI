#!/bin/bash

num_threads_list=(1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 8 8 8 8 8 8)
shuffled_num_threads_list=($(shuf -e "${num_threads_list[@]}"))
arr_idx=0
for i in {0..1}; do
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
        echo $num_threads_per_node
        temp="1 $num_threads_per_node"
        echo $temp
        echo $((e-s+1))
	num_threads=`echo $temp | tr -d ' '`
	echo $num_threads
done
