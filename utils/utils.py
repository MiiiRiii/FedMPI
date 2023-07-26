from datetime import datetime

import random
import torch
import torch.distributed as dist

def client_select_randomly(clients_idx, num_selected_clients):
    shuffled_clients_idx = clients_idx[:]
    random.shuffle(shuffled_clients_idx)
    return shuffled_clients_idx[0:num_selected_clients]

def client_select_by_loss(num_clients, num_selected_clients, global_loss):
    cnt=0
    client_select_checklist=[False for i in range(num_clients+1)]
    selected_clients_list=[]
    local_loss=torch.zeros(1)
    local_loss_list={}
    remain_res=[]

    for idx in range(num_clients):
        req=dist.irecv(tensor=local_loss)
        if cnt<num_selected_clients :
            req.wait()
            printLog(f"client {req.source_rank()}의 local loss는 {local_loss.item()}입니다.")
            local_loss_list[req.source_rank()]=local_loss.item()
            if local_loss.item()>global_loss:
                selected_clients_list.append(req.source_rank())
                client_select_checklist[req.source_rank()]=True
                cnt=cnt+1
        else:
            remain_res.append(req)

    if cnt < num_selected_clients :
        sorted_local_loss_list = sorted(local_loss_list.items(), key=lambda item:item[1], reverse=True)
        for idx in range(num_clients):
            if cnt>=num_selected_clients:
                break
            if client_select_checklist[sorted_local_loss_list[idx][0]] == False:
                selected_clients_list.append(sorted_local_loss_list[idx][0])
                client_select_checklist[sorted_local_loss_list[idx][0]]=True
                cnt=cnt+1

    return selected_clients_list, remain_res

def omp_num_threads_per_clients(num_clients, system_heterogeneity):
    if system_heterogeneity==1: # omp_num_threads 1~3
        return get_num_threads_by_gaussian(num_clients, 1, 3, 2, 1)
    elif system_heterogeneity==2: # omp_num_threads 1~17
        return get_num_threads_by_gaussian(num_clients, 1, 20, 10, 5)

def get_num_threads_by_gaussian(num_clients, min, max, mean, std):
    result = []
    for i in range(num_clients):
        while True:
            sample_value=round(random.gauss(mean, std))
            if sample_value>=min and sample_value<=max:
                break
        result.append(sample_value)
    return result

def printLog(message):
    now = str(datetime.now())
    print("["+now+"] " + message)

def set_num_local_epoch_by_random(num_clients, min, max):
    clients_local_epoch=[]
    for i in range(num_clients):
        clients_local_epoch.append(random.randint(min,max))
    return clients_local_epoch
