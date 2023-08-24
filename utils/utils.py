from datetime import datetime

import random
import torch
import torch.distributed as dist



def omp_num_threads_per_clients(num_clients, dataset_name):
    num_threads=[]

    if dataset_name=="CIFAR10" or dataset_name=="MNIST" or dataset_name=="FashionMNIST":
        thread_list=[1,2,5] # 1: 10.68s | 2: 6.73s | 3: 5.31s 

    for t in thread_list:
        num_threads+=[t for idx in range(int(num_clients/len(thread_list)))]
    
    random.shuffle(num_threads)
    return num_threads

def printLog(message):
    now = str(datetime.now())
    print("["+now+"] " + message)

def set_num_local_epoch_by_random(num_clients, min, max):
    clients_local_epoch=[]
    for i in range(num_clients):
        clients_local_epoch.append(random.randint(min,max))
    return clients_local_epoch
