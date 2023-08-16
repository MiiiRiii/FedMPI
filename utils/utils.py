from datetime import datetime

import random
import torch
import torch.distributed as dist



def omp_num_threads_per_clients(num_clients, system_heterogeneity):
    if system_heterogeneity==1: # omp_num_threads 1~3
        return get_num_threads_by_gaussian(num_clients, 1, 5, 2, 1)
    elif system_heterogeneity==2: # omp_num_threads 1~8
        # thread1 : 60s | thread2 : 38s | thread3 : 31s | thread4 : 26s | thread5 : 25s | thread6 : 22s | thread7 : 21s | thread8 :  20s
        num_threads_per_client = [1 for i in range(10)] + [2 for i in range(5)] + [3 for i in range(5)] + [4 for i in range(2)] + [5 for i in range(2)] + [6 for i in range(2)] + [7 for i in range(2)] + [8 for i in range(2)]
        random.shuffle(num_threads_per_client)
        return num_threads_per_client

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
