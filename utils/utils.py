from datetime import datetime

import random
import torch
import torch.distributed as dist



def omp_num_threads_per_clients(num_clients, dataset_name):
    num_threads=[]

    thread_list=[1,2,5]

    cnt=0
    for idx in range(num_clients):
        if cnt>=3:
            cnt=0
        num_threads.append(thread_list[cnt])
        cnt+=1
    
    
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
