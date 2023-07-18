from datetime import datetime
import random


def client_random_select(clients_idx, num_selected_clients):
    shuffled_clients_idx = clients_idx[:]
    random.shuffle(shuffled_clients_idx)
    return shuffled_clients_idx[0:num_selected_clients]

def printLog(message):
    now = str(datetime.now())
    print("["+now+"] " + message)

def set_num_local_epoch_by_random(num_clients, min, max):
    clients_local_epoch=[]
    for i in range(num_clients):
        clients_local_epoch.append(random.randint(min,max))
    return clients_local_epoch
