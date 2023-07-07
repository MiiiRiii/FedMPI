from mpi4py import MPI
from Server import Server
from Client import Client
from utils import printLog

import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import os
import socket

MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def init_FL(rank, size, FLgroup): 
    with open('/home/wise/miri/mpitest/FedAvg/config.yaml') as f:
        configs = list(yaml.load_all(f, Loader=yaml.FullLoader))
        wandb_config = configs[0]["wandb_config"]
        data_config = configs[1]["data_config"]
        fed_config = configs[2]["fed_config"]          
        
    if rank == 0:
        printLog(f"I am server in {socket.gethostname()} rank {rank}")           
        ps=Server(size-1, fed_config["C"], fed_config["B"], fed_config["R"], fed_config["target_acc"], FLgroup)
        ps.setup(data_config)
        ps.start()
    
    else:
        printLog(f"I am client in {socket.gethostname()} rank {rank}")
        client = Client(size-1, fed_config["C"], fed_config["B"], fed_config["E"], fed_config["lr"], data_config["dataset_name"], FLgroup)
        client.setup()
        client.start()
        


def init_process(rank, size, backend='gloo'):

    FLgroup = dist.init_process_group(backend, rank=rank, world_size=size, init_method=f'tcp://{MASTER_ADDR}:{MASTER_PORT}')
    init_FL(rank, size, FLgroup)

if __name__ == "__main__":
    init_process(WORLD_RANK, WORLD_SIZE)