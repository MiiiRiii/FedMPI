from mpi4py import MPI
from Server import Server
from Client import Client
from utils import printLog

import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import os
import socket

def init_FL(rank, size, FLgroup, wandb_config, data_config, fed_config):           
    if rank == 0:
        printLog(f"I am server in {socket.gethostname()} rank {rank}")           
        ps=Server(fed_config["K"], fed_config["C"], fed_config["B"], fed_config["R"], fed_config["target_acc"], FLgroup)
        ps.setup(data_config)
        ps.start()
    
    else:
        printLog(f"I am client in {socket.gethostname()} rank {rank}")
        client = Client(fed_config["K"], fed_config["C"], fed_config["B"], fed_config["E"], fed_config["lr"], data_config["dataset_name"], FLgroup)
        client.setup()
        client.start()
        


def init_process(rank, size, fn, wandb_config, data_config, fed_config, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '295000'
    FLgroup = dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, FLgroup, wandb_config, data_config, fed_config)

if __name__ == "__main__":
    with open('/home/wise/miri/mpitest/FedAvg/config.yaml') as f:
        configs = list(yaml.load_all(f, Loader=yaml.FullLoader))
        wandb_config = configs[0]["wandb_config"]
        data_config = configs[1]["data_config"]
        fed_config = configs[2]["fed_config"]
    size = fed_config["K"]+1
    processes=[]
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target = init_process, args = (rank, size, init_FL,wandb_config, data_config, fed_config))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()   