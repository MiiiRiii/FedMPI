from Server import Server
from Client import Client
from utils import printLog

import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import os
import socket
import wandb
import argparse

MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def init_FL(FLgroup, args): 
    if WORLD_RANK == 0:
        if args.wandb_on == "True":
            wandb.init(project=args.project, entity=args.entity, group=args.group, name=args.name,
                    config={
                "num_clients": WORLD_SIZE-1,
                "batch_size": args.batch_size,
                "local_epoch": args.local_epochs,
                "learning_rate": args.lr,
                "dataset": args.dataset,
                "data_split": args.split,
            })
        printLog(f"I am server in {socket.gethostname()} rank {WORLD_RANK}")           
        ps=Server(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
        ps.setup(args.dataset, args.iid, args.split)
        ps.start()
    
    else:
        printLog(f"I am client in {socket.gethostname()} rank {WORLD_RANK}")
        client = Client(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)
        client.setup()
        client.start()
    
    if args.wandb_on == True:
        wandb.finish()
        
def init_process(args, backend='gloo'):
    FLgroup = dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE, init_method=f'tcp://{MASTER_ADDR}:{MASTER_PORT}')
    init_FL(FLgroup, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_ratio", type=float)
    parser.add_argument("--round", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--local_epochs", type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--target_acc", type=float)
    parser.add_argument("--omp_num_threads", type=str)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--iid", type=str)
    parser.add_argument("--split", type=str)

    parser.add_argument("--wandb_on", type=str)
    parser.add_argument("--project",type=str)
    parser.add_argument("--entity",type=str)
    parser.add_argument("--group",type=str)
    parser.add_argument("--name",type=str)
    
    args=parser.parse_args()
    init_process(args)
    