from Server import Server
from Client import Client
from utils.utils import printLog

from methods import FedAvg
from methods import CHAFL
from methods import powerofchoice

import torch.distributed as dist
import torch.multiprocessing as mp
import os
import socket
import wandb
import argparse
import time


MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def init_FL(FLgroup, args): 
    for itr in range(1):
        method=None
        if args.method=="FedAvg":
            method = FedAvg.FedAvg()
        elif args.method=="CHAFL":
            method = CHAFL.CHAFL()
        elif args.method=="rpow_d" or args.method=="cpow_d" or args.method=="pow_d":
            method=powerofchoice.rpow_d(args.method, args.d)


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
                    "method": args.method,
                    "system_heterogeneity": args.system_heterogeneity
                })
            printLog(f"I am server in {socket.gethostname()} rank {WORLD_RANK}")           
            ps=Server(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            ps.setup(args.dataset, args.iid, args.split, args.system_heterogeneity)

            
            method.runServer(ps)
            if args.wandb_on == True:
                wandb.finish()
        else:
            #torch.set_num_threads(args.omp_num_threads)
            printLog(f"I am client in {socket.gethostname()} rank {WORLD_RANK}")
            client = Client(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)
            client.setup()
            method.runClient(client)

        
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
    parser.add_argument("--system_heterogeneity", choices=[0,1,2], default=0, type=int)

    parser.add_argument("--dataset", choices=['MNIST', 'CIFAR10', 'FashionMNIST'], default='CIFAR10', type=str)
    parser.add_argument("--iid", choices=['True', 'False'], default='False', type=str)
    parser.add_argument("--split", choices=['uniform', 'gaussian'], default='gaussian', type=str)

    parser.add_argument("--method", choices=['FedAvg', 'CHAFL','pow_d','cpow_d'], default='FedAvg', type=str)
    parser.add_argument("--rpow_d", type=int)
    
    parser.add_argument("--wandb_on", choices=['True', 'False'], default='False', type=str)
    parser.add_argument("--project",type=str)
    parser.add_argument("--entity",type=str)
    parser.add_argument("--group",type=str)
    parser.add_argument("--name",type=str)
    
    args=parser.parse_args()
    init_process(args)
    
