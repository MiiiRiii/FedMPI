from methods.FedAvg import FedAvg, FedAvgClient, FedAvgServer
from methods.CHAFL import CHAFL, CHAFLClient, CHAFLServer
from methods.PowerOfChoice import PowerOfChoice, PowerOfChoiceClient, PowerOfChoiceServer
from methods.SemiAsyncFL import SemiAsync, SemiAsyncClient, SemiAsyncServer
from methods.SemiAsyncPM1 import SemiAsyncPM1, SemiAsyncPM1Client, SemiAsyncPM1Server
from methods.FedAsync import FedAsync, FedAsyncClient, FedAsyncServer
from methods.LossUtilityFedAvg import LossUtilityFedAvg, LossUtilityFedAvgClient, LossUtilityFedAvgServer
from methods.SemiAsyncPM3 import SemiAsyncPM3, SemiAsyncPM3Client, SemiAsyncPM3Server
from methods.SASAFL import SASAFL, SASAFLClient, SASAFLServer

from utils.utils import printLog

import torch.distributed as dist
import os
import socket
import wandb
import argparse
import torch
import gc


MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])

def init_FL(FLgroup, args): 
    for itr in range(args.repeat):

        if args.cluster_type == "KISTI":
            num_thread = [item for item in args.num_threads]
            print(f"Process {WORLD_RANK} uses {num_thread[LOCAL_RANK]} threads")
            torch.set_num_threads(int(num_thread[LOCAL_RANK]))
        else:
            num_thread = [-1 for idx in range(WORLD_SIZE)]

        method=None
        if args.method=="FedAvg":
            method = FedAvg.FedAvg()
            Server = FedAvgServer.FedAvgServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = FedAvgClient.FedAvgClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)

        elif args.method=="CHAFL":
            method = CHAFL.CHAFL()
            Server = CHAFLServer.CHAFLServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = CHAFLClient.CHAFLClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)

        elif args.method=="rpow_d" or args.method=="cpow_d" or args.method=="pow_d":
            method = PowerOfChoice.PowerOfChoice(args.method, args.d)
            Server = PowerOfChoiceServer.PowerOfChoiceServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = PowerOfChoiceClient.PowerOfChoiceClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)
        
        elif args.method=="SemiAsync":
            method = SemiAsync.SemiAsync()
            Server = SemiAsyncServer.SemiAsyncServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = SemiAsyncClient.SemiAsyncClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)

        elif args.method=="SemiAsyncPM1":
            method = SemiAsyncPM1.SemiAsyncPM1()
            Server = SemiAsyncPM1Server.SemiAsyncPM1Server(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = SemiAsyncPM1Client.SemiAsyncPM1Client(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)

        elif args.method=="FedAsync":
            method = FedAsync.FedAsync()
            Server = FedAsyncServer.FedAsyncServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = FedAsyncClient.FedAsyncClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)

        elif args.method=="LossUtilityFedAvg":
            method = LossUtilityFedAvg.LossUtilityFedAvg()
            Server = LossUtilityFedAvgServer.LossUtilityFedAvgServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = LossUtilityFedAvgClient.LossUtilityFedAvgClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)

        elif args.method=="SemiAsyncPM3":
            method = SemiAsyncPM3.SemiAsyncPM3()
            Server = SemiAsyncPM3Server.SemiAsyncPM3Server(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = SemiAsyncPM3Client.SemiAsyncPM3Client(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup) 

        elif args.method=="SASAFL":
            method = SASAFL.SASAFL()
            Server = SASAFLServer.SASAFLServer(WORLD_SIZE-1, args.selection_ratio, args.batch_size, args.round, args.target_acc, args.wandb_on, FLgroup)
            Client = SASAFLClient.SASAFLClient(int((WORLD_SIZE-1)*args.selection_ratio), args.batch_size, args.local_epochs, args.lr, args.dataset, FLgroup)       
        
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
                    "selection_ratio": args.selection_ratio,
                    "d": args.d,
                    "cluster_type":args.cluster_type,
                })
            printLog("MAIN",f"I am server in {socket.gethostname()} rank {WORLD_RANK}")           
            Server.setup(args.dataset, args.iid, args.split, args.cluster_type)

            
            method.runServer(Server)
            if args.wandb_on == "True":
                wandb.finish()
        else:
            printLog("MAIN",f"I am client in {socket.gethostname()} rank {WORLD_RANK}")
            Client.setup(args.cluster_type)
            method.runClient(Client)

        del Server
        del Client
        del method
        gc.collect()
        
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

    parser.add_argument("--dataset", choices=['MNIST', 'CIFAR10', 'FashionMNIST'], default='CIFAR10', type=str)
    parser.add_argument("--iid", choices=['True', 'False'], default='False', type=str)
    parser.add_argument("--split", choices=['uniform', 'gaussian'], default='gaussian', type=str)

    parser.add_argument("--method", choices=['FedAvg', 'CHAFL','pow_d','cpow_d', 'SemiAsync', 'SemiAsyncPM1', 'FedAsync','LossUtilityFedAvg', 'SemiAsyncPM3', 'SASAFL'], default='FedAvg', type=str)
    parser.add_argument("--d", type=int)
    
    parser.add_argument("--wandb_on", choices=['True', 'False'], default='False', type=str)
    parser.add_argument("--project",type=str)
    parser.add_argument("--entity",type=str)
    parser.add_argument("--group",type=str)
    parser.add_argument("--name",type=str)

    parser.add_argument("--repeat",type=int)

    parser.add_argument("--cluster_type", choices=['WISE', 'KISTI'], type=str)
    parser.add_argument("--num_threads", type=str)

    args=parser.parse_args()
    init_process(args)
    
