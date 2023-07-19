from utils import *

import torch.distributed as dist
import torch
import wandb

class CHAFL(object):
    def client(self, Client):
        while True:
            Client.receive_global_model_from_server()
            Client.evaluate()
            selected=False
            selected_clients=torch.zeros(Client.num_selected_clients).type(torch.int64)
            dist.broadcast(tensor=selected_clients, src=0, group=Client.FLgroup)
            
            for idx in selected_clients:
                if idx == Client.id:
                    selected=True
                    break

            if(selected):
                Client.train()
                printLog(f"CLIENT {Client.id} >> 평균 학습 소요 시간: {Client.total_train_time/Client.num_of_selected}")
                Client.send_local_model_to_server()

            dist.barrier()

            continueFL = torch.zeros(1)
            dist.broadcast(tensor=continueFL, src=0, group=Client.FLgroup)

            if(continueFL[0]==0): #FL 종료
                break
    def server(self, Server):
        None