from methods.FedAvg import FedAvgServer
from utils.utils import *

import torch
import torch.distributed as dist


class PowerOfChoiceServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)

    def calculate_coefficient(self, selected_client_idx):

        coefficient={}
        sum=0
        for idx in selected_client_idx:
            coefficient[idx]=self.len_local_dataset[idx]
            sum+=self.len_local_dataset[idx]
        
        for idx in selected_client_idx:
            coefficient[idx]=coefficient[idx]/sum

        return coefficient
    
    def client_select_pow_d(self, d, num_selected_clients):
        local_loss_list={}
        local_loss=torch.zeros(1)

        for idx in range(d):
            req=dist.irecv(tensor=local_loss)
            req.wait()
            local_loss_list[req.source_rank()] = local_loss.item()

        sorted_local_loss_list = sorted(local_loss_list.items(), key=lambda item:item[1], reverse=True)
        
        selected_clients_list=[]
        for idx in range(num_selected_clients):
            selected_clients_list.append(sorted_local_loss_list[idx][0])
        
        return selected_clients_list    
