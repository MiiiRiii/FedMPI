from methods.FedAvg import FedAvgClient

import torch.distributed as dist

import torch

class SemiAsyncClient(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
    
    def receive_global_model_from_server(self):
        super().receive_global_model_from_server()
        
        global_model_version = torch.zeros(1)
        dist.recv(tensor=global_model_version, src=0)
        self.local_model_version = global_model_version.item()