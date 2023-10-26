from methods.FedAvg import FedAvgClient
from utils.utils import *

import torch.distributed as dist

import torch

class SemiAsyncClient(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
    
    def receive_global_model_from_server(self):
        global_model_version = torch.tensor(self.local_model_version).type(torch.FloatTensor)
        dist.recv(tensor=global_model_version, src=0)
        
        if global_model_version.item() == -1:
            return 0
        
        else :
            self.local_model_version = global_model_version.item()

            printLog(f"CLIENT {self.id} >> 로컬 모델 버전 : {int(self.local_model_version)}")
            super().receive_global_model_from_server()
            return 1
        
