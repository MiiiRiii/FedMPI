from methods.FedAvg import FedAvgClient
from utils.utils import *
from utils.model_utils import TensorBuffer

import torch.distributed as dist

import torch
import time
import math

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

class SemiAsyncPM3Client(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
        self.len_local_dataset = len(dataset)
    
    def receive_global_model_from_server(self):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        model_state_dict =self.model.state_dict()
        global_model_info = torch.zeros(len(flatten_model.buffer)+1)
        dist.recv(tensor=global_model_info, src=0)
        
        if global_model_info[-1].item() == -1:
            return 0
        
        else :
            self.local_model_version = global_model_info[-1].item()
            flatten_model.buffer = global_model_info[:-1]
            flatten_model.unpack(model_state_dict.values())
            self.model.load_state_dict(model_state_dict)
            printLog(f"CLIENT {self.id}", f"로컬 모델 버전 : {int(self.local_model_version)}")
            
            return 1

    def train(self):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            

        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)
       

        ########## my ##########
        loss_function = CrossEntropyLoss()
        epoch_train_loss = 0.0
        ########################

        for e in range(self.local_epoch):
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)

                ########## my ##########
                if e==1:
                    epoch_train_loss += loss.detach().item()
                loss.backward()
                ########################
                
                optimizer.step()
            printLog(f"CLIENT {self.id}", f"{e+1} epoch을 수행했습니다.")

        self.total_train_time += time.time()-start

        utility = math.sqrt(epoch_train_loss / self.len_local_dataset) * self.len_local_dataset

        printLog(f"CLIENT {self.id}", f"local utility: {utility}")

        return utility

    def terminate(self):
        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0) # 서버 데몬 스레드 죽이기 용
