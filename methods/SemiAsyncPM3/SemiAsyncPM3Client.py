from methods.FedAvg import FedAvgClient
from utils.utils import *

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
        global_model_version = torch.tensor(self.local_model_version).type(torch.FloatTensor)
        dist.recv(tensor=global_model_version, src=0)
        
        if global_model_version.item() == -1:
            return 0
        
        else :
            self.local_model_version = global_model_version.item()

            printLog(f"CLIENT {self.id}", f"로컬 모델 버전 : {int(self.local_model_version)}")
            super().receive_global_model_from_server()
            return 1

    def train(self):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            

        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        """
        ########## oort ##########
        loss_function = CrossEntropyLoss(reduction='none')
        epoch_train_loss = None
        loss_decay = 0.2
        ##########################
        """

        ########## my ##########
        loss_function = CrossEntropyLoss()
        epoch_train_loss = 0.0
        ########################

        for e in range(self.local_epoch):
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                """
                ########## oort ##########
                temp_loss = 0.
                loss_cnt = 1.

                loss_list = loss.tolist()
                for l in loss_list:
                    temp_loss += l**2

                loss_cnt = len(loss_list)

                temp_loss = temp_loss/float(loss_cnt)
    
                if e==1: # only measure the loss of the first epoch
                    if epoch_train_loss is None:
                        epoch_train_loss = temp_loss
                    else:
                        epoch_train_loss = (1. - loss_decay) * epoch_train_loss + loss_decay * temp_loss
                loss.mean().backward()
                ##########################
                """

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
        # 클라이언트 1이 대표로 실행
        if self.id == 1:
            isTerminate = torch.tensor(1).type(torch.FloatTensor)
            dist.recv(tensor = isTerminate, src=0)
            if isTerminate == 0:
                self.send_local_model_to_server()
            