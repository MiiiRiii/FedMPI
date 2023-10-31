from methods.FedAvg import FedAvgClient
from utils.utils import *

import torch.distributed as dist

import torch
import time
import threading

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

class SemiAsyncPM1Client(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
        self.last_global_loss=100.
        self.current_local_epoch = self.local_epoch
        self.receive_global_model_flag = threading.Event()
        self.receive_global_model_flag.clear()

    def receive_global_model_from_server(self):
        global_model_info = torch.zeros(2)

        while True:
            dist.recv(tensor=global_model_info, src=0) #global_model_info=[글로벌 모델 버전, global loss]
            if global_model_info[0].item() == -1: #FL 프로세스 종료
                break
            else:
                if global_model_info[1].item() < self.last_global_loss :
                    printLog(f"CLIENT {self.id}", f"gl: {global_model_info[1].item()}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받습니다.")
                
                    super().receive_global_model_from_server()
                    self.local_model_version = global_model_info[0].item()
                    self.last_global_loss = global_model_info[1].item()
                    self.receive_global_model_flag.set()
                    dist.send(tensor=torch.tensor(1), dst=0) # 서버에게 로컬 모델 버전을 교체하라고 알려줌
                else:
                    printLog(f"CLIENT {self.id}", f"gl: {global_model_info[1].item()}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받지 않고 로컬 업데이트를 이어갑니다.")
                    dist.send(tensor=torch.tensor(0), dst=0) # 서버에게 로컬 모델 버전을 교체하지 말라고 알려줌
                    continue


    def train(self):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            
        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss(reduction='none')
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        ########## oort ##########
        epoch_train_loss = None
        loss_decay = 0.2
        ##########################
        
        e=0
        while e < self.current_local_epoch:
            if self.receive_global_model_flag.is_set(): # 학습 중간에 글로벌 모델을 받았다면 교체
                e=0
                self.current_local_epoch -= 1
                self.receive_global_model_flag.clear()
                continue
            
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                
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
                ##########################

                loss.mean().backward()
                optimizer.step()
                e+=1

            printLog(f"CLIENT {self.id}", f"{e+1} epoch을 수행했습니다.")
            
        self.total_train_time += time.time()-start
        printLog(f"CLIENT {self.id}", f"epoch train loss: {epoch_train_loss}")

        return epoch_train_loss
    
    
    def terminate(self):
        # 클라이언트 1이 대표로 실행
        if self.id == 1:
            isTerminate = torch.tensor(1).type(torch.FloatTensor)
            dist.recv(tensor = isTerminate, src=0)
            if isTerminate == 0:
                self.send_local_model_to_server()
    
    def send_local_model_to_server(self):
        self.current_local_epoch = self.local_epoch
        return super().send_local_model_to_server()