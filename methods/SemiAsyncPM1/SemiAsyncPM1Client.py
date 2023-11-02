from methods.FedAvg import FedAvgClient
from utils.utils import *

import torch.distributed as dist

import torch
import time
import threading
import math

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
        self.len_local_dataset = len(dataset)

    def receive_global_model_from_server(self, can_local_update_flag, terminate_FL_flag):
        global_model_info = torch.zeros(2)

        while True:
            dist.recv(tensor=global_model_info, src=0) #global_model_info=[글로벌 모델 버전, global loss]

            
            if int(global_model_info[0].item()) == -1: #FL 프로세스 종료
                terminate_FL_flag.set()
                break

            elif can_local_update_flag.is_set() and int(global_model_info[0].item())>=2: # 로컬 학습 중에 글로벌 모델을 받는 경우
                

                if global_model_info[1].item() < self.last_global_loss: # 새로운 글로벌 모델이 더 퀄리티가 좋은 경우
                    printLog(f"CLIENT {self.id}", f"gl: {global_model_info[1].item()}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받습니다.")
                
                    super().receive_global_model_from_server()
                    self.local_model_version = int(global_model_info[0].item())
                    self.last_global_loss = global_model_info[1].item()
                    self.receive_global_model_flag.set()


                else: # 현재 학습 중인 모델이 더 퀄리티가 좋은 경우
                    printLog(f"CLIENT {self.id}", f"gl: {global_model_info[1].item()}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받지 않고 로컬 업데이트를 이어갑니다.")
                    continue

            else: # 처음 라운드이거나 정상적으로 로컬 모델을 업로드 한 후 글로벌 모델을 기다리고 있는 경우
                super().receive_global_model_from_server()
                self.local_model_version = int(global_model_info[0].item())
                self.last_global_loss = global_model_info[1].item()
                can_local_update_flag.set()
        

    def train(self):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            
        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        epoch_train_loss = 0.0
        
        
        e=0
        while e < self.current_local_epoch:
            if self.receive_global_model_flag.is_set(): # 학습 중간에 글로벌 모델을 받았다면 교체
                e=0
                epoch_train_loss = 0.0
                self.current_local_epoch -= 1
                self.receive_global_model_flag.clear()
                continue
            
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                
                if e==0:
                    epoch_train_loss += loss.detach().item()

                loss.backward()
                optimizer.step()
            e+=1

            printLog(f"CLIENT {self.id}", f"{e} epoch을 수행했습니다.")
            
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
    
    def send_local_model_to_server(self, utility):
        self.current_local_epoch = self.local_epoch
        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        local_model_info = flatten_model.buffer.tolist()
        local_model_info.append(utility)
        local_model_info.append(self.local_model_version)
        local_model_info = torch.tensor(local_model_info)
        print(local_model_info.size())
        dist.send(tensor=local_model_info, dst=0)