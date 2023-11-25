from methods.FedAvg import FedAvgClient
from utils.utils import *

from utils.model_utils import TensorBuffer
import torch.distributed as dist

import torch
import time
import copy
import threading

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

class SAFAClient(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
        self.len_local_dataset = len(dataset)

        self.replace_global_model_during_local_update = threading.Event() # 학습 중간에 글로벌 모델로 교체하는지 확인하는 용도
        self.replace_global_model_during_local_update.clear()

    def setup(self, cluster_type):
        super().setup(cluster_type)
        
        profiling_group = dist.new_group([idx for idx in range(0,self.num_selected_clients+1)])
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        profiling_model_state_dict = self.model.state_dict()
        dist.broadcast(tensor=flatten_model.buffer, src=0, group=profiling_group)

        dist.broadcast(tensor=flatten_model.buffer, src=0, group=self.FLgroup)

        flatten_model.unpack(profiling_model_state_dict.values())
        self.model.load_state_dict(profiling_model_state_dict)

        self.train()

        T_train_k = self.total_train_time

        self.total_train_time=0

        dist.send(tensor=torch.tensor(T_train_k), dst=0)
        

    
    def receive_global_model_from_server(self, is_ongoing_local_update_flag):

        self.received_global_model = self.model_controller.Model()

        model_state_dict =self.model.state_dict()
        
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        self.global_model_info = torch.zeros(len(flatten_model.buffer)+1)

        while True:
            dist.recv(tensor=self.global_model_info, src=0)

            flatten_model.buffer = self.global_model_info[:-1]
            flatten_model.unpack(model_state_dict.values())
            global_model_version = int(self.global_model_info[-1].item())

            if global_model_version == -1:
                break
            
            elif is_ongoing_local_update_flag.is_set():
                printLog(f"CLIENT {self.id}", "학습 도중에 글로벌 모델을 받았습니다.")
                self.received_global_model.load_state_dict(model_state_dict)
                self.replace_global_model_during_local_update.set()
                
            else:
                self.model.load_state_dict(model_state_dict)
                self.local_model_version = global_model_version
                is_ongoing_local_update_flag.set()

    def train(self, terminate_flag=None):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            

        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        loss_function = CrossEntropyLoss()

        e=0
        is_terminate_FL=0
        while e < self.local_epoch:
            if self.replace_global_model_during_local_update.is_set(): # 학습 중간에 글로벌 모델로 교체
                e=0
                
                self.local_model_version = int(self.global_model_info[-4].item())
                self.last_global_loss = self.global_model_info[-3].item()
                self.model = copy.deepcopy(self.received_global_model)
                
                self.replace_global_model_during_local_update.clear()
                continue
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
  
                loss.backward()
                
                optimizer.step()
            e+=1
            printLog(f"CLIENT {self.id}", f"{e} epoch을 수행했습니다.")
            if terminate_flag!= None and terminate_flag.is_set():
                printLog(f"CLIENT {self.id}", f"학습 도중에 FL 프로세스가 종료되어 로컬 학습을 멈춥니다.")
                is_terminate_FL = -1
                break

        self.total_train_time += time.time()-start
        return is_terminate_FL
    
    def send_local_model_to_server(self):
        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        local_model_info = flatten_model.buffer.tolist()
        local_model_info.append(self.local_model_version)
        local_model_info = torch.tensor(local_model_info)
        dist.send(tensor=local_model_info, dst=0)