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
    def __init__(self, num_clients, selection_ratio, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(int(num_clients * selection_ratio), 40, local_epoch, 0.001, dataset, FLgroup)
        self.num_clients = num_clients
        self.local_model_version=0
        self.len_local_dataset = len(dataset)


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
    
    def receive_global_model_from_server(self, terminate_FL):
        model_state_dict = self.model.state_dict()
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        self.global_model_info = torch.zeros(len(flatten_model.buffer)+1)

        while True:
    
            dist.recv(tensor=self.global_model_info, src=0)
            global_model_version=int(self.global_model_info[-1].item())
            if global_model_version == -1:
                terminate_FL.set()
                break
            
            flatten_model.buffer=self.global_model_info[:-1]
            flatten_model.unpack(model_state_dict.values())
            




    def train(self, terminate_flag=None):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            

        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        loss_function = CrossEntropyLoss()

        for e in range(self.local_epoch):
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            printLog(f"CLIENT {self.id}", f"{e+1} epoch을 수행했습니다.")
        self.total_train_time += time.time()-start

    def send_local_model_to_server(self):

        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)