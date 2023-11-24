from methods.FedAvg import FedAvgServer
from utils.model_utils import TensorBuffer
from utils.utils import printLog

import torch.distributed as dist
import torch
import copy
import threading
import gc
import time

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class SAFAServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.local_model_version = [0 for idx in range(0,self.num_clients+1)]

        self.lag_tolerance = 5
        self.picked_history_per_client={}
        for idx in range(1, self.num_clients+1):
            self.picked_history_per_client[idx]=-1

        self.T_lim = 5000 #https://github.com/Jerrling02/ASFL/blob/main/FL/option.py
        self.Quota=int(self.num_clients*self.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수
        self.flatten_client_models={}
        self.cache={}

        self.num_cached_local_model_lock = threading.Lock()
        self.terminate_FL = threading.Event()
        self.terminate_FL.clear()

        self.terminate_background_thread = threading.Event()
        self.terminate_background_thread.clear()

        self.idle_clients=[]

    def setup(self, dataset, iid, split, cluster_type):
        super().setup(dataset, iid, split, cluster_type)

        profiling_group = dist.new_group([idx for idx in range(0,self.Quota+1)])
        
        T_dist=time.time()
        dist.broadcast(tensor=TensorBuffer(list(self.model.state_dict().values())).buffer, group=profiling_group)
        T_dist = time.time() - T_dist

        dist.broadcast(tensor=TensorBuffer(list(self.model.state_dict().values())).buffer, group=self.FLgroup)

        T_train_k=torch.zeros(1)
        T_train_k_list=[]
        
        for idx in range(1, self.num_clients+1):
            dist.irecv(tensor=T_train_k)
            T_train_k_list.append(T_train_k)
        
        T_max_train_k = max(T_train_k_list)

        self.T = min(self.T_lim, T_dist + T_max_train_k)


    def receive_local_model_from_any_clients(self):
        start = time.time()
        P=[]
        Q=[]
        while time.time() - start < self.T:
            if len(self.idle_clients) + self.num_cached_local_model == self.num_clients:
                break
            temp_local_model=TensorBuffer(list(self.model.state_dict().values()))
            req = dist.irecv(tensor=temp_local_model.buffer)
            req.wait()
            printLog("SERVER", f"CLIENT{req.source_rank()}에게 로컬 모델을 받음")

            self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
            
            if self.picked_history_per_client[req.source_rank()] == self.current_round-1:
                Q.append(req.source_rank())
            else:
                P.append(req.source_rank())

        return P, Q

    def CFCFM(self, P, Q):
        if len(P) < self.Quota:
            for i in range(0,self.Quota - len(P)):
                first_client_in_Q = Q.pop(0)
                P.append(first_client_in_Q)
        
        for idx in len(P):
            self.picked_history_per_client[idx] = self.current_round
        
        return P, Q
    
    def average_aggregation(self, coefficient):
        printLog("SERVER", "global aggregation을 진행합니다.")
        averaged_weights = OrderedDict()
        
        for idx in range(1, self.num_clients+1):
            local_weights = self.cache[idx]
            for key in local_weights.keys():
                if idx==0:
                    averaged_weights[key] = coefficient[idx] * local_weights[key]
                else:
                    averaged_weights[key] += coefficient[idx] * local_weights[key]

        self.model.load_state_dict(averaged_weights)
    
    def pre_aggregation_cache_update(self, P):
        
        for idx in range(1,self.num_clients):
            if idx in P:
                self.cache[idx]=copy.deepcopy(self.flatten_client_models[idx])
            elif self.picked_history_per_client[idx] < self.current_round-self.lag_tolerance:
                self.cache[idx]= copy.deepcopy(self.model.state_dict())
    
    def post_aggregation_cache_update(self, Q):
        for idx in range(1,self.num_clients):
            if idx in Q:
                self.cache[idx]=copy.deepcopy(self.flatten_client_models[idx])

    def send_global_model_to_clients(self):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(self.current_round)
        global_model_info = torch.tensor(global_model_info)

        for idx in range(1,self.num_clients+1):
            if self.local_model_version[idx] == self.current_round-1: # up-to-date client
                dist.send(tensor=flatten_model.buffer, dst=idx)
                self.local_model_version[idx]=self.current_round
            elif self.local_model_version[idx]<self.current_round-self.lag_tolerance: # deprecated client
                dist.send(tensor=flatten_model.buffer, dst=idx)
                self.local_model_version[idx]=self.current_round
    

