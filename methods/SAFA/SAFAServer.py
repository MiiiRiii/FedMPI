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

        self.lag_tolerance = 3
        self.picked_history_per_client={}
        for idx in range(1, self.num_clients+1):
            self.picked_history_per_client[idx]=-1

        self.T_lim = 5000 # WISE
        # self.T_lim = 
        self.Quota=int(self.num_clients*self.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수
        self.flatten_client_models={}
        self.cache={}

        self.idle_clients=[]

    def setup(self, dataset, iid, split, cluster_type):
        super().setup(dataset, iid, split, cluster_type)

        profiling_group = dist.new_group([idx for idx in range(0,self.Quota+1)])
        
        T_dist=time.time()
        dist.broadcast(tensor=TensorBuffer(list(self.model.state_dict().values())).buffer, src=0, group=profiling_group)
        T_dist = time.time() - T_dist

        dist.broadcast(tensor=TensorBuffer(list(self.model.state_dict().values())).buffer, src=0, group=self.FLgroup)

        T_train_k=torch.zeros(1)
        T_train_k_list=[]
        
        for idx in range(1, self.num_clients+1):
            req = dist.irecv(tensor=T_train_k)
            req.wait()
            T_train_k_list.append(T_train_k)
        
        T_max_train_k = max(T_train_k_list)

        self.T = min(self.T_lim, T_dist + T_max_train_k)

        printLog(f"SERVER", f"한 라운드의 deadline은 {self.T}입니다.")

        for idx in range(1, self.num_clients+1):
            self.cache[idx] = TensorBuffer(list(self.model.state_dict().values()))


    def receive_local_model_from_any_clients(self):
        start = time.time()
        P=[]
        Q=[]
        while time.time() - start < self.T:
            printLog("SERVER", f"{time.time()-start}")
            temp_local_model=TensorBuffer(list(self.model.state_dict().values()))
            local_model_info = torch.zeros(len(temp_local_model.buffer)+1)

            req = dist.irecv(tensor=local_model_info)
            req.wait()
            printLog("SERVER", f"CLIENT{req.source_rank()}에게 로컬 모델을 받음")

            temp_local_model.buffer = local_model_info[:-1]
            self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
            if self.current_round>1 and self.picked_history_per_client[req.source_rank()] == self.current_round-1:
                Q.append(req.source_rank())
            else:
                P.append(req.source_rank())
            if len(P) >= self.Quota: break
        printLog("SERVER", f"[receive 단계] P {P}\n\
                                                      Q {Q}")
        return P, Q
    def CFCFM(self, P, Q):
        if len(P) < self.Quota:
            for i in range(0,self.Quota - len(P)):
                first_client_in_Q = Q.pop(0)
                P.append(first_client_in_Q)
        
        for idx in P:
            self.picked_history_per_client[idx] = self.current_round

        printLog("SERVER", f"[CFCFM 단계] P {P}\n\
                                                      Q {Q}")
        return P, Q
    
    def average_aggregation(self, P, coefficient):
        printLog("SERVER", "global aggregation을 진행합니다.")
        averaged_weights = OrderedDict()
        
        for idx, client_idx in enumerate(P):
            model = self.model_controller.Model()
            model_state_dict = model.state_dict()
            self.cache[client_idx].unpack(model_state_dict.values())
            for key in model_state_dict.keys():
                if idx==0:
                    averaged_weights[key] = coefficient[client_idx] * model_state_dict[key]
                else:
                    averaged_weights[key] += coefficient[client_idx] * model_state_dict[key]

        self.model.load_state_dict(averaged_weights)
    
    def pre_aggregation_cache_update(self, P):
        model = self.model_controller.Model()
        model_state_dict = model.state_dict()

        for idx in range(1,self.num_clients+1):
            if idx in P:
                self.flatten_client_models[idx].unpack(model_state_dict.values())
                self.cache[idx]=copy.deepcopy(self.flatten_client_models[idx])
            if self.current_round-self.lag_tolerance>=0 and self.local_model_version[idx] <= self.current_round-self.lag_tolerance:
                self.cache[idx]= copy.deepcopy(TensorBuffer(list(self.model.state_dict().values())))
    
    def post_aggregation_cache_update(self, Q):
        model = self.model_controller.Model()
        model_state_dict = model.state_dict()
        for idx in range(1,self.num_clients+1):
            if idx in Q:
                self.flatten_client_models[idx].unpack(model_state_dict.values())
                self.cache[idx]=copy.deepcopy(self.flatten_client_models[idx])

    def send_global_model_to_clients(self, P, Q=None):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(self.current_round)
        global_model_info = torch.tensor(global_model_info)

        for idx in range(1,self.num_clients+1):
            if self.current_round ==0:
                dist.send(tensor=flatten_model.buffer, dst=idx)
 
            elif (idx in P or idx in Q) and self.local_model_version[idx] == self.current_round-1: # up-to-date client
                printLog(f"SERVER", f"CLIENT {idx}는 up-to-date이므로 글로벌 모델을 전송합니다.")
                dist.send(tensor=flatten_model.buffer, dst=idx)
                self.local_model_version[idx] = self.current_round
            elif self.current_round - self.lag_tolerance>=0 and self.local_model_version[idx]<=self.current_round-self.lag_tolerance: # deprecated client
                printLog(f"SERVER", f"CLIENT {idx}는 deprecated이므로 글로벌 모델을 전송합니다.")
                dist.send(tensor=flatten_model.buffer, dst=idx)
                self.local_model_version[idx] = self.current_round
            else:
                printLog(f"SERVER", f"CLIENT {idx}는 tolerable이므로 글로벌 모델을 전송하지 않습니다.")
    
    def terminate_FL(self):
        temp_global_model=TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = torch.zeros(len(temp_global_model.buffer)+1)
        global_model_info[-1] = -1
        for idx in range(1, self.num_clients+1):
            dist.send(tensor=global_model_info, dst=idx)



