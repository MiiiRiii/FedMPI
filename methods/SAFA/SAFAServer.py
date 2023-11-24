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

class SemiAsyncServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.local_model_version = [0 for idx in range(0,self.num_clients+1)]

        self.lag_tolerance = 5
        self.picked_history_per_client={}
        for idx in range(1, self.num_clients+1):
            self.picked_history_per_client[idx]=-1

        self.T = 1000
        self.Quota=int(self.num_clients*self.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수

        self.num_cached_local_model_lock = threading.Lock()
        self.terminate_FL = threading.Event()
        self.terminate_FL.clear()

        self.terminate_background_thread = threading.Event()
        self.terminate_background_thread.clear()

        self.idle_clients=[]

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
    
    def pre_aggregation_cache_update(self, P):

        cache=[]
        #for k in self.Quota:

        
    
    def post_aggregation_cache_update(self):
        pass

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
    


    def evaluate_local_model(self, client_idx):
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.test_data, self.batch_size)

        model = self.model_controller.Model()
        model_state_dict = model.state_dict()
        self.flatten_client_models[client_idx].unpack(model_state_dict.values())
        model.load_state_dict(model_state_dict)

        model.eval()

        test_loss, correct = 0, 0

        for data, labels in dataloader:
            outputs = model(data)
            test_loss = test_loss + loss_function(outputs, labels).item()

            predicted = outputs.argmax(dim=1, keepdim=True)

            correct = correct + predicted.eq(labels.view_as(predicted)).sum().item()

        test_loss = test_loss / len(dataloader)
        test_accuracy = correct / len(self.test_data)

        printLog("Server", f"CLIENT{client_idx}의 로컬 모델 평가 결과 : acc=>{test_accuracy}, test_loss=>{test_loss}")

    def average_aggregation(self, selected_client_idx, coefficient):
        for idx in selected_client_idx:
            printLog("SERVER", f"CLIENT {idx}의 staleness는 {self.current_round - self.local_model_version[idx]}입니다.")
        return super().average_aggregation(selected_client_idx, coefficient)

    def terminate(self, clients_idx):
        self.idle_clients+=clients_idx
        while self.num_cached_local_model + len(clients_idx) < self.num_clients:
            pass

        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = torch.zeros(len(flatten_model.buffer)+1)
        global_model_info[-1]=-1
        for idx in range(1,self.num_clients+1):
            dist.send(tensor=global_model_info, dst=idx)
