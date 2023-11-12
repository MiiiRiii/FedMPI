from methods.FedAvg import FedAvgServer
from utils.model_utils import TensorBuffer
from utils.utils import printLog

import torch.distributed as dist
import torch
import copy
import threading
import gc

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class FedAsyncServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.terminate_FL_flag = threading.Event()
        self.terminate_FL_flag.clear()
        self.received_client_idx = []
        self.local_model_version = {}
        self.num_received_client = 0
        self.terminated_clients=[]

        self.received_client_idx_lock = threading.Lock()
        self.num_received_client_lock = threading.Lock()
        self.is_background_thread_terminate=False

    def receive_local_model_from_any_clients(self):
        while not self.terminate_FL_flag.is_set():
            temp_local_model=TensorBuffer(list(self.model.state_dict().values()))
            local_model_info = torch.zeros(len(temp_local_model.buffer)+1)
            req = dist.irecv(tensor=local_model_info)
            req.wait()

            printLog("SERVER", f"백그라운드 스레드에서 CLIENT{req.source_rank()}에게 로컬 모델을 받음")
            
            temp_local_model.buffer = local_model_info[:-1]
            self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
            self.local_model_version[req.source_rank()] = local_model_info[-1].item()
            with self.received_client_idx_lock and self.num_received_client_lock:
                self.received_client_idx.append(req.source_rank())
                self.num_received_client += 1

        self.is_background_thread_terminate=True

    
    def wait_until_can_update_global_model(self):
        while True:
            with self.received_client_idx_lock and self.num_received_client_lock:
                if self.num_received_client > 0:
                    client_idx = self.received_client_idx.pop(0)
                    self.num_received_client-=1
                    printLog("SERVER", f"CLIENT {client_idx}의 로컬 모델을 사용해서 글로벌 업데이트를 수행합니다.")
                    break

        return client_idx

    def global_update(self, client_idx):
        decay_coefficient = 0.9
        staleness = self.current_round - self.local_model_version[client_idx]
        local_coefficient = decay_coefficient**(staleness)
        local_model = self.model_controller.Model()
        local_model_state_dict = local_model.state_dict()
        self.flatten_client_models[client_idx].unpack(local_model_state_dict.values())

        global_model_state_dict = self.model.state_dict()

        interpoloated_weights = OrderedDict()

        for key in global_model_state_dict.keys():
            interpoloated_weights[key] = local_coefficient*local_model_state_dict[key]
        for key in global_model_state_dict.keys():
            interpoloated_weights[key] += (1-local_coefficient)*global_model_state_dict[key]

        self.model.load_state_dict(interpoloated_weights)

    def send_global_model_to_client(self, client_idx):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(self.current_round)
        global_model_info = torch.tensor(global_model_info)

        dist.send(tensor=global_model_info, dst=client_idx)
        
    def terminate_FL(self, sent_client_idx):
        self.terminate_FL_flag.set()
        
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(-1)
        global_model_info = torch.tensor(global_model_info)

        temp_local_model = TensorBuffer(list(self.model.state_dict().values()))
        local_model_info = torch.zeros(len(temp_local_model.buffer)+1)


        while not self.is_background_thread_terminate :
            pass
        
        self.received_client_idx.insert(0, sent_client_idx)
        for idx in self.received_client_idx:
            dist.send(tensor=global_model_info, dst=idx)
            printLog("SERVER", f"CLIENT {idx}에게 끝났음을 알림")
            self.terminated_clients.append(idx)

        
        printLog("SERVER", "백그라운드 스레드 종료됨")
        while len(self.terminated_clients) < self.num_clients :
            printLog("SERVER", f"{local_model_info.size()}")
            req=dist.irecv(tensor=local_model_info)
            req.wait()
            sent_client_idx = req.source_rank()
            
            dist.send(tensor=global_model_info, dst=sent_client_idx)
            self.terminated_clients.append(sent_client_idx)

        