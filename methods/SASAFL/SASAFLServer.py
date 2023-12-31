from methods.FedAvg import FedAvgServer
from utils.model_utils import TensorBuffer
from utils.utils import printLog

import torch.distributed as dist
import torch
import copy
import threading
import random

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class SASAFLServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.local_model_version = [0 for idx in range(0,self.num_clients+1)]
        self.cached_client_idx = []
        self.num_cached_local_model = 0
        self.cached_client_idx_lock = threading.Lock()
        self.num_cached_local_model_lock = threading.Lock()
        self.terminate_FL = threading.Event()
        self.terminate_FL.clear()
        self.local_utility = {}

        self.last_global_loss = None

        self.idle_clients=[]

    def receive_local_model_from_any_clients(self):
        while not self.terminate_FL.is_set() or len(self.idle_clients) + self.num_cached_local_model < self.num_clients :
            temp_local_model=TensorBuffer(list(self.model.state_dict().values()))
            local_model_info = torch.zeros(len(temp_local_model.buffer)+2)

            req = dist.irecv(tensor=local_model_info) # [local model, utility, local_model version]
            req.wait()

            with self.cached_client_idx_lock and self.num_cached_local_model_lock:
                self.cached_client_idx.append(req.source_rank())
                self.num_cached_local_model += 1
                
                temp_local_model.buffer = local_model_info[:-2]
                self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
                self.local_utility[req.source_rank()] = local_model_info[-2]  
                self.local_model_version[req.source_rank()] = local_model_info[-1]

                printLog("SERVER", f"CLIENT {req.source_rank()}에게 로컬 모델을 받음 \n\
                                    => 로컬 모델 버전: {local_model_info[-1].item()} \n\
                                    => 로컬 utility: {local_model_info[-2]}")      

        printLog("SERVER" ,"백그라운드 스레드를 종료합니다.")

    def wait_until_can_update_global_model(self, num_local_model_limit):
        printLog("SERVER" , f"현재까지 받은 로컬 모델 개수: {self.num_cached_local_model}")
        printLog("SERVER", f"이번 라운드에서 {num_local_model_limit}개의 로컬 모델을 사용합니다.")
        
        while True:
            with self.cached_client_idx_lock and self.num_cached_local_model_lock:
                if self.num_cached_local_model >= num_local_model_limit:
                    
                    picked_client_idx = []
                    for idx in range(self.num_cached_local_model):
                        picked_client_idx.append(self.cached_client_idx.pop(0))
                    
                    self.num_cached_local_model = 0
                    
                    printLog("SERVER", f"picked client list : {picked_client_idx}")
                    
                    break

        return picked_client_idx
            

    def refine_received_local_model(self, picked_client_idx): 
        decay_coefficient = 0.9

        for idx in picked_client_idx:
            staleness = self.current_round - self.local_model_version[idx]
            local_coefficient = decay_coefficient**(staleness)
            local_model = self.model_controller.Model()
            local_model_state_dict=local_model.state_dict()
            self.flatten_client_models[idx].unpack(local_model_state_dict.values())

            global_model_state_dict = self.model.state_dict()

            interpolated_weights = OrderedDict()

            for key in global_model_state_dict.keys():
                interpolated_weights[key] = local_coefficient * local_model_state_dict[key]
            for key in global_model_state_dict.keys():
                interpolated_weights[key] += (1-local_coefficient) * global_model_state_dict[key]
            

            self.flatten_client_models[idx] = TensorBuffer(list(interpolated_weights.values()))

    def send_global_model_to_clients(self, clients_idx, picked_clients_idx, global_loss):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))

        shuffled_clients_idx = copy.deepcopy(clients_idx)
        random.shuffle(shuffled_clients_idx)
    

        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(self.current_round)
        global_model_info.append(global_loss)
        global_model_info = torch.tensor(global_model_info)


        for idx in shuffled_clients_idx:
            if idx in picked_clients_idx:
                dist.send(tensor=torch.cat([global_model_info, torch.tensor([1])]), dst=idx) # 글로벌 모델, 현재 라운드, global loss 전송, 선택되었음 전송
            else:
                dist.send(tensor=torch.cat([global_model_info, torch.tensor([0])]), dst=idx) # 글로벌 모델, 현재 라운드, global loss 전송, 선택되지 않음 전송


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

    def calculate_coefficient(self, picked_client_idx):
        
        
        ########## PM1 ##########
        coefficient = super().calculate_coefficient(picked_client_idx)
        #########################
        
        """
        ########## PM2 ##########
        data_coefficient = super().calculate_coefficient(picked_client_idx)
        
        refined_utility_sum=0
        decay_coefficient = 0.9

        refined_local_utility={}

        for idx in picked_client_idx:
            stalness = self.current_round - self.local_model_version[idx]
            refined_local_utility[idx] = self.local_utility[idx]*(decay_coefficient**(stalness))
       
        for idx in picked_client_idx:
            refined_utility_sum += refined_local_utility[idx]

        utility_coefficient = {}
        
        for idx in picked_client_idx:
            utility_coefficient[idx] = refined_local_utility[idx]/refined_utility_sum

        coefficient={}
        for idx in picked_client_idx:
            coefficient[idx] = (data_coefficient[idx] + utility_coefficient[idx])/2
        
        #########################
        """


        return coefficient
    
    def average_aggregation(self, selected_client_idx, coefficient):

        picked_client_info = ""
        for idx in selected_client_idx:
            picked_client_info += f"CLIENT {idx}의 staleness: {self.current_round - self.local_model_version[idx]}\n"
        printLog("SERVER", f"picked clients info: \n{picked_client_info}")
        super().average_aggregation(selected_client_idx, coefficient)

    def get_next_round_minimum_local_model(self, current_global_loss, current_num_picked_client, miminum_num_picked_client):
        if current_global_loss > self.last_global_loss:
            next_num_picked_client = min(int(current_num_picked_client*current_global_loss/self.last_global_loss), self.num_clients)

        else:
            next_num_picked_client = max(int(current_num_picked_client*current_global_loss/self.last_global_loss), miminum_num_picked_client)

        self.last_global_loss = current_global_loss
        return next_num_picked_client

    def terminate(self, clients_idx):
        self.terminate_FL.set()
        self.idle_clients += clients_idx
        while self.num_cached_local_model + len(clients_idx) < self.num_clients:
            continue

        temp_global_model=TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = torch.zeros(len(temp_global_model.buffer)+3)
        global_model_info[-3] = -1

        for idx in range(1,self.num_clients+1):
            dist.send(tensor=global_model_info, dst=idx)
        