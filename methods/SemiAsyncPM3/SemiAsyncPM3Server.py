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

class SemiAsyncPM3Server(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.local_model_version = [0 for idx in range(0,self.num_clients+1)]
        self.cached_client_idx = []
        self.num_cached_local_model = 0
        self.cached_client_idx_lock = threading.Lock()
        self.num_cached_local_model_lock = threading.Lock()
        self.terminate_FL = threading.Event()
        self.terminate_FL.clear()
        
        self.last_global_loss = 1.0

        self.terminate_background_thread = threading.Event()
        self.terminate_background_thread.clear()

        self.idle_clients=[]

    def receive_local_model_from_any_clients(self):
        while not self.terminate_FL.is_set() or len(self.idle_clients) + self.num_cached_local_model < self.num_clients :
            #if len(self.idle_clients) + self.num_cached_local_model == self.num_clients:
            #    break
            temp_local_model=TensorBuffer(list(self.model.state_dict().values()))
            req = dist.irecv(tensor=temp_local_model.buffer)
            req.wait()
            printLog("SERVER", f"CLIENT{req.source_rank()}에게 로컬 모델을 받음")
            
            self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
            with self.cached_client_idx_lock and self.num_cached_local_model_lock:
                self.cached_client_idx.append(req.source_rank())
                self.num_cached_local_model += 1
        self.terminate_background_thread.set()
        printLog("SERVER" ,"백그라운드 스레드를 종료합니다.")

    def wait_until_can_update_global_model(self, num_local_model_limit):
        printLog("SERVER" , f"현재까지 받은 로컬 모델 개수: {self.num_cached_local_model}")
        # get most staled client

        most_staled_client=self.local_model_version.index(min(self.local_model_version))+1
        #printLog("SERVER", f"이번 라운드에서 {num_local_model_limit}개의 로컬 모델을 사용하며, 가장 stale 된 클라이언트의 ID는 {most_staled_client}입니다.")
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

    def send_global_model_to_clients(self, sucess_uploaded_client_idx):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(self.current_round)
        global_model_info = torch.tensor(global_model_info)
        for idx in sucess_uploaded_client_idx:
            dist.send(tensor=global_model_info, dst=idx) # 글로벌 모델 전송
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
    

    def get_next_round_minimum_local_model(self, current_global_loss, current_num_picked_client, minimum_num_picked_client):

        """
        
        # get current global_loss_gap
        current_global_loss_gap = current_global_loss - self.last_global_loss

        # get delta
        self.sum_global_loss_gap += abs(current_global_loss_gap)
        delta = self.sum_global_loss_gap/self.current_round

        next_num_picked_client = minimum_num_picked_client
        if delta < abs(current_global_loss_gap) and current_global_loss >0: # global loss가 증가했다면 aggregation에 더 많은 로컬 모델을 사용
            next_num_picked_client=int(current_num_picked_client*(abs(current_global_loss_gap)/delta))
            print("SERVER", f"current_global_loss_gap: {current_global_loss_gap}")
            print("SERVER", f"delta: {delta}")
        """
        if current_global_loss > self.last_global_loss:
            next_num_picked_client = min(int(current_num_picked_client * current_global_loss / self.last_global_loss), self.num_clients)
        else:
            next_num_picked_client = max(int(current_num_picked_client * current_global_loss / self.last_global_loss), minimum_num_picked_client)

        self.last_global_loss = current_global_loss
        
        return min(next_num_picked_client, self.num_clients)

    def terminate(self, clients_idx):
        self.terminate_FL.set()
        self.idle_clients+=clients_idx
        while self.num_cached_local_model + len(clients_idx) < self.num_clients:
            pass
        
        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = torch.zeros(len(flatten_model.buffer)+1)
        global_model_info[-1]=-1
        for idx in range(1,self.num_clients+1):
            dist.send(tensor=global_model_info, dst=idx)


