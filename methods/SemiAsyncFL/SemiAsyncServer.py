from methods.FedAvg import FedAvgServer
from utils.model_utils import TensorBuffer
from utils.utils import printLog

import torch.distributed as dist
import torch
import copy

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class SemiAsyncServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.local_model_version = [0 for idx in range(0,self.num_clients+1)]
        self.cached_client_idx = []
        self.num_cached_local_model = 0
            
    """
    def receive_local_model_from_any_clients(self, num_clients, num_local_model_limit):
        picked_client_idx=[]
        remain_res=[]
        cnt=0
        for idx in range(num_clients):
            temp_local_model =TensorBuffer(list(self.model.state_dict().values()))
            req = dist.irecv(tensor=temp_local_model.buffer)
            if cnt < num_local_model_limit:
                req.wait()
                self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
                printLog(f"Server >> CLIENT {req.source_rank()}에게 local model을 받음")
                cnt=cnt+1
                picked_client_idx.append(req.source_rank())
            else:
                remain_res.append(req)
        
        return picked_client_idx, remain_res
    """

    def receive_local_model_from_any_clients(self):
        while True:
            temp_local_model=TensorBuffer(list(self.model.state_dict().values()))
            req = dist.irecv(tensor=temp_local_model.buffer)
            req.wait()
            self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
            self.cached_client_idx.append(req.source_rank())
            self.num_cached_local_model += 1

    def wait_until_can_update_global_model(self, num_local_model_limit):
        printLog(f"SERVER >> 현재까지 받은 로컬 모델 개수: {self.num_cached_local_model}")
        while True:
            if self.num_cached_local_model == num_local_model_limit:
                break
        
        self.num_cached_local_model -= num_local_model_limit
        
        picked_client_idx = []

        for idx in range(num_local_model_limit):
            picked_client_idx.append(self.cached_client_idx.pop(0))
    
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
    """
    def calculate_coefficient(self, picked_client_idx):

        coefficient={}
        for idx in picked_client_idx : 
            coefficient[idx] = self.len_local_dataset[idx] / self.len_total_local_dataset
        
        return coefficient
    """
    def send_global_model_to_clients(self, sucess_uploaded_client_idx):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        for idx in sucess_uploaded_client_idx:
            dist.send(tensor=flatten_model.buffer, dst=idx) # 글로벌 모델 전송
            dist.send(tensor=torch.tensor(self.current_round).type(torch.FloatTensor), dst=idx) # 모델 버전 전송
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

        printLog(f"Server >> CLIENT{client_idx}의 로컬 모델 평가 결과 : acc=>{test_accuracy}, test_loss=>{test_loss}")


