from methods.FedAvg import FedAvgServer
from utils.model_utils import TensorBuffer
from utils.utils import printLog

import torch.distributed as dist
import torch
import copy


from collections import OrderedDict

class SemiAsyncServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)

    def receive_local_model_from_any_clients(self, num_clients, num_local_model_limit):
        upload_success_client_idx=[]
        remain_res=[]
        cnt=0
        for idx in range(num_clients):
            temp_local_model =TensorBuffer(list(self.model.state_dict().values()))
            req = dist.irecv(tensor=temp_local_model)
            if cnt < num_local_model_limit:
                req.wait()
                self.flatten_client_models[req.source_rank()] = copy.deepcopy(temp_local_model)
                printLog(f"Server >> CLIENT {req.source_rank()}에게 local model을 받음")
                cnt=cnt+1
                upload_success_client_idx.append(req.source_rank())
            else:
                remain_res.append(req)
        
        return upload_success_client_idx, remain_res
    
    def refine_received_local_model(self, upload_success_client_idx, local_model_version):
        decay_coefficient = 0.9

        for idx in upload_success_client_idx:
            local_model_version[idx] += local_model_version[idx]
            local_coefficient = decay_coefficient**(local_model_version[idx])
            
            local_model = self.model_controller.Model()
            local_model_state_dict=local_model.state_dict()
            self.flatten_client_models[idx].unpack(local_model_state_dict.values())

            global_model_state_dict = self.model.state_dict()

            interpolated_weights = OrderedDict()

            for key in global_model_state_dict.keys():
                interpolated_weights[key] = local_coefficient * local_model_state_dict[key]
            for key in global_model_state_dict.keys():
                interpolated_weights[key] += (1-local_coefficient) * global_model_state_dict[key]
            
            self.flatten_client_models[idx].load_state_dict(interpolated_weights)

    
    def calculate_coefficient(upload_success_client_idx, Server):

        coefficient={}
        for idx in upload_success_client_idx : 
            coefficient[idx] = Server.len_local_dataset[idx] / Server.len_total_local_dataset
        
        return coefficient
    
    def send_global_model_to_clients(self, sucess_uploaded_client_idx):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        for idx in sucess_uploaded_client_idx:
            dist.send(tensor=flatten_model.buffer, dst=idx) # 글로벌 모델 전송
            dist.send(tensor=torch.tensor(self.current_round+1).type(torch.FloatTensor), dst=idx) # 모델 버전 전송