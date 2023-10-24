from FedAvg import FedAvgServer

import torch.distributed as dist
import torch
import copy

from utils.model_utils import TensorBuffer
from utils.utils import printLog

class SemiAsyncServer(FedAvgServer):
    def __init__(self):
        None

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
    
    