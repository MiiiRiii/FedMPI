from FedAvg import FedAvgServer
import torch

from utils.utils import printLog

import torch.distributed as dist

class CHAFLServer(FedAvgServer):
    def __init__(self):
        None

    def wait_local_update_of_selected_clients(self, currentRoundGroup, selected_clients_idx):
        reqs=[]

        for idx in selected_clients_idx:
            req=dist.irecv(tensor=torch.zeros(1))
            reqs.append(req)

        for req in reqs:
            req.wait()
            printLog(f"Server >> CLIENT {req.source_rank()}가 최소 할당량을 완료함")
        
        dist.broadcast(tensor=torch.tensor(float(1)), src=0, async_op=True, group=currentRoundGroup)