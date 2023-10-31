from methods.FedAvg import FedAvgServer
import torch

from utils.utils import printLog

import torch.distributed as dist

class CHAFLServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)

    def wait_local_update_of_selected_clients(self, currentRoundGroup, selected_clients_idx):
        reqs=[]

        for idx in selected_clients_idx:
            req=dist.irecv(tensor=torch.zeros(1))
            reqs.append(req)

        for req in reqs:
            req.wait()
            printLog("SERVER", f"CLIENT {req.source_rank()}가 최소 할당량을 완료함")
        
        dist.broadcast(tensor=torch.tensor(float(1)), src=0, async_op=True, group=currentRoundGroup)

    def client_select_by_loss(self, num_clients, num_selected_clients, global_loss):
        cnt=0
        client_select_checklist=[False for i in range(num_clients+1)]
        selected_clients_list=[]
        local_loss=torch.zeros(1)
        local_loss_list={}
        remain_res=[]

        for idx in range(num_clients):
            req=dist.irecv(tensor=local_loss)
            if cnt<num_selected_clients :
                req.wait()
                local_loss_list[req.source_rank()]=local_loss.item()
                if local_loss.item()>global_loss:
                    selected_clients_list.append(req.source_rank())
                    client_select_checklist[req.source_rank()]=True
                    cnt=cnt+1
            else:
                remain_res.append(req)

        if cnt < num_selected_clients :
            sorted_local_loss_list = sorted(local_loss_list.items(), key=lambda item:item[1], reverse=True)
            for idx in range(num_clients):
                if cnt>=num_selected_clients:
                    break
                if client_select_checklist[sorted_local_loss_list[idx][0]] == False:
                    selected_clients_list.append(sorted_local_loss_list[idx][0])
                    client_select_checklist[sorted_local_loss_list[idx][0]]=True
                    cnt=cnt+1

        return selected_clients_list, remain_res
    
    def calculate_coefficient(self, selected_client_idx):
        selected_client_squared_local_epoch={}
        tensor=torch.zeros(1)
        sum_squared_local_epoch=0
        for idx in selected_client_idx:
            dist.recv(tensor=tensor, src=idx, tag=1)
            printLog("SERVER", f"CLIENT {idx}가 수행한 local epoch은 {tensor.item()}입니다.")
            selected_client_squared_local_epoch[idx]=(tensor.item())**2
            sum_squared_local_epoch+=(tensor.item())**2

        local_epoch_coefficient={}
        for idx in selected_client_idx:
            local_epoch_coefficient[idx]=(1-(selected_client_squared_local_epoch[idx]/sum_squared_local_epoch))/(int(self.selection_ratio * self.num_clients)-1)

        data_coefficient={}
        sum_local_data=0
        for idx in selected_client_idx:
            data_coefficient[idx]=self.len_local_dataset[idx]
            sum_local_data+=self.len_local_dataset[idx]
        for idx in selected_client_idx:
            data_coefficient[idx]/=sum_local_data
        
        coefficient={}
        for idx in selected_client_idx:
            coefficient[idx]=local_epoch_coefficient[idx]*0.7+data_coefficient[idx]*0.3

        return local_epoch_coefficient