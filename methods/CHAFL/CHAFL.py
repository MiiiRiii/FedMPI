from utils.utils import *

import torch.distributed as dist
import torch
import wandb
import time

class CHAFL(object):
    def __init__(self):
        None
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

    def calculate_coefficient(self, selected_client_idx, Server):
        selected_client_squared_local_epoch={}
        tensor=torch.zeros(1)
        sum_squared_local_epoch=0
        for idx in selected_client_idx:
            dist.recv(tensor=tensor, src=idx, tag=1)
            printLog(f"PS >> CLIENT {idx}가 수행한 local epoch은 {tensor.item()}입니다.")
            selected_client_squared_local_epoch[idx]=(tensor.item())**2
            sum_squared_local_epoch+=(tensor.item())**2

        local_epoch_coefficient={}
        for idx in selected_client_idx:
            local_epoch_coefficient[idx]=(1-(selected_client_squared_local_epoch[idx]/sum_squared_local_epoch))/(int(Server.selection_ratio * Server.num_clients)-1)

        data_coefficient={}
        sum_local_data=0
        for idx in selected_client_idx:
            data_coefficient[idx]=Server.len_local_dataset[idx]
            sum_local_data+=Server.len_local_dataset[idx]
        for idx in selected_client_idx:
            data_coefficient[idx]/=sum_local_data
        
        coefficient={}
        for idx in selected_client_idx:
            coefficient[idx]=local_epoch_coefficient[idx]*0.7+data_coefficient[idx]*0.3

        return local_epoch_coefficient


    def runClient(self, Client):

        while True:
            Client.receive_global_model_from_server()
            local_loss = Client.evaluate()
            printLog(f"CLIENT {Client.id} >> local loss는 {round(local_loss,4)}입니다.")
            req = dist.isend(tensor=torch.tensor([local_loss]), dst=0)
            req.wait()

            selected=False
            selected_clients=torch.zeros(Client.num_selected_clients).type(torch.int64)
            dist.broadcast(tensor=selected_clients, src=0, group=Client.FLgroup)
            
            for idx in selected_clients:
                if idx == Client.id:
                    selected=True
                    break

            current_round_group=selected_clients.tolist()+[0]
            currentRoundGroup = dist.new_group(current_round_group)
            if(selected):
                                               
                # Do minimal quota
                real_local_epoch = Client.train(currentRoundGroup)
                
                # Send local model to server
                Client.send_local_model_to_server()
                dist.send(torch.tensor([float(real_local_epoch)]), dst=0, tag=1)
                dist.destroy_process_group(currentRoundGroup)

            dist.barrier()

            continueFL = torch.zeros(1)
            dist.broadcast(tensor=continueFL, src=0, group=Client.FLgroup)

            if(continueFL[0]==0): #FL 종료
                break
    
    def runServer(self, Server):
        current_FL_start=time.time()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        random_clients_idx=clients_idx[:]
        global_acc, global_loss = Server.evaluate()
        printLog(f"PS >> 초기 글로벌 모델의 loss는 {round(global_loss,4)}입니다.")
        while True:
            current_round_start=time.time()
            random.shuffle(random_clients_idx)
            Server.send_global_model_to_clients(random_clients_idx)

            selected_client_idx, remain_reqs = self.client_select_by_loss(Server.num_clients, int(Server.selection_ratio * Server.num_clients), global_loss)
            printLog(f"PS >> 학습에 참여하는 클라이언트는 {selected_client_idx}입니다.")
            dist.broadcast(tensor=torch.tensor(selected_client_idx), src=0, group=Server.FLgroup)
            if len(remain_reqs)>0:
                for req in remain_reqs:
                    req.wait()

            current_round_group = selected_client_idx+[0]
            currentRoundGroup = dist.new_group(current_round_group, backend="gloo")            
            Server.wait_local_update_of_selected_clients(currentRoundGroup, selected_client_idx)
            Server.receive_local_model_from_selected_clients(selected_client_idx)
            dist.destroy_process_group(currentRoundGroup)

            coefficient = self.calculate_coefficient(selected_client_idx, Server)
            for idx in selected_client_idx:
                printLog(f"PS >> CLIENT {idx}의 coefficient는 {coefficient[idx]}입니다.")

            Server.average_aggregation(selected_client_idx, coefficient)
            global_acc, global_loss = Server.evaluate()
            Server.current_round+=1

            printLog(f"PS >> {Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60 })

            dist.barrier()
            
            if global_acc>=Server.target_accuracy:
                dist.broadcast(tensor=torch.tensor([0.]), src=0, group=Server.FLgroup)
                printLog(f"PS >> 목표한 정확도에 도달했으며, 수행한 라운드 수는 {Server.current_round}회 입니다.")
                break
            elif Server.current_round == Server.target_rounds:
                dist.broadcast(tensor=torch.tensor([0.]), src=0, group=Server.FLgroup)
                printLog(f"PS >> 목표한 라운드 수에 도달했으며, 최종 정확도는 {round(global_acc*100,4)}% 입니다.")
                break
            else:
                printLog(f"PS >> 다음 라운드를 수행합니다.")
                dist.broadcast(tensor=torch.tensor([1.]), src=0, group=Server.FLgroup)
               
