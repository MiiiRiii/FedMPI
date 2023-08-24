from utils.utils import *

import torch.distributed as dist
import torch
import wandb
import time

# Only implement pow-d, cpow-d
# Not implement rpow-d

class powerofchoice(object):

    def __init__(self, method, d):
        self.method = method
        self.d=d

    def calculate_coefficient(self, selected_client_idx, Server):

        coefficient={}
        sum=0
        for idx in selected_client_idx:
            coefficient[idx]=Server.len_local_dataset[idx]
            sum+=Server.len_local_dataset[idx]
        
        for idx in selected_client_idx:
            coefficient[idx]=coefficient[idx]/sum

        return coefficient
    
    def client_select_pow_d(self, d, num_selected_clients):
        local_loss_list={}
        local_loss=torch.zeros(1)

        for idx in range(d):
            req=dist.irecv(tensor=local_loss)
            req.wait()
            printLog(f"client {req.source_rank()}의 local loss는 {local_loss.item()}입니다.")
            local_loss_list[req.source_rank()] = local_loss.item()

        sorted_local_loss_list = sorted(local_loss_list.items(), key=lambda item:item[1], reverse=True)
        
        selected_clients_list=[]
        for idx in range(num_selected_clients):
            selected_clients_list.append(sorted_local_loss_list[idx][0])
        
        return selected_clients_list
    
    def runClient(self, Client):
        while True:

            # check if i am a candidate client
            candidate=False
            candidate_clients = torch.zeros(self.d).type(torch.int64)
            dist.broadcast(tensor=candidate_clients, src=0, group=Client.FLgroup)

            for idx in candidate_clients:
                if idx == Client.id:
                    candidate=True
                    break
            
            # when i am a candidate client
            if candidate :
                printLog(f"CLIENT {Client.id} >> I'm candidate group")
                candidate_clients_group = dist.new_group(candidate_clients.tolist()+[0])

                # get local loss
                Client.receive_global_model_from_server()
                if self.method =="pow_d":
                    local_loss = Client.evaluate()
                elif self.method == "cpow_d":
                    local_loss = Client.evaluate(method="cpow_d")
                
                
                req = dist.isend(tensor=torch.tensor([local_loss]), dst=0)
                req.wait()

                # check if i am the selected client
                selected=False
                selected_clients=torch.zeros(Client.num_selected_clients).type(torch.int64)
                dist.broadcast(tensor=selected_clients, src=0, group=candidate_clients_group)
                dist.destroy_process_group(candidate_clients_group)

                for idx in selected_clients:
                    if idx == Client.id:
                        selected=True
                        break

                # when i am a selected client
                if(selected):
                    Client.train()
                    printLog(f"CLIENT {Client.id} >> 평균 학습 소요 시간: {Client.total_train_time/Client.num_of_selected}")
                    Client.send_local_model_to_server()

            dist.barrier()

            continueFL = torch.zeros(1)
            dist.broadcast(tensor=continueFL, src=0, group=Client.FLgroup)

            if(continueFL[0]==0): #FL 종료
                break

    def runServer(self, Server):
        current_FL_start=time.time()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        while True:
            current_round_start=time.time()
            # candidate clients
            candidate_clients = random.sample(clients_idx, self.d)
            printLog(f"PS >> candidate group is {candidate_clients}")
            dist.broadcast(tensor=torch.tensor(candidate_clients), src=0, group=Server.FLgroup)
            candidate_clients_group = dist.new_group(candidate_clients+[0])

            # select clients
            Server.send_global_model_to_clients(candidate_clients)
            selected_client_idx = self.client_select_pow_d(self.d, int(Server.selection_ratio * Server.num_clients))
            printLog(f"PS >> 학습에 참여할 클라이언트는 {selected_client_idx}입니다.")
            dist.broadcast(tensor=torch.tensor(selected_client_idx), src=0, group=candidate_clients_group)
            dist.destroy_process_group(candidate_clients_group)
            
            # receive local models and aggregation
            Server.receive_local_model_from_selected_clients(selected_client_idx)
            coefficient=self.calculate_coefficient(selected_client_idx, Server)
            Server.average_aggregation(selected_client_idx, coefficient)
            global_acc, global_loss = Server.evaluate()
            Server.current_round+=1
            printLog(f"PS >> {Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})

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
               