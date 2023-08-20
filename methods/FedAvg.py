from utils.utils import *

import torch.distributed as dist
import torch
import wandb
import time

class FedAvg(object):

    def __init__(self):
        None

    def calculate_coefficient(self, selected_client_idx, Server):

        coefficient={}
        sum=0
        for idx in selected_client_idx:
            coefficient[idx]=Server.len_local_dataset[idx]
            sum+=Server.len_local_dataset[idx]
        
        for idx in selected_client_idx:
            coefficient[idx]=coefficient[idx]/sum

        return coefficient
    
    def client_select_randomly(self, clients_idx, num_selected_clients):
        shuffled_clients_idx = clients_idx[:]
        random.shuffle(shuffled_clients_idx)
        return shuffled_clients_idx[0:num_selected_clients]
    
    def runClient(self, Client):
        while True:
            selected=False
            selected_clients=torch.zeros(Client.num_selected_clients).type(torch.int64)
            dist.broadcast(tensor=selected_clients, src=0, group=Client.FLgroup)
            
            for idx in selected_clients:
                if idx == Client.id:
                    selected=True
                    break

            if(selected):
                Client.receive_global_model_from_server()
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
            selected_client_idx = self.client_select_randomly(clients_idx, int(Server.selection_ratio * Server.num_clients))
            printLog(f"PS >> 학습에 참여할 클라이언트는 {selected_client_idx}입니다.")
            dist.broadcast(tensor=torch.tensor(selected_client_idx), src=0, group=Server.FLgroup)

            Server.send_global_model_to_clients(selected_client_idx)
            
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
               