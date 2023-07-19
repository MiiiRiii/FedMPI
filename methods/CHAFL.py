from utils.utils import *

import torch.distributed as dist
import torch
import wandb

class CHAFL(object):
    def __init__(self):
        None

    def calculate_coefficient(self, selected_client_idx, Server):
        selected_client_squared_local_epoch={}
        tensor=torch.zeros(1)
        sum_squared_local_epoch=0
        for idx in selected_client_idx:
            dist.recv(tensor=tensor, src=idx)
            selected_client_squared_local_epoch[idx]=(tensor.item())**2
            sum_squared_local_epoch+=tensor.item()

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
            coefficient[idx]=(local_epoch_coefficient[idx]+data_coefficient[idx])/2

        return coefficient

    def runClient(self, Client):

        Client.receive_num_local_epoch_from_server()
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

            if(selected):
                local_epoch = Client.train()
                printLog(f"CLIENT {Client.id} >> 평균 학습 소요 시간: {Client.total_train_time/Client.num_of_selected}")
                Client.send_local_model_to_server()
                dist.send(torch.tensor([float(local_epoch)]), dst=0)

            dist.barrier()

            continueFL = torch.zeros(1)
            dist.broadcast(tensor=continueFL, src=0, group=Client.FLgroup)

            if(continueFL[0]==0): #FL 종료
                break
    
    def runServer(self, Server):
        Server.send_num_local_epoch_to_clients()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        global_acc, global_loss = Server.evaluate()
        printLog(f"PS >> 초기 글로벌 모델의 loss는 {round(global_loss,4)}입니다.")
        while True:
            Server.send_global_model_to_clients(clients_idx)
            selected_client_idx = client_select_by_loss(Server.num_clients, clients_idx, int(Server.selection_ratio * Server.num_clients), global_loss)
            printLog(f"PS >> 학습에 참여하는 클라이언트는 {selected_client_idx}입니다.")
            dist.broadcast(tensor=torch.tensor(selected_client_idx), src=0, group=Server.FLgroup)

            Server.receive_local_model_from_selected_clients(selected_client_idx)

            coefficient = self.calculate_coefficient(selected_client_idx, Server)

            Server.average_aggregation(selected_client_idx, coefficient)
            global_acc, global_loss = Server.evaluate()
            Server.current_round+=1
            printLog(f"PS >> {Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4)})

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
               
