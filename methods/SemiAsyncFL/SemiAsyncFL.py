from utils.utils import *
from collections import OrderedDict

import torch.distributed as dist
import torch
import wandb
import time

class SemiAsyncFL(object):
    def __init__(self):
        None
    
    def refine_received_local_model(self, Server, upload_success_client_idx, local_model_version):
        decay_coefficient = 0.9

        for idx in upload_success_client_idx:
            local_model_version[idx] += local_model_version[idx]
            local_coefficient = decay_coefficient**(local_model_version[idx])
            
            local_model = Server.model_controller.Model()
            local_model_state_dict=local_model.state_dict()
            Server.flatten_client_models[idx].unpack(local_model_state_dict.values())

            global_model_state_dict = Server.model.state_dict()

            interpolated_weights = OrderedDict()

            for key in global_model_state_dict.keys():
                interpolated_weights[key] = local_coefficient * local_model_state_dict[key]
            for key in global_model_state_dict.keys():
                interpolated_weights[key] += (1-local_coefficient) * global_model_state_dict[key]
            
            Server.flatten_client_models[idx].load_state_dict(interpolated_weights)

    def calculate_coefficient(upload_success_client_idx, Server):

        coefficient={}
        sum=0
        for idx in upload_success_client_idx : 
            coefficient[idx] = Server.len_local_dataset[idx] / Server.len_total_local_dataset
        
        return coefficient



    def runClient(self, Client):
        while True:
            Client.receive_global_model_from_server()
            Client.train()
            Client.increase_model_version()
            printLog(f"CLIENT {Client.id} >> 평균 학습 소요 시간: {Client.total_train_time/Client.num_of_selected}")
            Client.send_local_model_to_server()
            
    def runServer(self, Server):
        current_FL_start = time.time()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        local_model_version = [0 for idx in range(0,Server.num_clients)] 

        num_local_model_limit=int(Server.num_clients*0.1)
        # 학습 처음에 global model을 모든 클라이언트에게 보냄
        Server.send_global_model_to_clients(clients_idx)

        # FL 프로세스 시작
        while True:
            current_round_start=time.time()
            upload_success_client_idx, remain_reqs = Server.receive_local_model_from_any_clients(Server.num_clients, num_local_model_limit)
            if len(remain_reqs) :
                for req in remain_reqs:
                    req.wait()
            
            self.refine_received_local_model(upload_success_client_idx, local_model_version)
            coefficient = self.calculate_coefficient(upload_success_client_idx, Server)
            Server.average_aggregation(upload_success_client_idx, coefficient)

            global_acc, global_loss = Server.evaluate()

            Server.current_round+=1

            printLog(f"PS >> {Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})
            