from utils.utils import *

import torch.distributed as dist
import torch
import wandb
import time
import threading

class FedAsync(object):
    def __init__(self):
        None
    
    def runClient(self, Client):
        while True:
            isTerminate = Client.receive_global_model_from_server()
            if isTerminate == 0:
                printLog(f"CLIENT{Client.id}", "FL 프로세스를 종료합니다.")
                break
            Client.train()
            Client.send_local_model_to_server()

        dist.barrier()

    def runServer(self, Server):
        current_FL_start=time.time()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        Server.send_global_model_to_clients(clients_idx)
        
        listen_local_update = threading.Thread(target=Server.receive_local_model_from_any_clients, args=(), daemon=True)
        listen_local_update.start()

        while True:
            Server.current_round+=1
            
            current_round_start = time.time()
            
            sent_client_idx = Server.wait_until_can_update_global_model()
            Server.global_update(sent_client_idx)
            
            global_acc, global_loss = Server.evaluate()
            
            printLog("SERVER", f"{Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            
            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})

            if global_acc>=Server.target_accuracy:
                printLog("SERVER", f"목표한 정확도에 도달했으며, 수행한 라운드 수는 {Server.current_round}회 입니다.")
                break   
            elif Server.current_round == Server.target_rounds:
                printLog("SERVER", f"목표한 라운드 수에 도달했으며, 최종 정확도는 {round(global_acc*100,4)}% 입니다.")
                break
            else:
                Server.send_global_model_to_client(sent_client_idx)
        
        Server.terminate_FL(sent_client_idx)
        printLog(f"SERVER", "FL 프로세스를 종료합니다.")
        dist.barrier()

            