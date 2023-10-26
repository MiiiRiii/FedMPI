from utils.utils import *

import wandb
import time
import threading

class SemiAsync(object):
    def __init__(self):
        None
    
    def remain_req(self, remain_reqs):
        for req in remain_reqs:
            req.wait()

    def runClient(self, Client):
        while True:
            Client.receive_global_model_from_server()
            Client.train()
            Client.send_local_model_to_server()
            
    def runServer(self, Server):
        current_FL_start = time.time()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        upload_success_client_idx=clients_idx # 학습 처음엔 모든 클라이언트에게 보냄
        num_local_model_limit=int(Server.num_clients*Server.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수

        # FL 프로세스 시작
        while True:
            Server.current_round+=1
            
            current_round_start=time.time()

            Server.send_global_model_to_clients(upload_success_client_idx)

            upload_success_client_idx, remain_reqs = Server.receive_local_model_from_any_clients(Server.num_clients, num_local_model_limit)

            #for idx in upload_success_client_idx:
                #Server.evaluate_local_model(idx)

            other_clients_sending = threading.Thread(target=self.remain_req, args=(remain_reqs,), daemon=True)
            other_clients_sending.start()

            Server.refine_received_local_model(upload_success_client_idx)            

            coefficient = Server.calculate_coefficient(upload_success_client_idx)
            Server.average_aggregation(upload_success_client_idx, coefficient)

            global_acc, global_loss = Server.evaluate()

            printLog(f"PS >> {Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})
            