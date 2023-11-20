from utils.utils import *

import wandb
import time
import threading

class SemiAsyncPM1(object):
    def __init__(self):
        None
    
    def runClient(self, Client):
        is_ongoing_local_update_flag = threading.Event() # 로컬 업데이트가 현재 진행중인지
        is_ongoing_local_update_flag.clear()

        terminate_FL_flag = threading.Event()
        terminate_FL_flag.clear()

        listen_global_model = threading.Thread(target=Client.receive_global_model_from_server, args=(is_ongoing_local_update_flag, terminate_FL_flag,), daemon=True)
        listen_global_model.start()
        while True:
            if terminate_FL_flag.is_set():
                printLog(f"CLIENT {Client.id}", "FL 프로세스를 종료합니다.")
                break
            if is_ongoing_local_update_flag.is_set():
                utility = Client.train(terminate_FL_flag)
                
                if terminate_FL_flag.is_set() or utility == -1:
                    printLog(f"CLIENT {Client.id}", "FL 프로세스를 종료합니다.")
                    break
                Client.send_local_model_to_server(utility)
                is_ongoing_local_update_flag.clear()

        Client.terminate()
            
    def runServer(self, Server):
        
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        picked_client_idx =clients_idx # 학습 처음엔 모든 클라이언트에게 보냄
        num_local_model_limit=int(Server.num_clients*Server.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수
        
        listen_local_update = threading.Thread(target=Server.receive_local_model_from_any_clients, args=(), daemon=True)
        
        global_acc, global_loss = Server.evaluate()

        current_FL_start = time.time()
        
        listen_local_update.start()

        # FL 프로세스 시작
        while True:
            Server.current_round+=1
            
            current_round_start=time.time()

            Server.send_global_model_to_clients(clients_idx, picked_client_idx, global_loss)

            picked_client_idx  = Server.wait_until_can_update_global_model(num_local_model_limit)

            Server.refine_received_local_model(picked_client_idx )            

            coefficient = Server.calculate_coefficient(picked_client_idx )
            Server.average_aggregation(picked_client_idx , coefficient)

            global_acc, global_loss = Server.evaluate()
            
            picked_client_info=""
            for idx, client_idx in enumerate(picked_client_idx):
                picked_client_info += f"                                 => 클라이언트{client_idx}의 staleness: {Server.current_round - Server.local_model_version[client_idx]}, 로컬 utility: {Server.local_utility[client_idx]}\n"
            printLog("SERVER", f"{Server.current_round}번째 글로벌 모델 \n\
                                 => test_accuracy: {round(global_acc*100,4)}% \n\
                                 => test_loss: {round(global_loss,4)}\n\{picked_client_info}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})
            
            if global_acc>=Server.target_accuracy:
                printLog("SERVER", f"목표한 정확도에 도달했으며, 수행한 라운드 수는 {Server.current_round}회 입니다.")
                printLog("SERVER", "마무리 중입니다..")
                break
            
            elif Server.current_round == Server.target_rounds:
                printLog(f"SERVER", f"목표한 라운드 수에 도달했으며, 최종 정확도는 {round(global_acc*100,4)}% 입니다.")
                printLog(f"SERVER", "마무리 중입니다..")
                break
        
        Server.terminate(clients_idx)
        printLog(f"SERVER", "FL 프로세스를 종료합니다.")